import os
import os.path as osp
import random
from typing import Any, Dict

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

from video_to_video.modules import *
from video_to_video.utils.config import cfg
from video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion
from video_to_video.diffusion.schedules_sdedit import noise_schedule
from video_to_video.utils.logger import get_logger

logger = get_logger()


class VideoToVideo():
    def __init__(self, opt):
        self.opt = opt
        cfg.model_path = opt.model_path
        cfg.embedder.pretrained = osp.join(opt.model_dir, opt.ckpt_clip)
        self.device = torch.device(f'cuda:0')
        clip_encoder = FrozenOpenCLIPEmbedder(
            pretrained=cfg.embedder.pretrained, device=self.device)
        clip_encoder.model.to(self.device)
        self.clip_encoder = clip_encoder
        logger.info(f'Build encoder with {cfg.embedder.type}')

        generator = ControlledV2VUNet()
        generator = generator.to(self.device)
        generator.eval()

        load_dict = torch.load(cfg.model_path, map_location='cpu')
        if 'state_dict' in load_dict:
            load_dict = load_dict['state_dict']
        ret = generator.load_state_dict(load_dict, strict=True)
        
        self.generator = generator.half()
        logger.info('Load model path {}, with local status {}'.format(cfg.model_path, ret))

        sigmas = noise_schedule(
            schedule='logsnr_cosine_interp',
            n=1000,
            zero_terminal_snr=True,
            scale_min=2.0,
            scale_max=4.0)
        diffusion = GaussianDiffusion(sigmas=sigmas)
        self.diffusion = diffusion
        logger.info('Build diffusion with GaussianDiffusion')

        cfg.auto_encoder.pretrained = osp.join(opt.model_dir, opt.ckpt_autoencoder)
        autoencoder = AutoencoderKL(**cfg.auto_encoder)
        autoencoder.eval()
        for param in autoencoder.parameters():
            param.requires_grad = False
        autoencoder.to(self.device)
        self.autoencoder = autoencoder
        torch.cuda.empty_cache()

        self.negative_prompt = cfg.negative_prompt
        self.positive_prompt = cfg.positive_prompt

        negative_y = clip_encoder(self.negative_prompt).detach()
        self.negative_y = negative_y


    def test(self, input: Dict[str, Any], total_noise_levels=1000, \
                 steps=50, guide_scale=7.5, noise_aug=200):
        video_data = input['video_data']
        y = input['y']
        mask_cond = input['mask_cond']
        up_scale = input['up_scale']
        interp_f_num = input['interp_f_num']

        _, _, h, w = video_data.shape
        if h*up_scale < 720:
            up_scale *= 720/(h*up_scale)
            logger.info(f'changing up_scale to: {up_scale}')
        video_data = F.interpolate(video_data, scale_factor=up_scale, mode='bilinear')

        key_f_num = len(video_data)
        aug_video = []
        for i in range(key_f_num):
            if i == key_f_num - 1:
                aug_video.append(video_data[i:i+1])
            else:
                aug_video.append(video_data[i:i+1].repeat(interp_f_num+1, 1, 1, 1))
        video_data = torch.concat(aug_video, dim=0)

        logger.info(f'video_data shape: {video_data.shape}')
        frames_num, _, h, w = video_data.shape

        padding = pad_to_fit(h, w)
        video_data = F.pad(video_data, padding, 'constant', 1)

        video_data = video_data.unsqueeze(0)
        bs = 1
        video_data = video_data.to(self.device)

        mask_cond = mask_cond.unsqueeze(0).to(self.device)
        up_scale = torch.LongTensor([up_scale]).to(self.device)

        video_data = rearrange(video_data, 'b f c h w -> (b f) c h w')
        video_data_list = torch.chunk(
            video_data, video_data.shape[0] // 1, dim=0)
        with torch.no_grad():
            decode_data = []
            for vd_data in video_data_list:
                encoder_posterior = self.autoencoder.encode(vd_data)
                tmp = get_first_stage_encoding(encoder_posterior).detach()
                decode_data.append(tmp)
            video_data_feature = torch.cat(decode_data, dim=0)
            video_data_feature = rearrange(
                video_data_feature, '(b f) c h w -> b c f h w', b=bs)
        torch.cuda.empty_cache()

        y = self.clip_encoder(y).detach() 

        with amp.autocast(enabled=True):

            t_hint = torch.LongTensor([noise_aug-1]).to(self.device)
            video_in_low_fps = video_data_feature[:,:,::interp_f_num+1].clone()
            noised_hint = self.diffusion.diffuse(video_in_low_fps, t_hint)

            t = torch.LongTensor([total_noise_levels-1]).to(self.device)
            noised_lr = self.diffusion.diffuse(video_data_feature, t)
            
            model_kwargs = [{'y': y}, {'y': self.negative_y}]
            model_kwargs.append({'hint': noised_hint})
            model_kwargs.append({'mask_cond': mask_cond})
            model_kwargs.append({'up_scale': up_scale})
            model_kwargs.append({'t_hint': t_hint})

            solver = 'dpmpp_2m_sde' # 'heun' | 'dpmpp_2m_sde' 
            gen_vid = self.diffusion.sample(
                noise=noised_lr,
                model=self.generator,
                model_kwargs=model_kwargs,
                guide_scale=guide_scale,
                guide_rescale=0.2,
                solver=solver,
                steps=steps,
                t_max=total_noise_levels - 1,
                t_min=0,
                discretization='trailing')
            torch.cuda.empty_cache()

            scale_factor = 0.18215
            vid_tensor_feature = 1. / scale_factor * gen_vid
            vid_tensor_feature = rearrange(vid_tensor_feature,
                                           'b c f h w -> (b f) c h w')
            vid_tensor_feature_list = torch.chunk(
                vid_tensor_feature, vid_tensor_feature.shape[0] // 2, dim=0)
            decode_data = []
            for vd_data in vid_tensor_feature_list:
                tmp = self.autoencoder.decode(vd_data)
                decode_data.append(tmp)
            vid_tensor_gen = torch.cat(decode_data, dim=0)

        w1, w2, h1, h2 = padding
        vid_tensor_gen = vid_tensor_gen[:,:,h1:h+h1,w1:w+w1]

        gen_video = rearrange(
            vid_tensor_gen, '(b f) c h w -> b c f h w', b=bs)

        logger.info(f'sampling, finished.')

        return gen_video.type(torch.float32).cpu()


def pad_to_fit(h, w):
    BEST_H, BEST_W = 720, 1280

    if h < BEST_H:
        h1, h2 = _create_pad(h, BEST_H)
    elif h == BEST_H:
        h1 = h2 = 0
    else: 
        h1 = 0
        h2 = int((h + 48) // 64 * 64) + 64 - 48 - h

    if w < BEST_W:
        w1, w2 = _create_pad(w, BEST_W)
    elif w == BEST_W:
        w1 = w2 = 0
    else:
        w1 = 0
        w2 = int(w // 64 * 64) + 64 - w
    return (w1, w2, h1, h2)

def _create_pad(h, max_len):
    h1 = int((max_len - h) // 2)
    h2 = max_len - h1 - h
    return h1, h2