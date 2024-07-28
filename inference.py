import os
import subprocess
import tempfile
import cv2
import torch
from argparse import ArgumentParser, Namespace
from PIL import Image
from typing import Any, Dict, List, Mapping, Tuple
from easydict import EasyDict
from einops import rearrange

from video_to_video.video_to_video_model import VideoToVideo
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as transforms_F

from video_to_video.utils.seed import setup_seed
from video_to_video.utils.logger import get_logger

logger = get_logger()


def tensor2vid(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)

    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    video = video * 255.0

    images = rearrange(video, 'b c f h w -> b f h w c')[0]
    return images


def preprocess(input_frames):
    out_frame_list = []
    for pointer in range(len(input_frames)):
        frame = input_frames[pointer]
        frame = convert_to_img(frame)
        frame = transforms_F.to_tensor(frame)
        out_frame_list.append(frame)
    
    out_frames = torch.stack(out_frame_list, dim=0)
    out_frames.clamp_(0, 1)
    mean = out_frames.new_tensor([0.5, 0.5, 0.5]).view(-1)
    std = out_frames.new_tensor([0.5, 0.5, 0.5]).view(-1)
    out_frames.sub_(mean.view(1, -1, 1, 1)).div_(std.view(1, -1, 1, 1))
    return out_frames


def convert_to_img(input):
    if len(input.shape) == 2:
        img = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
    img = input[:, :, ::-1]
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    return img


def load_video(vid_path, load_frame_num=48, start_frame=0):
    capture = cv2.VideoCapture(vid_path)
    _fps = capture.get(cv2.CAP_PROP_FPS)
    _total_frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    pointer = 0
    frame_list = []
    stride = 1
    while len(frame_list) < load_frame_num:
        ret, frame = capture.read()
        pointer += 1
        if (not ret) or (frame is None):
            break
        if pointer < start_frame:
            continue
        if pointer >= _total_frame_num + 1:
            break
        if (pointer - start_frame) % stride == 0:
            frame_list.append(frame)
    capture.release()
    return frame_list, _fps


def save_video(video, save_dir, file_name, fps=16.0):
    output_path = os.path.join(save_dir, file_name)
    images = [(img.numpy()).astype('uint8') for img in video]
    temp_dir = tempfile.mkdtemp()
    for fid, frame in enumerate(images):
        tpth = os.path.join(temp_dir, '%06d.png' % (fid + 1))
        cv2.imwrite(tpth, frame[:, :, ::-1],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
    tmp_path = os.path.join(save_dir, 'tmp.mp4')
    cmd = f'ffmpeg -y -f image2 -framerate {fps} -i {temp_dir}/%06d.png \
     -vcodec libx264 -crf 17 -pix_fmt yuv420p {tmp_path}'

    status, output = subprocess.getstatusoutput(cmd)
    if status != 0:
        logger.error('Save Video Error with {}'.format(output))
    os.system(f'rm -rf {temp_dir}')
    os.rename(tmp_path, output_path)


def collate_fn(data, device):
    """Prepare the input just before the forward function.
    This method will move the tensors to the right device.
    Usually this method does not need to be overridden.

    Args:
        data: The data out of the dataloader.
        device: The device to move data to.

    Returns: The processed data.

    """
    from torch.utils.data.dataloader import default_collate

    def get_class_name(obj):
        return obj.__class__.__name__

    if isinstance(data, dict) or isinstance(data, Mapping):
        return type(data)({
            k: collate_fn(v, device) if k != 'img_metas' else v
            for k, v in data.items()
        })
    elif isinstance(data, (tuple, list)):
        if 0 == len(data):
            return torch.Tensor([])
        if isinstance(data[0], (int, float)):
            return default_collate(data).to(device)
        else:
            return type(data)(collate_fn(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        if data.dtype.type is np.str_:
            return data
        else:
            return collate_fn(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (bytes, str, int, float, bool, type(None))):
        return data
    else:
        raise ValueError(f'Unsupported data type {type(data)}')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    parser.add_argument("--input_path", required=True, type=str, help="input video path")
    parser.add_argument("--save_dir", type=str, default='results', help="save directory")
    parser.add_argument("--model_path", required=True, type=str, help="model path")
    parser.add_argument("--prompt", type=str, default='a good video', help="prompt")

    parser.add_argument("--total_noise_levels", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=7.5)
    
    parser.add_argument("--noise_aug", type=int, default=200, help='noise augmentation')
    parser.add_argument("--interp_f_num", type=int, default=0)
    parser.add_argument("--target_fps", type=int, default=0)
    parser.add_argument("--up_scale", type=float, default=4)

    parser.add_argument("--max_frame_num", type=int, default=48)
    parser.add_argument("--start_frame", type=int, default=0)

    return parser.parse_args()


def main():
    
    args = parse_args()

    input_path = args.input_path
    prompt = args.prompt
    model_path = args.model_path
    save_dir = args.save_dir

    noise_aug = args.noise_aug
    up_scale = args.up_scale
    interp_f_num = args.interp_f_num
    target_fps = args.target_fps

    max_frame_num = args.max_frame_num
    start_frame = args.start_frame

    total_noise_levels = args.total_noise_levels
    steps = args.steps
    guide_scale = args.cfg

    model_cfg = EasyDict(__name__='model_cfg')
    model_cfg.model_dir = 'ckpts'
    model_cfg.model_path = model_path
    model_cfg.ckpt_clip = 'open_clip_pytorch_model.bin' 
    model_cfg.ckpt_autoencoder = 'v2-1_512-ema-pruned.ckpt'
    
    model = VideoToVideo(model_cfg)

    os.makedirs(save_dir, exist_ok=True)
    logger.info(f'input video path: {input_path}')
    text = prompt
    logger.info(f'text: {text}')

    input_frames, input_fps = load_video(input_path, max_frame_num, start_frame)
    in_f_num = len(input_frames)
    logger.info(f'input fps: {input_fps}')
    logger.info(f'input frames length: {in_f_num}')
    if target_fps > 0:
        interp_f_num = max(round(target_fps/input_fps)-1, 0)
    target_fps = input_fps * (interp_f_num+1)
    logger.info(f'target_fps: {target_fps}')
    if in_f_num + interp_f_num * (in_f_num - 1) > max_frame_num:
        in_f_num = max_frame_num // (1+interp_f_num)
        logger.info(f'input frames length after: {in_f_num}')

    video_data = preprocess(input_frames[:in_f_num])
    
    caption = text + model.positive_prompt

    mask_cond = []
    interp_cond = [1 for _ in range(interp_f_num)]
    for i in range(in_f_num):
        mask_cond.append(0)
        if i != in_f_num - 1:
            mask_cond += interp_cond
    logger.info(f'mask_cond: {mask_cond}')
    mask_cond = torch.Tensor(mask_cond).long()

    pre_data = {'video_data': video_data, 'y': caption}
    pre_data['mask_cond'] = mask_cond
    pre_data['up_scale'] = up_scale
    pre_data['interp_f_num'] = interp_f_num
    pre_data['t_hint'] = noise_aug
    
    setup_seed(666)

    with torch.no_grad():
        data_tensor = collate_fn(pre_data, 'cuda:0')
        output = model.test(data_tensor, total_noise_levels, steps=steps, \
                             guide_scale=guide_scale, noise_aug=noise_aug)

    output = tensor2vid(output)
    file_name = f'{text}.mp4'
    save_video(output, save_dir, file_name, fps=target_fps)

if __name__ == '__main__':
    main()