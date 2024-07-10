<div align="center">

<h1>VEnhancer: Generative Space-Time Enhancement<br>for Video Generation</h1>

<div>
    <a href='https://scholar.google.com/citations?user=GUxrycUAAAAJ&hl=zh-CN' target='_blank'>Jingwen He</a>,&emsp;
    <a href='https://tianfan.info' target='_blank'>Tianfan Xue</a>,&emsp;
    <a href='https://github.com/ChrisLiu6' target='_blank'>Dongyang Liu</a>,&emsp;
    <a href='https://github.com/0x3f3f3f3fun' target='_blank'>Xinqi Lin</a>,&emsp;
    <a href='https://gaopengcuhk.github.io' target='_blank'>Peng Gao</a>,&emsp;
</div>
    <a href='https://scholar.google.com/citations?user=GMzzRRUAAAAJ&hl=en' target='_blank'>Dahua Lin</a>,&emsp;
    <a href='https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=en' target='_blank'>Yu Qiao</a>,&emsp;
    <a href='https://wlouyang.github.io' target='_blank'>Wanli Ouyang</a>,&emsp;
    <a href='https://liuziwei7.github.io' target='_blank'>Ziwei Liu</a>
<div>
</div>
<div>
    The Chinese University of Hong Kong,&emsp;Shanghai Artificial Intelligence Laboratory,&emsp; 
</div>
<div>
    
</div>
<div>
 S-Lab, Nanyang Technological University&emsp; 
</div>

<div>
    <h4 align="center">
        <a href="https://vchitect.github.io/VEnhancer-project/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2312.06640-b31b1b.svg">
        </a>
        <a href="https://youtu.be/QMR_5weifGg" target='_blank'>
        <img src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white">
        <!-- </a>
        <img src="https://api.infinitescript.com/badgen/count?name="> -->
    </h4>
</div>

<strong>VEnhancer, a generative space-time enhancement framework that can improve the existing T2V results. </strong>

<table class="center">
  <tr>
    <td colspan="1">VideoCrafter2</td>
    <td colspan="1">+VEnhancer</td>
  </tr>
  <tr>
  <td>
    <img src=assets/input_raccoon_4.gif width="380">
  </td>
  <td>
    <img src=assets/out_raccoon_4.gif width="380">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/input_fish.gif width="380">
  </td>
  <td>
    <img src=assets/out_fish.gif width="380">
  </td>
  </tr>

  

</table>

:open_book: For more visual results, go checkout our <a href="https://vchitect.github.io/VEnhancer-project/" target="_blank">project page</a>


---

</div>


## 🔥 Update
- [2024.07] This repo is created.

## 🎬 Overview
The architecture of VEnhancer. It follows ControlNet and copies the architecures and weights of  multi-frame  encoder and middle block of a pretrained video diffusion model to build a trainable condition network. 
This video ControlNet accepts low-resolution key frames as well as full frames of noisy latents as inputs. 
Also, the noise level $\sigma$ regarding noise augmentation and downscaling factor $s$ serve as additional network conditioning apart from timestep $t$ and prompt $c_{text}$. 
![overall_structure](assets/venhancer_arch.png)


## BibTeX
If you use our work in your research, please cite our publication:
```
@misc{he2024venhancer,
      title={VEnhancer: Generative Space-Time Enhancement for Video Generation}, 
      author={Jingwen He and Tianfan Xue and Dongyang Liu and Xinqi Lin and Peng Gao and Dahua Lin and Yu Qiao and Wanli Ouyang and Ziwei Liu},
      year={2024},
      eprint={2308.15070},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 📧 Contact
If you have any questions, please feel free to reach us at `hejingwenhejingwen@outlook.com`.
