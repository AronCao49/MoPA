# MoPA
[**MoPA: Multi-Modal Prior Aided Domain Adaptation for 3D Semantic Segmentation**][PaperLink]
<br>
[Haozhi Cao](https://scholar.google.com/citations?user=EaRJECUAAAAJ&hl)<sup>1,\*</sup>,
[Yuecong Xu](https://xuyu0010.wixsite.com/xuyu0010)<sup>2</sup>,
[Jianfei Yang](https://marsyang.site/)<sup>2</sup>,
[Pengyu Yin](https://pamphlett.github.io/)<sup>1</sup>,
[Shenghai Yuan](https://scholar.google.com/citations?user=XcV_sesAAAAJ&hl=en)<sup>1</sup>,
[Lihua Xie](https://scholar.google.com.sg/citations?user=Fmrv3J8AAAAJ&hl=en)<sup>1</sup>
<br>
### [[Paper]][PaperLink] | [[Video]](https://youtu.be/kjjzzBdmm9E) ###

[PaperLink]: https://arxiv.org/pdf/2309.11839.pdf

## :scroll: About MoPA (ICRA'24)

MoPA is a MM-UDA method that aims to alleviate the imbalanced class-wise performance on Rare Objects (ROs) and the lack of 2D dense supervision signals through Valid Ground-based Insertion (VGI) and Segment Anything Mask consistency (SAM consistency). An overall structure is as follows.

<p align="middle">
  <img src="figs/Main_Method.jpg" width="600" />
</p>

Specifically, VGI insert more ROs from the wild with ground truth to guide the recognition of ROs during UDA process without introducing artificial artifacts, while SAM consistency leverage image masks from [Segment Anything Model](https://github.com/facebookresearch/segment-anything) to encourage mask-wise prediction consistency.

<p align="middle">
  <img src="figs/Full_VGI.gif" width="600" />
</p>

<p align="middle">
  <img src="figs/SAM_consistency.jpg" width="600" />
</p>

## :eyes: Updates

* [2024.03] We are now refactoring our code and evaluating its feasibility. Code will be available shortly. 
* [2024.01] Our paper is accepted by ICRA 2024! Check our paper on arxiv [here][Paperlink].


## :writing_hand: TODO List

- [x] Initial release. :rocket:
- [ ] Add installation and prerequisite details.
- [ ] Add data preparation details.
- [ ] Add training details.
- [ ] Add evaluation details.


## :clap: Acknowledgement
We greatly appreciate the contributions of the following public repos:
- [torchsparse](https://github.com/mit-han-lab/torchsparse)
- [SPVNAS](https://github.com/mit-han-lab/spvnas)
- [SalsaNext](https://github.com/TiagoCortinhal/SalsaNext)
- [Patchwork++](https://github.com/url-kaist/patchwork-plusplus)
- [xMUDA](https://github.com/valeoai/xmuda)

## :pencil: Citation
```
@article{cao2023mopa,
  title={Mopa: Multi-modal prior aided domain adaptation for 3d semantic segmentation},
  author={Cao, Haozhi and Xu, Yuecong and Yang, Jianfei and Yin, Pengyu and Yuan, Shenghai and Xie, Lihua},
  journal={arXiv preprint arXiv:2309.11839},
  year={2023}
}
```
