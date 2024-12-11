# DeFace

> Learning to Decouple the Lights for 3D Face Texture Modeling  
> Tianxin Huang, Zhenyu Zhang, Ying Tai, Gim Hee Lee  
> NeurIPS'24 

![intro](overall.png)

[My Personal Page :)](https://tianxinhuang.github.io/) | [Our Project Page :)](https://tianxinhuang.github.io/projects/Deface)

## TODO List and ETA
- [x] Code for optimization.

- [x] Upload datasets and other adopted data.


## Installation
We suggest to use Anaconda for configuration:

1. Clone/download this repositry, create a new conda env: 

```
conda create -n deface python=3.8.5
```

and then activate it by (`conda activate deface`)

2. Install Cuda 11.3 by: 

```
conda install -c "nvidia/label/cuda-11.3.1" cuda
```

3. Install Torch by running:

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

4. Install other packages by running: 

```
pip install -r requirements.txt
```

5. We introduce [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch) for the segmentation of face region. Please download its [checkpoint](https://drive.google.com/file/d/1vYrfG-pXzU4g_YGDHWcJDtwXaVykx4Qt/view?usp=drive_link) and put it under (`faceparsing/res/cp`)

6. Please download the [baselMorphableModel](https://drive.google.com/file/d/13hsGFaAVgEde60hD9OxV5X0wfZoC7zvh/view?usp=drive_link) and unzip it under the dir. 

The sub-folders under (`Deface`) should be organized as

```
Deface/
├── cfgs/
│   ├── vox2_img.ini
│   ├── por_img.ini
│   ├── celeba.ini
│   └── vox2_seq.ini
├── faceparsing/
│   ├── ...
│   └── res/
│       └──cp/79999.pth
├── baselMorphableModel
└── ...
```

7. Download our [data collection](https://drive.google.com/file/d/1EDxHPe35WLn15jprmWkSbs0FLolrXN0y/view?usp=sharing), and run the codes on them.

## How to Use

Just replace following dirs with your own ones, and run the optimization with:

```
python3 evaluate.py --configs your_config_dir --input_dir your_img_dir --output_dir your_output_dir --ckpt_dir your_ckpt_dir >log.txt
```

We have provided 4 different config files for our evaluation data in the (`cfgs`) dir. You can also adjust the hyper-parameters for your own data.

For example, if you want to conduct experiments on single images from Voxceleb2 downloaded in (`shadow_data/voxceleb_pics`), the command to optimize would be:


After running the optimization, you can evaluate the performances by:

```
python3 metrics.py --input_dir your_img_dir --output_dir your_output_dir --result_dir your_result_file.txt
```

Then, the quantitative results would be written into the (`--result_dir`).

For example, if you want to conduct experiments on single images from Voxceleb2 downloaded in (`shadow_data/voxceleb_pics`), the command to optimize would be:

```
python3 evaluate.py --configs cfgs/vox2_img.ini --input_dir shadow_data/voxceleb_pics --output_dir output_pics --ckpt_dir output_ckpts >log.txt
```

After optimization, the evaluation would be:

```
python3 metrics.py --input_dir shadow_data/voxceleb_pics --output_dir output_pics --result_dir result.txt
```

The results would be written into (`result.txt`).


## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{huanglearning,
  title={Learning to Decouple the Lights for 3D Face Texture Modeling},
  author={Huang, Tianxin and Zhang, Zhenyu and Tai, Ying and Lee, Gim Hee},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```

```bibtex
@inproceedings{dib2021practical,
  title={Practical face reconstruction via differentiable ray tracing},
  author={Dib, Abdallah and Bharaj, Gaurav and Ahn, Junghyun and Th{\'e}bault, C{\'e}dric and Gosselin, Philippe and Romeo, Marco and Chevallier, Louis},
  booktitle={Computer Graphics Forum},
  volume={40},
  number={2},
  pages={153--164},
  year={2021},
  organization={Wiley Online Library}
}
```

# Acknowledgements
We built our source codes based on [NextFace](https://github.com/abdallahdib/NextFace). Following NextFace, the uvmap is taken from [here](https://github.com/unibas-gravis/parametric-face-image-generator/blob/master/data/regions/face12.json), landmarks prediction from [here](https://github.com/kimoktm/Face2face/blob/master/data/custom_mapping.txt). [redner](https://github.com/BachiLi/redner/) is used for ray tracing, albedoMM model from [here](https://github.com/waps101/AlbedoMM/) is introduced for the texture modeling.


# contact 
mail: 21725129 @at zju.edu.cn

twitter: @huangtxx
