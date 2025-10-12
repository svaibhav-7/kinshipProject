# Citations and References

This document lists all the academic papers, datasets, and open-source projects that this kinship verification project builds upon.

## Primary Research Papers

### 3D Face Reconstruction
```bibtex
@inproceedings{deng2019accurate,
    title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
    author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
    booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
    year={2019}
}
```

### Differentiable Rendering
```bibtex
@article{Laine2020diffrast,
    title   = {Modular Primitives for High-Performance Differentiable Rendering},
    author  = {Samuli Laine and Janne Hellsten and Tero Karras and Yeongho Seol and Jaakko Lehtinen and Timo Aila},
    journal = {ACM Transactions on Graphics},
    year    = {2020},
    volume  = {39},
    number  = {6}
}
```

### Basel Face Model
```bibtex
@article{paysan20093d,
    title={A 3D face model for pose and illumination invariant face recognition},
    author={Paysan, Pascal and Knothe, Reinhard and Amberg, Brian and Romdhani, Sami and Vetter, Thomas},
    journal={Advanced video and signal based surveillance},
    year={2009}
}
```

## Datasets

### KinFaceW-II Dataset
```bibtex
@inproceedings{lu2014neighborhood,
    title={Neighborhood repulsed metric learning for kinship verification},
    author={Lu, Jiwen and Zhou, Xiuzhuang and Tan, Yap-Peng and Shang, Yuanyuan and Zhou, Jie},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    pages={2594--2601},
    year={2014}
}
```

### FFHQ Dataset
```bibtex
@inproceedings{karras2019style,
    title={A style-based generator architecture for generative adversarial networks},
    author={Karras, Tero and Laine, Samuli and Aila, Timo},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    pages={4401--4410},
    year={2019}
}
```

### CelebA Dataset
```bibtex
@inproceedings{liu2015faceattributes,
    title = {Deep Learning Face Attributes in the Wild},
    author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
    booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
    month = {December},
    year = {2015}
}
```

## Open Source Libraries

### PyTorch
```bibtex
@incollection{NEURIPS2019_9015,
    title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
    author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
    booktitle = {Advances in Neural Information Processing Systems 32},
    pages = {8024--8035},
    year = {2019},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf}
}
```

### ArcFace
```bibtex
@inproceedings{deng2019arcface,
    title={Arcface: Additive angular margin loss for deep face recognition},
    author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={4690--4699},
    year={2019}
}
```

## Methodological References

### Siamese Networks for Face Verification
```bibtex
@article{chopra2005learning,
    title={Learning a similarity metric discriminatively, with application to face verification},
    author={Chopra, Sumit and Hadsell, Raia and LeCun, Yann},
    journal={Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on},
    volume={1},
    pages={539--546},
    year={2005}
}
```

### Kinship Verification Survey
```bibtex
@article{wang2017kinship,
    title={Kinship verification: A comprehensive survey and benchmark},
    author={Wang, Shengyong and Robinson, Joseph P and Fu, Yun},
    journal={ACM Computing Surveys},
    volume={50},
    number={2},
    pages={1--32},
    year={2017}
}
```

## Additional Dependencies

See `requirements.txt` and individual component README files for complete dependency lists.

## Usage Note

When using this project or its components in your research, please cite the relevant papers listed above. The specific citations needed depend on which components you use:

- For 3D face reconstruction: Cite the Deep3DFaceRecon paper
- For differentiable rendering: Cite the Nvdiffrast paper  
- For datasets: Cite the respective dataset papers
- For face recognition: Cite ArcFace if used
- For kinship verification: Cite relevant kinship papers

## Acknowledgments

We thank all the researchers and developers who have made their code and datasets publicly available, enabling this research.