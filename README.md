# Kinship Verification Project

A comprehensive kinship verification system using 3D face reconstruction and Siamese neural networks.

## Overview

This project implements a kinship verification system that combines:
- 3D face reconstruction using Deep3DFaceRecon_pytorch
- Siamese neural networks for kinship relationship classification
- Feature extraction and analysis pipelines

## Dependencies and Attribution

This project builds upon several open-source libraries and datasets. We acknowledge and thank the original authors:

### Core Components

1. **Deep3DFaceRecon_pytorch**
   - Repository: https://github.com/sicxu/Deep3DFaceRecon_pytorch
   - License: MIT License
   - Citation: 
   ```
   @inproceedings{deng2019accurate,
       title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
       author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
       booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
       year={2019}
   }
   ```

2. **Nvdiffrast**
   - Repository: https://github.com/NVlabs/nvdiffrast
   - License: Nvidia Source Code License (1-Way Commercial)
   - Citation:
   ```
   @article{Laine2020diffrast,
     title   = {Modular Primitives for High-Performance Differentiable Rendering},
     author  = {Samuli Laine and Janne Hellsten and Tero Karras and Yeongho Seol and Jaakko Lehtinen and Timo Aila},
     journal = {ACM Transactions on Graphics},
     year    = {2020},
     volume  = {39},
     number  = {6}
   }
   ```

3. **KinFaceW-II Dataset**
   - Please ensure you have proper licensing for the KinFaceW-II dataset
   - Original paper: [Citation needed for KinFaceW-II dataset]

### Additional Dependencies

See `requirements.txt` for a complete list of Python dependencies.

## Installation

1. Clone this repository:
```bash
git clone <your-repository-url>
cd kinshipProject
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r pipelines/requirements.txt
```

4. Follow the setup instructions for Deep3DFaceRecon_pytorch in the respective directory.

## Usage

[Add usage instructions here]

## Structure

```
kinshipProject/
├── Deep3DFaceRecon_pytorch/    # 3D face reconstruction module
├── nvdiffrast/                 # Differentiable rendering library
├── pipelines/                  # Main processing pipelines
├── KinFaceW-II/               # Dataset (ensure proper licensing)
├── combined_results/          # Output results
├── decoder_model/            # Decoder model components
├── final_siamese_model/      # Siamese network implementation
└── model_visuals_ganmodel/   # Visualization tools
```

## License

This project combines multiple components with different licenses:
- Deep3DFaceRecon_pytorch: MIT License
- Nvdiffrast: Nvidia Source Code License (Non-commercial use)
- Custom code: [Specify your license]

**Important**: The nvdiffrast component has a non-commercial license restriction. Please review all license terms before using this project commercially.

## Acknowledgments

We thank the authors and contributors of all the open-source projects and datasets used in this work.

## Citation

If you use this project in your research, please cite the relevant papers listed above.

## Contributing

[Add contributing guidelines]

## Contact

[Add contact information]