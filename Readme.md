# VCformer (IJCAI 2024)

​	![Paper](https://img.shields.io/badge/Paper-IJCAI-blue)! [Language](https://img.shields.io/badge/Language-Python-green) ![red](https://img.shields.io/badge/Framework-Pytorch-yellow) ![red](https://img.shields.io/badge/Domain-Multivaraite_Time_Series_Forecasting-red)

#### The repo is the official implementation for the paper: [VCformer: Variable Correlation Transformer with Inherent Lagged Correlation for Multivariate Time Series Forecasting](https://arxiv.org/abs/2405.11470)  

## Overall Architecture

![Architecture](https://github.com/CSyyn/VCformer/blob/main/image/VCformer.png)

The pseudo-code of VCformer is as simple as the following:

![pseudo-code](https://github.com/CSyyn/VCformer/blob/main/image/pseudo-code.png)



## Usage

1. Install Python 3.8. For convenience, execute the following command.

   ```shell
   pip install -r requirements.txt 
   ```

   

2.  Prepare data. You can obtain the well pre-processed datasets from [[Google Drive\]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive\]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`. Here is a summary of used datasets.

![datasets](https://github.com/CSyyn/VCformer/blob/main/image/dataset_desc.png)

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

   ```bash
   sh ./scripts/Traffic/VCformer.sh
   ```



## Citation

If you want to cite our paper, use the citation below:

```latex
@misc{yang2024vcformer,
      title={VCformer: Variable Correlation Transformer with Inherent Lagged Correlation for Multivariate Time Series Forecasting}, 
      author={Yingnan Yang and Qingling Zhu and Jianyong Chen},
      year={2024},
      eprint={2405.11470},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```



## Acknowledgement

We appreciate the following Github repos a lot for their valuable code and efforts.

- Time-Series-Library ([https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library))
- PatchTST ([https://github.com/yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST))
- Crossformer ([https://github.com/Thinklab-SJTU/Crossformer](https://github.com/Thinklab-SJTU/Crossformer))

