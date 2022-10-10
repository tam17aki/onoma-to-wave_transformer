# Unofficial implementations of environmental sound synthesis system with Transformer
This repository provides unofficial implementations of environmental sound synthesis system with Transformer [1][2].

## Licence
MIT licence.

Copyright (C) 2022 Akira Tamamori

## Dependencies
We tested the implemention on Ubuntu 22.04. The verion of Python was `3.10.6`. The following modules are required:

- torch
- hydra
- progressbar2
- pandas
- soundfile
- librosa
- joblib
- numpy
- sklearn

## Datasets
You need to prepare the following two datasets.

   - [Real World Computing Partnership-Sound Scene Database (RWCP-SSD)](http://research.nii.ac.jp/src/en/RWCP-SSD.html)

   - [RWCP-SSD-Onomatopoeia](https://github.com/KeisukeImoto/RWCPSSD_Onomatopoeia)

## Configurations
- `unconditional/`: The models are NOT conditioned on sound event labels.
- `conditional/`:  The models are conditioned on sound event labels.


## Recipes
1. Modify `config.yaml` according to your environment. It contains settings for experimental conditions. For immediate use, you can edit mainly the directory paths according to your environment.

2. Run `preprocess.py`. It performs preprocessing steps.

3. Run `training.py`. It performs model training.

4. Run `inference.py`. It performs inference using trained model (i.e., generate audios from onomatopoeia).

After training, you can also use `synthesis.py`. This is a script for environmental sound synthesis using pretrained models. Unlike `inference.py`, it can easily synthesis audios using onomatopoeia and acoustic events specified in the yaml file. It is somewhat simply implemented since it does not use DataSet and DataLoader.
  
## References

[1] 岡本 悠希，井本 桂右，高道 慎之介，福森 隆寛，山下 洋一，"Transformerを用いたオノマトペからの環境音合成，" 日本音響学会2021年秋季研究発表会，pp. 943-946.

[2] Yuki Okamoto, Keisuke Imoto, Shinnosuke Takamichi, Takahiro Fukumori, and Yoichi Yamashita, "How Should We Evaluate Synthesized Environmental Sounds," arXiv:2208.07679 [Sound (cs.SD)].

```
@misc{https://doi.org/10.48550/arxiv.2208.07679,
  doi = {10.48550/ARXIV.2208.07679},
  
  url = {https://arxiv.org/abs/2208.07679},
  
  author = {Okamoto, Yuki and Imoto, Keisuke and Takamichi, Shinnosuke and Fukumori, Takahiro and Yamashita, Yoichi},
  
  title = {How Should We Evaluate Synthesized Environmental Sounds},
  
  publisher = {arXiv},
  
  year = {2022},
}
```
