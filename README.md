# pc-darts-anti-spoofing

This repository contains implementation of our paper [Partially-Connected Differentiable Architecture Search for Deepfake and Spoofing Detection](https://arxiv.org/abs/2104.03123) accepted to INTERSPEECH 2021.

### Dependencies
```
pip install -r requirements.txt
```

### Dataset
The ASVspoof2019 database can be downloaded from [here](https://datashare.ed.ac.uk/handle/10283/3336)

The extracted data should be orginased as:
* LA/
   * ASVspoof2019_LA_dev/flac/...
   * ASVspoof2019_LA_eval/flac/...
   * ASVspoof2019_LA_train/flac/...
   * ASVspoof2019.LA.cm.dev.trl.txt
   * ASVspoof2019.LA.cm.eval.trl.txt
   * ASVspoof2019.LA.cm.train.trn.txt
   * ASVspoof2019.LA.cm.train.trn_h.txt (uploaded in /split_protocols)
   * ASVspoof2019.LA.cm.train.trn_t.txt (uploaded in /split_protocols)
   * ASVspoof2019.LA.cm.eval.trl.txt (provided in the database)
   * ...


For convience, you can change the codes' default `--data` argument to `'/path/to/your/LA'`, instead of typing it for each run.

### Usage
#### Architecture Search
To search with 4 layers with 16 initial channels, and with masked LFCC feature:
```
python train_search.py --layers=4 --init_channels=16 --frontend=lfcc --mask
```
#### Train from Scratch
To train with the reported best architecture in the paper, using 4 layers, 16 initial channels and masked LFCC feature:
```
python train_model.py --arch=ARCH --layers=4 --init_channels=16 --frontend=lfcc --mask
```
replace `ARCH` with `"Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 3), ('avg_pool_3x3', 2), ('avg_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 3), ('dil_conv_3x3', 2), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))"`
#### Evaluate
To evaluate the saved model using the same architecture in train from scratch on LA Evaluation partition:
```
python evaluate.py --arch=ARCH --model=/path/to/your/saved/models/epoch_x.pth --layers=4 --init_channels=16 --frontend=lfcc
```
also replace `ARCH` with the corresponding architecture.
#### Citation
If you use this repository, please consider citing:

```
@inproceedings{ge21c_interspeech,
  author={Wanying Ge and Michele Panariello and Jose Patino and Massimiliano Todisco and Nicholas Evans},
  title={{Partially-Connected Differentiable Architecture Search for Deepfake and Spoofing Detection}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={4319--4323},
  doi={10.21437/Interspeech.2021-1187}
}
```
#### Acknowledgement
This work is supported by the ExTENSoR project funded by the French Agence Nationale de la Recherche (ANR).

Codes are based on the implementations of [AutoSpeech](https://github.com/VITA-Group/AutoSpeech), [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS) and [project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts).
