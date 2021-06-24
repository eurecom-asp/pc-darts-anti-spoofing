# pc-darts-anti-spoofing

This repository contains implementation of our paper [Partially-Connected Differentiable Architecture Search for Deepfake and Spoofing Detection](https://arxiv.org/abs/2104.03123) submitted to INTERSPEECH 2021.

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
   * ASVspoof2019.LA.cm.train.trn_h.txt
   * ASVspoof2019.LA.cm.train.trn_t.txt
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
#### Acknowledgement
Codes are based on the implementations of [AutoSpeech](https://github.com/VITA-Group/AutoSpeech), [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS) and [project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts).
