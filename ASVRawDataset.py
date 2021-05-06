import numpy as np
import torch.utils.data as data
import librosa


class ASVRawDataset(data.Dataset):
    def __init__(self, root, partition, protocol_name):
        super(ASVRawDataset, self).__init__()
        self.root = root
        self.partition = partition

        self.sysid_dict = {
            'bonafide': 1,  
            'spoof': 0, 
        }
        
        protocol_dir = root.joinpath(protocol_name)
        protocol_lines = open(protocol_dir).readlines()

        self.features = []
        if self.partition == 'train':
            feature_address = 'ASVspoof2019_LA_train'
        elif self.partition == 'dev':
            feature_address = 'ASVspoof2019_LA_dev'  

        for protocol_line in protocol_lines:
            tokens = protocol_line.strip().split(' ')
            # The protocols look like this: 
            #  [0]      [1]       [2][3]  [4]
            # LA_0070 LA_D_7622198 -  -  bonafide 

            feature_path = self.root.joinpath(feature_address, 'flac', tokens[1] + '.flac')
            sys_id = self.sysid_dict[tokens[4]]
            self.features.append((feature_path, sys_id))

    def load_feature(self, feature_path):
        feature, sr = librosa.load(feature_path, sr=16000)
        fix_len = sr*4

        while feature.shape[0] < fix_len:
            feature = np.concatenate((feature, feature))
        feature = feature[:fix_len]

        return feature

    def __getitem__(self, index):
        feature_path, sys_id = self.features[index]
        feature = self.load_feature(feature_path)
        return feature, sys_id

    def __len__(self):
        return len(self.features)
    