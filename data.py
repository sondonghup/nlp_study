import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import pickle

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_datas(file):
    train_data = []
    train_label = []
    with open(file, "r", encoding="utf-8") as f:
        for text in f:
            data = text.split('\t')[0]
            label = text.split('\t')[1].replace("\n","")
            train_data.append(data.replace(" ",""))
            train_label.append(label.replace(" ",""))
    return train_data, train_label

class MyDataset(Dataset):
    def __init__(self,origin_data,reverse_data):
        self.origin_data = origin_data
        self.reverse_data = reverse_data
        self.vocab2index = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "<SOS>": 10, "<EOS>": 11, "<PAD>": 12}

    def __getitem__(self,index):
        origin = self.origin_data[index]
        reverse = self.reverse_data[index]

        origin_index = [self.vocab2index[i] for i in origin]
        reverse_index = [self.vocab2index[i] for i in reverse]

        return origin_index,reverse_index


    def batch_data_process(self,batch_datas):
        global device
        origin_index , reverse_index, base_reverse_index = [],[],[]
        origin_len , reverse_len = [],[]

        for origin,reverse in batch_datas:
            origin_index.append(origin)
            base_reverse_index.append(reverse)
            origin_len.append(len(origin))
            reverse_len.append(len(reverse))

        max_origin_len = max(origin_len)
        max_reverse_len = max(reverse_len)


        origin_index = [ i + [self.vocab2index["<PAD>"]] * (max_origin_len - len(i))   for i in origin_index]
        reverse_index = [[self.vocab2index["<SOS>"]] + i + [self.vocab2index["<EOS>"]] + [self.vocab2index["<PAD>"]] * (max_reverse_len - len(i))   for i in base_reverse_index]
        # label_index = [ i + [self.vocab2index["<EOS>"]] + [self.vocab2index["<PAD>"]] * (max_reverse_len - len(i))   for i in base_reverse_index]
        
        origin_index = torch.tensor(origin_index,device = device)
        reverse_index = torch.tensor(reverse_index,device = device)
        # label_index = torch.tensor(label_index,device = device )

        return origin_index,reverse_index


    def __len__(self):
        assert len(self.origin_data) == len(self.reverse_data)
        return len(self.reverse_data)