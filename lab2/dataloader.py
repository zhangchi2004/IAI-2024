import gensim
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def get_word2vec(word2vec_path):
    return gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

def word2vec(data_path, word2vec_path, pad_len):
    word2vec = get_word2vec(word2vec_path)
    
    with open(data_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    vecss = []
    labels = []
    for line in lines:
        words = line.strip().split()
        label = int(words[0])
        words = words[1:]
        vecs = []
        for word in words:
            try: vec = np.array(word2vec[word])
            except: vec = np.zeros(50)
            vecs.append(vec)
        vecs = np.stack(vecs)
        
        if vecs.shape[0] > pad_len: vecs = vecs[:pad_len, :]
        vecs = np.pad(vecs, ((0,pad_len - vecs.shape[0]),(0,0)), constant_values = 0)
        vecss.append(vecs)
        labels.append(label)
    
    vecss = np.stack(vecss)
    vecss = torch.tensor(vecss)
    labels = np.stack(labels)
    labels = torch.tensor(labels)
    return vecss, labels

def get_dataloader(data_path, split, batch_size, pad_len):
    assert split in ["train","test","validation"]
    path = data_path+split+".txt"
    w2vpath = data_path + "wiki_word2vec_50.bin"
    vecss, labels = word2vec(path,w2vpath, pad_len)
    dataset = TensorDataset(vecss, labels)
    print(len(dataset))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=3)
