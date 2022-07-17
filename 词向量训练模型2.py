import numpy as np
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset, DataLoader
import pickle,os,random
import torch
import torch.nn as nn
def train_vec(file="split.txt"):
    all_data = open(file, "r", encoding="utf-8").read().split("\n")
    model = Word2Vec(all_data, vector_size=107, min_count=1, workers=5, window=4, hs=0, sg=0)
    print()
if __name__=="__main__":
    train_vec()
    pass