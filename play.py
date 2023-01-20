import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import data
import model_seq2seq
import argparse
import os

vocab2index = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "<SOS>": 10, "<EOS>": 11, "<PAD>": 12}
index2vocab = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10:"<SOS>", 11:"<EOS>", 12:"<PAD>" }

corpus_len = len(vocab2index)

enc_embed_size = 100
dec_embed_size = 100
enc_hidden_size = 30
dec_hidden_size = 30

enc = model_seq2seq.Encoder(corpus_len, enc_embed_size, enc_hidden_size, dec_hidden_size)
attn = model_seq2seq.Attention(enc_hidden_size, dec_hidden_size)
dec = model_seq2seq.Decoder(corpus_len, dec_embed_size, enc_hidden_size, dec_hidden_size, attn)

model = model_seq2seq.Seq2Seq(enc , dec)
model.load_state_dict(torch.load("C:/Users/sondo/study/attention_test_4/save/epoch_30_step/pytorch_model.bin"))
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def translate(sentence, model):
    model.eval()
    
    tokens = [vocab2index[i] for i in sentence]
    tokens = tokens
    print(tokens)
    tokens = torch.LongTensor(tokens).unsqueeze(1)

    with torch.no_grad():
        enc_outputs, hidden = model.encoder(tokens)
    trg_index = [vocab2index["<SOS>"]]
    for i in range(50):
        trg_tensor = torch.LongTensor([trg_index[-1]])
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, enc_outputs)
        # print(output)
        # input()
        pred_token = output.argmax(1).item()
        trg_index.append(pred_token)
        if pred_token == 11:
            break
    # print(trg_index)
    trg_tokens = [index2vocab[i] for i in trg_index]
    print(trg_tokens)
    return trg_tokens

if __name__ == "__main__":

    while True:
        s = input("입력하세요: ")
        if len(s) < 10:
            translate(s, model)
        else :
            continue
    


