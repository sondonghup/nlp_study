import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import data
import model_seq2seq
import argparse
import os

train = "C:/Users/sondo/study/data/train/train.txt"
valid = "C:/Users/sondo/study/data/dev/dev.txt"
test = "C:/Users/sondo/study/data/test/test.txt"

def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="C:/Users/sondo/study/attention_test_4/save", type=str)
    parser.add_argument("--train_data_path", default='C:/Users/sondo/study/data/train/train.txt', type=str)
    parser.add_argument("--valid_data_path", default='C:/Users/sondo/study/data/dev/dev.txt', type=str)
    parser.add_argument("--batch_size", default=50, type=int, help="Total batch size for training.")
    parser.add_argument("--lr", default=5e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_epochs", default=60, type=int, help="Total number of training epochs to perform.")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = _get_args()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    vocab2index = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "<SOS>": 10, "<EOS>": 11, "<PAD>": 12}
    index2vocab = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10:"<SOS>", 11:"<EOS>", 12:"<PAD>" }
    corpus_len = len(vocab2index)

    train_origin,train_reverse = data.get_datas(args.train_data_path)
    valid_origin,valid_reverse = data.get_datas(args.valid_data_path)
    
    
    enc_embed_size = 100
    dec_embed_size = 100
    enc_hidden_size = 30
    dec_hidden_size = 30
    

    batch_size = args.batch_size
    epoch = args.num_epochs
    lr = args.lr
    pre_valid_loss = 100
    best_valid_loss = 10000
    teacher_force_ratio = 0.7

    train_dataset = data.MyDataset(train_origin,train_reverse)
    valid_dataset = data.MyDataset(valid_origin,valid_reverse)
    train_dataloader = DataLoader(train_dataset,batch_size = batch_size,shuffle=False,collate_fn = train_dataset.batch_data_process)
    valid_dataloader = DataLoader(valid_dataset,batch_size = batch_size,shuffle=False,collate_fn = valid_dataset.batch_data_process)

    enc = model_seq2seq.Encoder(corpus_len, enc_embed_size, enc_hidden_size, dec_hidden_size)
    attn = model_seq2seq.Attention(enc_hidden_size, dec_hidden_size)
    dec = model_seq2seq.Decoder(corpus_len, dec_embed_size, enc_hidden_size, dec_hidden_size, attn)

    
    model = model_seq2seq.Seq2Seq(enc , dec)

    opt = torch.optim.Adam(model.parameters(),lr = lr)
    
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
            
    model.apply(init_weights)
    
    

    def train(model, train_dataloader):
        model.train()
        epoch_loss = 0
        for i , batch in enumerate(tqdm(train_dataloader)):
            src = batch[0]
            trg = batch[1]
            src = src.permute(1,0)
            trg = trg.permute(1,0)
            
            opt.zero_grad()
            output = model(src,trg, teacher_force_ratio)
            # print("output",output)
            # print("output.shape",output.shape)
            # input()
            output_dim = output.shape[-1] # 출력 크기
            # print("output",output[1:])
            # print("output.shape",output[1:].shape)
            # input()
            output = output[1:].view(-1, output_dim)

            trg = trg[1:].reshape(-1)
            loss = nn.CrossEntropyLoss(ignore_index = 12)(output, trg)
            loss.backward() # 기울기(gradient) 계산
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            epoch_loss += loss.item()
        # print("------------------",epoch_loss)
        return epoch_loss / len(train_dataloader)
    
    def evaluate(model, valid_dataloader):
        model.eval() # 평가 모드
        epoch_loss = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(valid_dataloader)):
                src = batch[0]
                trg = batch[1]
                src = src.permute(1,0)
                trg = trg.permute(1,0)
                output = model(src, trg, teacher_force_ratio)
                output_dim = output.shape[-1]
                # print("trg",trg)
                # print("trg.shape",trg.shape)
                # input()
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].reshape(-1)
                loss = nn.CrossEntropyLoss(ignore_index = 12)(output, trg)
                epoch_loss += loss.item()
            
        return epoch_loss / len(valid_dataloader)
    
    for e in range(epoch):
        train_loss = train(model, train_dataloader)
        valid_loss = evaluate(model, valid_dataloader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_dir = os.path.join(args.output_dir, f"epoch_{e}_step")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        print("EPOCH : ", e)
        print("TRAIN LOSS", train_loss)
        print("VALID LOSS", valid_loss)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def train(model, train_dataloader, optimizer, criterion, clip):
        model.train()
        epoch_loss = 0
        
        for i , batch in enumerate(train_dataloader):
            src = batch[0]
            trg = batch[1]
            
            optimizer.zero_grad()
            
            output = model(src,trg)
            # print("output",output)
            # print("output.shape",output.shape)

    # for e in range(epoch):
    #     for origin_index, reverse_index, label_index in tqdm(train_dataloader):
    #         opt.zero_grad()
    #         loss = model(origin_index,reverse_index, label_index)
    #         loss.backward()
    #         opt.step()
            
        
    #     print(f"loss:{loss:.3f}")
        
            
    #     with torch.no_grad():
    #         for origin_index,reverse_index,label_index in tqdm(valid_dataloader):
    #             valid_loss = model(origin_index,reverse_index,label_index)

    #     if valid_loss < pre_valid_loss:
    #         pre_valid_loss = valid_loss
    #         save_dir = os.path.join(args.output_dir, f"epoch_{e}_step")
    #         if not os.path.exists(save_dir):
    #             os.makedirs(save_dir)
    #         torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    #     print(f"epoch:{e}")
    #     print(f"valid_loss:{valid_loss:.3f}")
