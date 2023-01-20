from json import encoder
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import random


device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
    def __init__(self,word_index_len, enc_embed_size,enc_hidden_size,dec_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(word_index_len,enc_embed_size)
        self.gru = nn.GRU(enc_embed_size,enc_hidden_size)
        self.fc = nn.Linear(enc_hidden_size, dec_hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self,origin_index): # [ word x batch ]

        origin_embedding = self.embedding(origin_index) # [ word x batch x enc_emb_size]

        origin_embedding = self.dropout(origin_embedding) # [ word x batch x enc_emb_size]

        enc_outputs, encoder_hidden = self.gru(origin_embedding) # enc_output = [ word x batch x enc_hidden_size ] , enc_hidden = [ word_one x batch x enc_hidden_size ] 
        encoder_hidden = torch.tanh(self.fc(encoder_hidden[-1,:,:]))
        return enc_outputs, encoder_hidden
    
class Attention(nn.Module):
    def __init__(self,  enc_hidden_size, dec_hidden_size, n_heads = 3, ):
        super().__init__()
        self.enc_hidden_size = enc_hidden_size
        self.n_heads = n_heads
        self.head_dim = enc_hidden_size // n_heads
        
        self.fc_q = nn.Linear(enc_hidden_size,enc_hidden_size)
        self.fc_k = nn.Linear(enc_hidden_size,enc_hidden_size)
        self.fc_v = nn.Linear(enc_hidden_size,enc_hidden_size)
        
        self.fc_o = nn.Linear(enc_hidden_size,enc_hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    
    def forward(self, query, key, value):

        query = query.permute(1,0,2) # b 1 h
        key = key.permute(1,0,2)
        value = value.permute(1,0,2)

        batch_size = query.shape[0]
        
        Q = self.fc_q(query)
        K = self.fc_q(key)
        V = self.fc_q(value)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # b head 1 hidden / head
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale # [batch_size, n_heads, query_len, key_len]
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V) # [batch_size, n_heads, query_len, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous() 
        x = x.view(batch_size, -1, self.enc_hidden_size)
        x = self.fc_o(x) # [batch_size, query_len, hidden_dim]
        print("x",x.shape) 
        
        print("query",Q.shape) 
        print("key",K.shape)
        print("value",V.shape)
        input()

        
        return x, attention
     
    
class Decoder(nn.Module):
    def __init__(self,word_index_len, dec_embed_size,enc_hidden_size, dec_hidden_size, attention):
        super().__init__()
        
        self.output_size = word_index_len
        self.attention = attention
        self.embedding = nn.Embedding(word_index_len,dec_embed_size)
        self.dropout = nn.Dropout(0.5)
        self.gru = nn.GRU(dec_embed_size ,dec_hidden_size)
        self.fc_out = nn.Linear(dec_embed_size + dec_hidden_size * 2, word_index_len)
        
    def forward(self,decoder_input,hidden, enc_outputs): # decoder_input = [ batch ] , hidden = [ batch , hidden ] , enc_outputs = [ word x batch x hidden ]
        # print("decoder_input",decoder_input.shape)
        decoder_input = decoder_input.unsqueeze(0) # [ word_one x batch ]
        de_embedding = self.embedding(decoder_input)
        de_embedding = self.dropout(de_embedding)
        print(de_embedding.shape)
        print(hidden.unsqueeze(0).shape)
        output, hidden = self.gru(de_embedding, hidden.unsqueeze(0))
        context, attention = self.attention(output, enc_outputs.unsqueeze(0), enc_outputs.unsqueeze(0)) # [ batch x word ]
        

        attention = attention.unsqueeze(1) # [ batch x 1 x word]

        
        enc_outputs = enc_outputs.permute(1,0,2) # [ batch x word x hidden]

        
        

        # if len(de_embedding.shape) == 2:
        #     # print("ok")
        #     de_embedding = de_embedding.unsqueeze(1)
        


        
        attention_weighted = torch.bmm(attention, enc_outputs) # [ batch x word_one x hidden ]

         
        attention_weighted = attention_weighted.permute(1, 0, 2) # [ word_one x batch x hidden ]
        # print("de_embedding",de_embedding.shape)
        # print("attention_weighted",attention_weighted.shape)

        rnn_input = torch.cat((de_embedding, attention_weighted), dim=2) # [ word_one x batch x ( dec_emb_size + hidden ) ]

        decoder_output,decoder_hidden = self.gru(rnn_input,hidden.unsqueeze(0)) # [ word_one x batch x ( dec_emb_size + hidden ) ] , [ 1 x batch x hidden ] / decoder_output = [ word_one x batch x hidden ] , decoder_hidden = [ 1(layer) x batch x hidden ] 

        assert (decoder_output == decoder_hidden).all()
        
        de_embedding = de_embedding.squeeze(0) # [ batch x dec_emb_size ]
        decoder_output = decoder_output.squeeze(0) # [ word_one x hidden ] 
        attention_weighted = attention_weighted.squeeze(0) # [ batch x hidden ]
    
        prediction = self.fc_out(torch.cat((decoder_output, attention_weighted, de_embedding), dim=1)) # [ batch x (dec_embsize + hidden * 2) ] / [ batch x output_len ]
        return prediction,decoder_hidden.squeeze(0)
        
        
         
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        vocab2index = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "<SOS>": 10, "<EOS>": 11, "<PAD>": 12}
        index2vocab = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10:"<SOS>", 11:"<EOS>", 12:"<PAD>" }
        self.vocab2index = vocab2index
        self.index2vocab = index2vocab
        self.encoder = encoder
        self.decoder = decoder
        

    def forward(self, origin_index,reverse_index, teacher_force_ratio):
        
        src_list = []
        pred_list = []
        
        enc_outputs, hidden = self.encoder(origin_index) # enc_outputs = [ word x batch x enc_emb_size ] , hidden = [ batch x enc_emb_size ]

        
        trg_len = reverse_index.shape[0]
        batch_size = reverse_index.shape[1]
        
        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        
        
        src_input = reverse_index[0,:]
        # print("???", src_input[2])
        # print("src_input.shape",src_input.shape)
        # print("src_input[0][0]",src_input[0])
        # print("src_input[0][1]",src_input[1])
        
        # print("reverse_index.shape",reverse_index.shape)
        # print("reverse_index[:-1]",reverse_index[:-1])
        for i in range(1, trg_len):
            # print("<input>", input)
            print("enc_outputs[i,:]",enc_outputs[i,:].shape)
            output, hidden = self.decoder(src_input, hidden, enc_outputs[i,:])
            
            outputs[i] = output
            top = output.argmax(1)
            # print("top", top)
            # print(count)
            # input()

            teacher_force = random.random() < teacher_force_ratio
            src_input = reverse_index[i] if teacher_force else top
            
            
        # print("pred------------------")
        # print(pred_list)
        # input()
            
        return outputs
        



    