import torch
import os
import torch.nn as nn
import torch.nn.functional as F


class SA(nn.Module):
    def __init__(self, X, numOfHead):        
        Batch, Patch, Embedding = X.shape()
        self.convolutionalEmbedding = nn.Sequential(
            nn.Conv3d
        )
        
        self.num_head = numOfHead
        
        
        self.query_weight = []
        self.key_weight = []
        self.value_weight = []
        for i in range(self.num_head):
            (self.query_weight).append(nn.Linear(Embedding, Embedding))
            (self.key_weight).append(nn.Linear(Embedding, Embedding))
            (self.value_weight).append(nn.Linear(Embedding, Embedding))
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, X):    
        Batch, Patch, embedding= X.shape()
    
        # making tokenized    
        multiple_img = []
        for img in X:
            
            Patch, embedding = img.shape()
            multi_head = []
            
            for i in range(self.num_head):
                ##### Query, Key and Value weights are trainable_ does it have to be linear MLP or CNN?
                Q = self.query_weight(img)
                K = self.key_wieght(img)
                V = self.value_weight(img)
                
                qk = Q @ K.T
                qk_normalized = qk / (embedding**2)
                qk_normalized_softmax = self.softmax(qk_normalized)
                qkv = qk_normalized_softmax @ V
                multi_head.appned(qkv)
                
            h_stacked_multihead = torch.hstack(multi_head)
            multiple_img.append(h_stacked_multihead)
            
            for_linear_layer_list = []
            for img in multiple_img:
                img_after_unsqueezed = torch.unsqueeze(img, dim = 0 )
                for_linear_layer_list.append(img_after_unsqueezed)
                
            
            

if __name__ =="__main__":
    print()