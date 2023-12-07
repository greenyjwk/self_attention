import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class SA(nn.Module):
    def __init__(self, hidden_dim, numOfHead):
        super().__init__()
        # Patch, Embedding = X.shape
        self.hidden_dim = hidden_dim
        self.num_head = numOfHead
        self.query_weight = []
        self.key_weight = []
        self.value_weight = []
        self.softmax = nn.Softmax(dim=-1)
        for _ in range(self.num_head):
            (self.query_weight).append(nn.Linear(hidden_dim, hidden_dim))
            (self.key_weight).append(nn.Linear(hidden_dim, hidden_dim))
            (self.value_weight).append(nn.Linear(hidden_dim, hidden_dim))
        self.linear = nn.Linear(numOfHead * hidden_dim, self.hidden_dim )

    def forward(self, X):
        multiple_img = []
        
        # # of Batch, # of Patch, embedding size  = X.shape 

        for img in X:
            # of patch_size, embedding_size
            patch_size, embedding = img.shape
            multi_head = []
            for idx in range(self.num_head):
                ##### Query, Key and Value weights are trainable_ does it have to be linear MLP or CNN?
                q = self.query_weight[idx](img)
                k = self.key_weight[idx](img)
                v = self.value_weight[idx](img)

                qk = q @ k.T
                qk_normalized = qk / (embedding**2)
                qk_normalized_softmax = self.softmax(qk_normalized)
                qkv = qk_normalized_softmax @ v
                multi_head.append(qkv)

            h_stacked_multihead = torch.hstack(multi_head)
            multiple_img.append(h_stacked_multihead)

            for_linear_layer_list = []
            for img in multiple_img:
                img_after_unsqueezed = torch.unsqueeze(img, dim=0)
                for_linear_layer_list.append(img_after_unsqueezed)

            H = torch.cat(for_linear_layer_list)
            out = self.linear(H)
            return out

if __name__ =="__main__":
    random_tensor = torch.rand([196, 1000])

    '''
    input -> (input embedding dimension, hidden_dimension, The Number of Heads)
    '''

    mhsa = SA(random_tensor, 1000, 8)
    
    '''
    input -> (Batch Size, Number of Patch, Dimension)
    '''
    input_random_tensor = torch.rand(32, 196, 1000)
    output = mhsa(input_random_tensor)
    print(output.shape)

    # summary(self_attention, (1000, 8000))