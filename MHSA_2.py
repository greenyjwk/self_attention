import torch
import torch.nn as nn
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
        self.linear = nn.Linear(numOfHead * hidden_dim, self.hidden_dim)


    def forward(self, X):
        multiple_img = []
        # # of Batch, # of Patch, embedding size  = X.shape
        
        N, C, H, W = X.shape
        X = X.reshape(N, 1024, -1)
        
        for img in X:
            # patch_size, embedding_size
            print("img.shape: ", img.shape)
            patch_size, embedding = img.shape

            multi_head = []
            for idx in range(self.num_head):
                # Query, Key and Value weights are trainable_ does it have to be linear MLP or CNN?
                q = self.query_weight[idx](img)
                k = self.key_weight[idx](img)
                v = self.value_weight[idx](img)
                qk = q @ k.T
                qk_normalized = (qk / self.hidden_dim**2)
                # print("=======")
                # print("qk_normalized")
                # print(qk_normalized)
                # print()
                # print("qk_normalized_softmax")
                # print(self.softmax(qk_normalized))

                temp = self.softmax(qk_normalized)
                temp = torch.round(temp * 1000) / 1000
                qk_normalized_softmax = self.softmax(qk_normalized)
                # print("qk_normalized_softmax", qk_normalized_softmax)
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
    
    '''
    input -> (input embedding dimension, hidden_dimension, The Number of Heads)
    '''
    mhsa = SA(256, 8)

    # Specify the desired range
    lower_bound = -40.0
    upper_bound = 40.0

    '''
    input -> (Batch Size, Number of Patch, Dimension)
    '''
    # input_random_tensor = torch.rand(2, 196, 5)
    input_random_tensor = torch.rand(2, 1024, 16, 16)

    # Scale and shift the tensor to the desired range
    input_random_tensor = input_random_tensor * (upper_bound - lower_bound) + lower_bound

    input_random_tensor = input_random_tensor.abs()
    output = mhsa(input_random_tensor)

    # print(output)
    # print(output.shape)