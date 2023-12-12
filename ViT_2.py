import torch.nn as nn
import torch
from MHSA_2 import SA

class VisionTransformer(nn.Module):
    def __init__(self, img_shape, patch_size, hidden_dim, num_heads, out_dim, num_encoder_blocks=6):
        super().__init__()

        self.img_shape = img_shape
        # patch size: total number of pixels for single patch.
        self.patch_size = img_shape[0] * patch_size[0] * patch_size[1]
        self.num_patches = int(img_shape[0]*img_shape[1]/patch_size[0])**2
        self.hidden_dim = hidden_dim
        self.num_head = num_heads
        self.out_dim = out_dim
        self.num_encoder_blocks = num_encoder_blocks

        # Linear patching
        # self.linear_patching = nn.Linear(self.patch_size, self.hidden_dim)
        self.linear_patching = nn.Linear(16384, self.hidden_dim)
        
        # CLS embedding
        self.cls_embeddings = nn.Parameter(torch.rand(1, self.hidden_dim))

        # Positional embedding
        self.pos_embeddings = nn.Parameter(torch.rand(1 + self.num_patches, self.hidden_dim))
        
        # Transformer
        self.transformer1 = nn.Sequential(
            nn.LayerNorm((1+self.num_patches, self.hidden_dim)),
            SA(self.hidden_dim, self.num_head)
        )

        self.transformer2 = nn.Sequential(
            nn.LayerNorm((1+self.num_patches, self.hidden_dim)),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.Tanh()
        )

    def forward(self, X):
        print("X.shape", X.shape)
        N, C, H, W = X.shape
        # patch_size is all pixesl from the single patch
        print("self.patch_size: " , self.patch_size)
        print("self.num_patches: ", self.num_patches)
        print("N: ", N)
        print("X", X.shape)
        # patches = X.reshape(N, self.num_patches, self.patch_size)
        patches = X.reshape(N, self.num_patches, -1)
        print("modified patches dimension: ", patches.shape)
        
        new_patch_embedding_dim = patches.shape[-1]
        print("new_patch_embedding_dim", new_patch_embedding_dim)
        print("new_patch_embedding_dim", patches.shape)
        E = self.linear_patching(patches)

        # Adding class token to the series of patches(Concatenating)
        cls_embedding = nn.Parameter(self.cls_embeddings.repeat(N, 1, 1))
        E = torch.cat([cls_embedding, E], dim=1)

        # Adding positional embedding to the patches(Addition)
        Epos = nn.Parameter(self.pos_embeddings.repeat(N,1,1))
        print("E", E.shape)
        print("Epos", Epos.shape)
        Z = E + Epos
        for _ in range(self.num_encoder_blocks):
            res1 = self.transformer1(Z)
            print("self.transformer1(Z)", res1.shape)
            Z = self.transformer2(res1 + Z)
        print("Z", Z.shape)
        C = self.mlp_head(Z[:, 0])
        return C

if __name__ == "__main__":
    print("Hello ViT")
    vit = VisionTransformer(img_shape=[2, 16, 16], patch_size=[4,4], hidden_dim=20, num_heads=7, out_dim=2,  num_encoder_blocks = 3)
    input_tensor = torch.rand(1, 2, 16, 16)
    outcome = vit(input_tensor)
    print("outcome")
    print(outcome)