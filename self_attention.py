import nibabel as nib
import torch.nn as nn
import torch
from einops import rearrange, reduce, repeat

def main(img):
    print("Hello ViT")

    #input patch
    embedder = simpleCNNEmbedder()
    embeddings = embedder(img)
    
    query_weight = nn.Linear(input, output)
    key_weight = nn.Linear(input, output)
    value_weight = nn.Linear(input, output)
    
    query = query_weight(x)
    key = key_weight(x)
    value = value_weight(x)
    
    q_k = torch.matmul(query, key.T)
    q_k_after_softmax = torch.softmax(q_k / hidden_dim ** 2)
    q_k_v = torch.matmul(q_k_after_softmax, value)
    nn.conv(c, c, kernel_size = (3,3), stride = 1)
    # This is for one image, but it still needs to work with multiple images
    
    
class simpleCNNEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Con2d(1, 32,  kernel_size = (3,3), stride = 1, padding = 1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Con2d(32, 32, kernel_size = (3,3), stride = 1, padding = 1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.maxPool2d(kernel_size=(2,2))
        
    def forward(self, x):
        # input 32x512x512, output 32x512x512
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop1(x)

        # input 32x512x512, output 32x256x256
        x = self.pool2(x)


'''
    input : n patches, and each patch (1, d) vector -> (n x d)
          : B batch Size(number of images), 
          -> thus, the input size is (B, n, d) 
    intermediate steps: 
        Q x K       : (n x n)
        V           : (n x d)
        (Q x K) x V : (n x d)
    output : (n+1, d) vectors, which represents the 
'''    
class MultiheadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.numOfHeads = 8
        self.embedded_dim = 100
        
        # They don't change the dimensions
        self.queryWeight = [nn.Linear(self.embedded_dim, self.embedded_dim) for _ in range(self.numOfHeads)]
        self.keyWeight = [nn.Linear(self.embedded_dim, self.embedded_dim) for _ in range(self.numOfHeads)]
        self.valueWeight = [nn.Linear(self.embedded_dim, self.embedded_dim) for _ in range(self.numOfHeads)]
        self.get_back_to_singlehead = nn.Linear(self.numOfHeads * self.embedded_dim , self.embedded_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, imgs):
        batchSize, num_patches, embedded_dim = imgs.shape
        
        # patches duplicate three to generate Q, K, V
        # singleImage = imgs[0]
        
        img_list = []
        concatenated_head = []
        #iterate the images
        for img in imgs:
            #iterate the number of heads
            multi_head = []
            for h in range(self.numOfHeads):
                # Generate Q, K, V
                query = self.queryWeight[h](img)
                key = self.keyWeight[h](img)
                value =  self.valueWeight[h](img)  # (n+1, embedded_dim)
                
                # query: nxd, key: dxn, queryxkey: nxn
                # q_k = torch.dot(query, key.T)
                q_k = query @ key.T
                
                '''
                    (n+1, n+1) It contans all the relations for between all patches.
                    each row represents the patch, and the all the columns from single row mean the 
                    relations from the selected row patch
                '''
                q_k_normalized = q_k / (embedded_dim ** 2)
                q_k_softmax = self.softmax(q_k_normalized)
                v_kq = q_k_softmax @ value  # (n+1, embedded_dim), and it contains attentions for all patches  -> n x d
                multi_head.append(v_kq)    # (num of head, n+1, embedded_dim)

            # This is for single image with concatednated multi heads
            single_img = torch.hstack(multi_head)
            img_list.append(single_img)
            
            '''
                img_list is array that contains the multiple images
                in order to make the list of images to the one single tensor that is compatible with list of images
                the following code is executed
            '''
            unsqueezed_img_list = [torch.unsqueeze(img, dim = 0) for img in img_list]
            H = torch.cat(unsqueezed_img_list)
            '''
                since the multiple heads are invovled, there are multiple outputs of the self-attention.
                Thus, it should be divided by the number of heads again.                
                And it contains multiple images attention information.
            '''
            output = self.get_back_to_singlehead(H)
            return output

if __name__ == "__main__":
    sample_image_path = "/mnt/storage/kaggle_abdominal_trauma/converted_nifti/19_14374.nii.gz"  #abdominal ct image
    nii_img = nib.load(sample_image_path)   # (512,512,300)
    ct_slice = nii_img.get_fdata()
    ct_slice = ct_slice[:,:,300]
    print(ct_slice.shape)  # Print the shape of the ndarray
        
    MHSA = MultiheadSelfAttention()
    
    imgs = torch.randn(10, 196, 100)
    MHSA(imgs)