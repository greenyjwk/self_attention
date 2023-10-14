import nibabel as nib
import torch.nn as nn
import torch

def main(img):
    print("Hello ViT")

    #input patch
    embedder = simpleCNNEmbedder()
    embeddings = embedder(img)
    
    query_weight = nn.Linear(intput, output)
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
        
if __name__ == "__main__":
    sample_image_path = "/mnt/storage/kaggle_abdominal_trauma/converted_nifti/19_14374.nii.gz"  #abdominal ct image
    nii_img = nib.load(sample_image_path)   # (512,512,300)
    ct_slice = nii_img.get_fdata()
    ct_slice = ct_slice[:,:,300]
    print(ct_slice.shape)  # Print the shape of the ndarray
    print(ct_slice[:,:,300])  # Print the shape of the ndarray
    main()