import torch
import torch.nn as nn
from MHSA import SA
from torchsummary import summary
from torch.nn.functional import relu
from ViT import VisionTransformer
from MHSA_2 import SA

class U_Net(nn.Module):
    def __init__(self, n_class):
        super(U_Net, self).__init__()

        # Encoder
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.e22 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Connection from Bottom
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        ## Self-Attention
        self.vit = VisionTransformer([1, 16, 16], patch_size=[4,4], hidden_dim=20, num_heads=7, out_dim=2, num_encoder_blocks=3)
        
        ## Self-Attention
        mhsa = SA(5, 8)

        # Decoder
        # Deconvolution
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride = 2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # output is same as the number of classes
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)
        # self.self_attention = SA()

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        print("xe52.shape", xe52.shape)

        # N, C, H, W = X.shape
        temp = self.vit(xe52)
        print("temp.shape", temp.shape)


        # Deocder
        xu1 = self.upconv1(xe52)
        # Skip Connection

        skip_connection_1 = torch.cat([xu1, xe42], dim=1)
        upsampled1_11 = relu(self.d11(skip_connection_1))
        upsampled1_12 = relu(self.d12(upsampled1_11))

        upsampled2 = self.upconv2(upsampled1_12)
        skip_connection_2 = torch.cat([xe32, upsampled2], dim=1)
        upsampled2_21 = relu(self.d21(skip_connection_2))
        upsampled2_22 = relu(self.d22(upsampled2_21))
        
        upsampled3 = self.upconv3(upsampled2_22)
        skip_connection_3 = torch.cat([xe22, upsampled3], dim=1)
        upsampled3_31 = relu(self.d31(skip_connection_3))
        upsampled3_32 = relu(self.d32(upsampled3_31))

        upsampled4 = self.upconv4(upsampled3_32)
        skip_connection_4 = torch.cat([xe12, upsampled4], dim=1)
        upsampled4_41 = relu(self.d41(skip_connection_4))
        upsampled4_42 = relu(self.d42(upsampled4_41))
        out = self.outconv(upsampled4_42)
        return out


# Define a hook function to print the output size
def hook_fn(module, input, output):
    print(f"{module.__class__.__name__} output size: {output.size()}")

if __name__ == "__main__":
    unet = U_Net(n_class = 2)
    summary(unet)

    # Register the hook for all Conv2d layers in the model
    for layer in unet.children():
        if isinstance(layer, nn.Conv2d):
            layer.register_forward_hook(hook_fn)

    # Create a random input tensor with the desired input size
    # input_size = (batch_size, 1, height, width)
    # input_tensor = torch.randn(input_size)

    # Forward pass to trigger the hooks
    # output_tensor = model(input_tensor)
    input = torch.rand([1, 3, 256, 256])
    output_tensor = unet(input)