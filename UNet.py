import torch
import torch.nn as nn

class U_Net(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        # Encoder
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.e21 = nn.Conv2d(64, 128, kernel = 3, padding = 1)
        self.e22 = nn.Conv2d(128, 128, kernel = 3, padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.e51 = nn.Conv2d(512,1024,  kernel_size=3, padding = 1)
        self.e51 = nn.Conv2d(1024,1024, kernel_size=3, padding = 1)
        
        
        # Decoder
        # Deconvolution
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d21 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=2, stride=2)
        self.d32 = nn.Conv2d(128, 128, kernel_size=2, stride=2)
        
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride = 2)
        self.d41 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.d41 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        
        # output is same as the number of classes
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)



def forward(self, x):

    # Encoder
    xe11 = nn.ReLU(self.e11(x))
    xe12 = nn.ReLU(self.e12(xe11))
    xp1 = self.pool1(xe12)

    xe21 = nn.ReLU(self.e21(xp1))
    xe22 = nn.ReLU(self.e22(xe21))
    xp2 = self.pool2(xe22)

    xe31 = nn.ReLU(self.e31(xp2))
    xe32 = nn.ReLU(self.e32(xe31))
    xp3 = self.pool3(xe32)

    xe41 = nn.ReLU(self.e41(xp3))
    xe42 = nn.ReLU(self.e41(xe41))
    xp4 = self.pool4(xe42)

    xe51 = nn.ReLU(self.e51(xp4))
    xe52 = nn.ReLU(self.e52(xe51))


    # Deocder
    upsampled1 = self.upconv1(xe52)
    # Skip Connection
    skip_connection_1 = torch.cat([xe42, upsampled1], dim =1)
    upsampled1_11 = nn.ReLU(self.d11(skip_connection_1))
    upsampled1_12 = nn.ReLU(self.d12(upsampled1_11))
    
    upsampled2 = self.upconv2(upsampled1_12)
    skip_connection_2 = torch.cat([xe32, upsampled2], dim=1)
    upsampled2_21 = nn.ReLU(self.d21(skip_connection_2))
    upsampled2_22 = nn.ReLU(self.d22(upsampled2_21))    
    
    upsampled3 = self.upconv3(upsampled2_22)
    skip_connection_3 = torch.cat([upsampled3, xe32], dim=1)
    upsampled3_31 = nn.ReLU(self.d31(skip_connection_3))
    upsampled3_32 = nn.ReLU(self.d32(upsampled3_31))
    
    upsampled4 = self.upconv4(upsampled3_32)
    skip_connection_4 = torch.cat([upsampled4, xe32], dim=1)
    upsampled4_41 = nn.ReLU(self.d41(skip_connection_4))
    upsampled4_42 = nn.ReLU(self.d42(upsampled4_41))
    
    out = self.outconv(upsampled4_42)    
    
    return out


if __name__ == "__main__":
    unet = U_Net
    print(unet)