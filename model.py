
import torch
import torch.nn as nn
from resnet import resnet18, resnet34,resnet50
import numpy as np
import cv2

def save_feats_mean(x):
    b, c, h, w = x.shape
    if h == 256:
        with torch.no_grad():
            x = x.detach().cpu().numpy()
            x = np.transpose(x[0], (1, 2, 0))
            x = np.mean(x, axis=-1)
            x = x/np.max(x)
            x = x * 255.0
            x = x.astype(np.uint8)
            x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
            x = np.array(x, dtype=np.uint8)
            return x

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.r1 = ResidualBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.r1(inputs)
        p = self.pool(x)
        return x, p

class Transformer(nn.Module):
    def __init__(self, in_c, out_c, dim, num_layers=1, patch_size=1):
        super().__init__()
        
        self.patch_size = patch_size
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.tblock = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)  # [-1, 2048, 8, 8] -> [-1, 256, 8, 8]
        b, c, h, w = x.shape
        
        # Handle patch size > 1
        if self.patch_size > 1:
            # Ensure dimensions are divisible by patch_size
            pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
            
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))
                b, c, h, w = x.shape
            
            # Calculate new dimensions
            h_patches = h // self.patch_size
            w_patches = w // self.patch_size
            patch_dim = c * self.patch_size * self.patch_size
            
            # Reshape to patches
            x = x.reshape(b, c, h_patches, self.patch_size, w_patches, self.patch_size)
            x = x.permute(0, 2, 4, 1, 3, 5)  # [b, h_patches, w_patches, c, patch_size, patch_size]
            x = x.reshape(b, h_patches * w_patches, patch_dim)  # [b, num_patches, patch_features]
            
            # Permute for transformer input (seq_len, batch, features)
            x = x.permute(1, 0, 2)  # [num_patches, b, patch_features]
            
            # Process with transformer
            x = self.tblock(x)  # [num_patches, b, patch_features]
            
            # Reshape back
            x = x.permute(1, 0, 2)  # [b, num_patches, patch_features]
            x = x.reshape(b, h_patches, w_patches, c, self.patch_size, self.patch_size)
            x = x.permute(0, 3, 1, 4, 2, 5)  # [b, c, h_patches, patch_size, w_patches, patch_size]
            x = x.reshape(b, c, h, w)
            
            # Remove padding if added
            if pad_h > 0 or pad_w > 0:
                x = x[:, :, :h-pad_h, :w-pad_w]
        else:
            # Original case: patch_size = 1
            x = x.reshape(b, c, h*w)  # [batch, channels, h*w]
            x = x.permute(2, 0, 1)    # [h*w, batch, channels]
            x = self.tblock(x)        # [h*w, batch, channels]
            x = x.permute(1, 2, 0)    # [batch, channels, h*w]
            x = x.reshape(b, c, h, w) # [batch, channels, h, w]
        
        x = self.conv2(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c[0]+in_c[1], out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r1(x)
        x = self.r2(x)
        return x

class TResUnet(nn.Module):
    def __init__(self, patch_size=4):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.patch_size = patch_size

        """ Local Encoder """
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        """ Global Encoder """
        # self.b1 = Bottleneck(1024, 256, 256, num_layers=2) <- original
        # self.b1 = Bottleneck(256, 64, 256, num_layers=1) # transformer block 

        # """Resnet34 """
        # self.t1 = Transformer(512, 128, 128, num_layers=1, patch_size=1) #for segmentation (128, layers=2)
        # self.t2 = Transformer(256,128,128, num_layers=1, patch_size=1)

        """"Resnet50"""
        self.t1 = Transformer(2048,256,256 * patch_size * patch_size, num_layers=1, patch_size=patch_size)
        self.t2 = Transformer(1024, 256, 256 * patch_size * patch_size, num_layers=1, patch_size=patch_size)

        # """ Decoder ResNet18 """
        # self.d1 = DecoderBlock([256, 128], 128)  # original: 256
        # self.d2 = DecoderBlock([128, 64], 64)
        # self.d3 = DecoderBlock([64, 64], 32)
        # self.d4 = DecoderBlock([32, 3], 16)

        # """ Decoder Resnet34 """
        # self.d1 = DecoderBlock([256, 128], 128)  # s4 is 512, s2 is 128
        # self.d2 = DecoderBlock([128, 64], 128)  # d1 output is 256, s1 is 64
        # self.d3 = DecoderBlock([128, 64], 64)    # d2 output is 128, s0 is 64
        # self.d4 = DecoderBlock([64, 3], 32)      # d3 output is 64, s0 is 3

        """ Decoder Resnet50 """
        self.d1 = DecoderBlock([512, 512], 256)  # s4 is 512, s2 is 128
        self.d2 = DecoderBlock([256, 256], 128)  # d1 output is 256, s1 is 64
        self.d3 = DecoderBlock([128, 64], 64)    # d2 output is 128, s0 is 64
        self.d4 = DecoderBlock([64, 3], 32)      # d3 output is 64, s0 is 3

        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, heatmap=None):
                                    #Resnet34                   #Resnet50
        s0 = self.layer0(x)    ## [-1, 64, h/2, w/2]          [-1,64,...]
        s1 = self.layer1(s0)    ## [-1, 64, h/4, w/4]         [-1,256,...]
        s2 = self.layer2(s1)    ## [-1, 128, h/8, w/8]        [-1,512,...]
        s3 = self.layer3(s2)    ## [-1, 256, h/16, w/16]      [-1,1024,...]
        s4 = self.layer4(s3)    ## [-1, 512, h/32, w/32]      [-1,2048.,..]

        t1 = self.t1(s4)  # [-1,128,8,8]
        t1 = self.up(t1) # [-1,128,16,16]
        t2 = self.t2(s3) # [-1,128,16,16]
        b3 = torch.cat([t1,t2], axis = 1) ## [1,256,...,...]

        d1 = self.d1(b3, s2)
        d2 = self.d2(d1, s1) 
        d3 = self.d3(d2, s0)
        d4 = self.d4(d3, x)

        y = self.output(d4)

        if heatmap != None:
            hmap = save_feats_mean(d4)
            return hmap, y
        else:
            return y

if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    model = TResUnet()
    y = model(x)
    print(y.shape)
