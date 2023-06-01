import torch
import torch.nn as nn

from collections import OrderedDict

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        
        features = init_features
        
        # Encoder
        self.encoder1 = UNet._block(in_channels, features, name='enc1')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name='enc2')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name='enc3')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name='enc4')
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = UNet._block(features * 8, features * 16, name='bottleneck')
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name='dec4')
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name='dec3')
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name='dec2')
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2) 
        self.decoder1 = UNet._block(features * 2, features, name='dec1')
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder
        upconv4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat((upconv4, enc4), dim=1))
        upconv3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((upconv3, enc3), dim=1))
        upconv2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((upconv2, enc2), dim=1))
        upconv1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((upconv1, enc1), dim=1))
        out = self.conv(dec1)
        return out
    
    @staticmethod
    def _block(in_channels, out_channels, name):
        return nn.Sequential(
            OrderedDict([
                (name + '_conv1', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                (name + '_norm1', nn.BatchNorm2d(num_features=out_channels)),
                (name + '_relu1', nn.ReLU(inplace=True)),
                (name + '_conv2', nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                (name + '_norm2', nn.BatchNorm2d(num_features=out_channels)),
                (name + '_relu2', nn.ReLU(inplace=True)),
            ])
        )
        
def get_model(model_name):
    if model_name == 'unet':
        return UNet()
    else:
        raise NotImplementedError