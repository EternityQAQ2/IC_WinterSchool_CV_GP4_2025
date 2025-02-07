#### Hlworld

#### 接下来介绍下Unet模型
```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # 流水线处理
        self.conv = nn.Sequential(
            # 卷积核使用3*3
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 编码器
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # 解码器
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码
        d1 = self.down1(x)  # [B,64,H,W]
        p1 = self.maxpool(d1)

        d2 = self.down2(p1) # [B,128,H/2,W/2]
        p2 = self.maxpool(d2)

        d3 = self.down3(p2) # [B,256,H/4,W/4]
        p3 = self.maxpool(d3)

        d4 = self.down4(p3) # [B,512,H/8,W/8]
        p4 = self.maxpool(d4)

        # 中间层
        bn = self.bottleneck(p4) # [B,1024,H/16,W/16]

        # 解码
        up_4 = self.up4(bn)                   # [B,512,H/8,W/8]
        cat_4 = torch.cat([up_4, d4], dim=1)  # skip connection
        c4 = self.conv4(cat_4)                # [B,512,H/8,W/8]

        up_3 = self.up3(c4)                   # [B,256,H/4,W/4]
        cat_3 = torch.cat([up_3, d3], dim=1)
        c3 = self.conv3(cat_3)                # [B,256,H/4,W/4]

        up_2 = self.up2(c3)                   # [B,128,H/2,W/2]
        cat_2 = torch.cat([up_2, d2], dim=1)
        c2 = self.conv2(cat_2)                # [B,128,H/2,W/2]

        up_1 = self.up1(c2)                   # [B,64,H,W]
        cat_1 = torch.cat([up_1, d1], dim=1)
        c1 = self.conv1(cat_1)                # [B,64,H,W]

        out = self.out_conv(c1)               # [B,1,H,W]
        out = torch.sigmoid(out)              # 概率输出 [0,1]
        return out

model = UNet(in_channels=3, out_channels=1).to(device)
```
输入参数是[3,1,240,240],即FLA图像的数据信息：RGB三通道，一个输出通道(灰度图像),240*240图像的大小。
按照Unet定义，先把输入通道从3变成512。
bottleneck（瓶颈层）位于 编码器（下采样）和解码器（上采样）之间，其作用是 进一步提取全局特征，增强网络的感受野，为后续的解码阶段提供更强的语义信息。
在这里使用bottleneck提取到1024条通道的信息。


