from unet.unet_parts import *
class UNet(nn.Module):
    def __init__(self,n_channels,n_classes,bilinear=False):
        super(UNet,self).__init__()
        self.n_channels=n_channels
        self.n_classes=n_classes
        self.bilinear=bilinear
        self.inc=depthwise_block(n_channels,64,1)
        self.down1=Down(64,128)
        self.down2=Down(128,256)
        self.down3=Down(256,512)
        self.down4=Down(512,1024)
        self.up1=up(1024,512,bilinear)
        self.up2=up(512,256,bilinear)
        self.up3=up(256,128,bilinear)
        self.up4=up(128,64,bilinear)
        self.outc=OutConv(64,n_classes)


    def forward(self,x):
        x1=self.inc(x)
        x2=self.down1(x1)
        # print("x2" + str(x2.size()))
        x3=self.down2(x2)
        # print("x3" + str(x3.size()))
        x4=self.down3(x3)
        # print("x4" + str(x4.size()))
        x5=self.down4(x4)
        # print("x5" + str(x5.size()))
        x=self.up1(x5,x4)
        # print("up1" + str(x.size()))
        x=self.up2(x4,x3)
        # print("up2" + str(x.size()))
        x=self.up3(x,x2)
        # print("up3"+str(x.size()))
        x=self.up4(x,x1)
        # print("up4"+str(x.size()))
        logits=self.outc(x)
        # print("logits" + str(logits.size()))
        return logits

