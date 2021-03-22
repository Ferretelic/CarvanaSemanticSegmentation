import torch

from layers import DoubleConv, Up, Down, OutConv

class UNet(torch.nn.Module):
  def __init__(self, input_channels, num_classes, bilinear=True):
    super(UNet, self).__init__()

    self.input_channels = input_channels
    self.num_classses = num_classes
    self.bilinear = bilinear

    self.input_conv = DoubleConv(input_channels, 64)
    self.down_1 = Down(64, 128)
    self.down_2 = Down(128, 256)
    self.down_3 = Down(256, 512)
    factor = 2 if bilinear else 1
    self.down_4 = Down(512, 1024 // factor)

    self.up_1 = Up(1024, 512 // factor, bilinear)
    self.up_2 = Up(512, 256 // factor, bilinear)
    self.up_3 = Up(256, 128 // factor, bilinear)
    self.up_4 = Up(128, 64, bilinear)
    self.output_conv = OutConv(64, num_classes)

  def forward(self, x):
    x1 = self.input_conv(x)
    x2 = self.down_1(x1)
    x3 = self.down_2(x2)
    x4 = self.down_3(x3)
    x5 = self.down_4(x4)

    x = self.up_1(x5, x4)
    x = self.up_2(x, x3)
    x = self.up_3(x, x2)
    x = self.up_4(x, x1)
    logits = self.output_conv(x)
    return logits

model = UNet(3, 10, bilinear=True).cuda()
images = torch.randn(64, 3, 128, 128).cuda()
outputs = model(images)
print(outputs.size())