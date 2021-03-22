import torch

class DoubleConv(torch.nn.Module):
  def __init__(self, input_channels, output_channels, middle_channels=None):
    super(DoubleConv, self).__init__()

    if middle_channels is None:
      middle_channels = output_channels

    self.double_conv = torch.nn.Sequential(
      torch.nn.Conv2d(input_channels, middle_channels, kernel_size=3, padding=1),
      torch.nn.BatchNorm2d(middle_channels),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(middle_channels, output_channels, kernel_size=3, padding=1),
      torch.nn.BatchNorm2d(output_channels),
      torch.nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)

class Down(torch.nn.Module):
  def __init__(self, input_channels, output_channels):
    super(Down, self).__init__()

    self.maxpool_conv = torch.nn.Sequential(
      torch.nn.MaxPool2d(2),
      DoubleConv(input_channels, output_channels)
    )

  def forward(self, x):
    return self.maxpool_conv(x)

class Up(torch.nn.Module):
  def __init__(self, input_channels, output_channels, bilinear=True):
    super(Up, self).__init__()

    if bilinear:
      self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
      self.conv = DoubleConv(input_channels, output_channels)
    else:
      self.up = torch.nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=2, stride=2)
      self.conv = DoubleConv(input_channels, output_channels)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    diff_y = x2.size(2) - x1.size(2)
    diff_x = x2.size(3) - x1.size(3)

    x1 = torch.nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)

class OutConv(torch.nn.Module):
  def __init__(self, input_channels, output_channels):
    super(OutConv, self).__init__()
    self.conv = torch.nn.Conv2d(input_channels, output_channels, kernel_size=1)

  def forward(self, x):
    return self.conv(x)
