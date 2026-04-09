import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")


class CustomNet(nn.Module):

  def __init__(self):

    super().__init__()

    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels = 3,
                  out_channels = 32,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2)
    )

    self.conv_block_2 = nn.Sequential(
      nn.Conv2d(in_channels = 32,
        out_channels = 64,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        ),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size = 2)
    )

    self.conv_block_3 = nn.Sequential(
        nn.Conv2d(in_channels = 64,
                  out_channels = 128,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv_block_4 = nn.Sequential(
        nn.Conv2d(in_channels = 128,
                  out_channels = 256,
                  kernel_size = 3,
                  padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = 256*14*14,
                  out_features = 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512,200)
    )


  def forward(self, x):

    x = self.conv_block_1(x)
    # print(f"Shape after block 1: {x.shape}")
    x = self.conv_block_2(x)
    # print(f"Shape after block 2: {x.shape}")
    x = self.conv_block_3(x)
    # print(f"Shape after block 3: {x.shape}")
    x = self.conv_block_4(x)
    x = self.classifier(x)
    # print(f"Shape after classifier: {x.shape}")
    return x
