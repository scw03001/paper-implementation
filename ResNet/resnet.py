import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # bias=False since we apply batch normalisation. When applying batch normalisation, bias is unnecessary since BN applies offset, which is similar to bias
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # skip connection
        self.downsampling = None
        # we need to perform downsampling if feature map size is different or in_channels and out_channels are different
        if stride != 1 or in_channels != out_channels:
            self.downsampling = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        if self.downsampling is not None:
            identity = self.downsampling(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out






class ResNet34(nn.Module):
    def __init__(self, num_classes = 1000):
        super(ResNet34, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(in_channels=3 , out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2)

        # conv2_x
        conv2_layers = []
        for _ in range(3):
            conv2_layers.append(ResidualBlock(in_channels=64, out_channels=64))    
        self.conv2 = nn.Sequential(*conv2_layers)

        # conv3_x
        conv3_layers = []
        for i in range(4):
            if i == 0:
                conv3_layers.append(ResidualBlock(in_channels=64, out_channels=128, stride=2))
            else:
                conv3_layers.append(ResidualBlock(in_channels=128, out_channels=128, stride=1))
        self.conv3 = nn.Sequential(*conv3_layers)

        # conv4_x
        conv4_layers = []
        for i in range(6):
            if i == 0:
                conv4_layers.append(ResidualBlock(in_channels=128, out_channels=256, stride=2))
            else:
                conv4_layers.append(ResidualBlock(in_channels=256, out_channels=256, stride=1))
        self.conv4 = nn.Sequential(*conv4_layers)
            
        # conv5_x
        conv5_layers = []
        for i in range(3):
            if i == 0:
                conv5_layers.append(ResidualBlock(in_channels=256, out_channels=512, stride=2))
            else:
                conv5_layers.append(ResidualBlock(in_channels=512, out_channels=512, stride=1))
        self.conv5 = nn.Sequential(*conv5_layers)


        # fc layer    
        # 512 x 7 x 7 -> 512 x 1 x 1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    
    def forward(self, x):
        # conv1 
        print(f"before conv1: {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        print(f"after conv1: {x.shape}")
        # conv2
        x = self.max_pooling(x)
        x = self.conv2(x)
        print(f"after conv2: {x.shape}")

        # conv3 
        x = self.conv3(x)
        print(f"after conv3: {x.shape}")

        # conv4
        x = self.conv4(x)
        print(f"after conv4: {x.shape}")

        # conv5
        x = self.conv5(x)
        print(f"after conv5: {x.shape}")

        x = self.avgpool(x)
        print(f"after avg pool: {x.shape}")
        # (batch size, 512, 1, 1) -> (batch_size, 512)
        x = torch.flatten(x, 1) 
        print(f"after flatten: {x.shape}")
        x = self.fc(x)
        print(f"after fc: {x.shape}")

        
        return x
    


if __name__ == "__main__":
    random_input = torch.randn(8, 3, 224, 224)
    model = ResNet34(num_classes=1000)

    output = model(random_input)
    print(f"output shape: {output.shape}")