"""ResNet-50 written by hand (He et al. 2015), no torchvision.models.

Layer names (conv1, bn1, layer1..4, downsample, fc) follow the torchvision naming convention.
"""
import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """1x1 reduce -> 3x3 -> 1x1 expand, plus a skip connection: out = relu(F(x) + x)."""

    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, bias=False)  # spatial conv (stride here = "v1.5")
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)  # expand channels x4
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # skip path must match the main path's shape when channels or resolution change
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))  # no ReLU yet: add the skip first

        return self.relu(out + identity)


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, zero_init_residual=False):
        super().__init__()
        # stem: 224x224x3 -> 112x112x64 -> 56x56x64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # four stages of (3, 4, 6, 3) bottleneck blocks
        self.in_ch = 64
        self.layer1 = self._make_layer(64, 3, stride=1)   # 56x56x256
        self.layer2 = self._make_layer(128, 4, stride=2)  # 28x28x512
        self.layer3 = self._make_layer(256, 6, stride=2)  # 14x14x1024
        self.layer4 = self._make_layer(512, 3, stride=2)  # 7x7x2048

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 7x7x2048 -> 1x1x2048
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        self._init_weights(zero_init_residual)

    def _make_layer(self, mid_ch, blocks, stride):
        # only the first block of a stage changes resolution/channels
        layers = [Bottleneck(self.in_ch, mid_ch, stride)]
        self.in_ch = mid_ch * Bottleneck.expansion
        layers += [Bottleneck(self.in_ch, mid_ch) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")  # He init
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_residual:  # every block starts as identity -> easier early optimization
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.bn3.weight)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


if __name__ == "__main__":  # architecture summary for the report
    m = ResNet50(num_classes=6).eval()
    x = torch.zeros(1, 3, 224, 224)
    print(f"{'stage':<10}{'output shape':<22}{'params':>12}")
    for name in ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]:
        if name == "fc":
            x = torch.flatten(x, 1)
        x = getattr(m, name)(x)
        print(f"{name:<10}{str(tuple(x.shape)):<22}{sum(p.numel() for p in getattr(m, name).parameters()):>12,}")
    print("total parameters:", f"{sum(p.numel() for p in m.parameters()):,}")
