import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class ResBlock(nn.Module):
    """
    ResNet Basic Block with residual connection.
    This is the standard ResNet building block for ResNet-18 and ResNet-34.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample layer for dimension matching (if needed)
        self.downsample = downsample
        
    def forward(self, x):
        # Store input for potential downsampling
        identity = x
        
        # First conv + bn + swish
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.silu(out)  # <-- PERUBAHAN 1: ReLU diganti ke SiLU (Swish)
        
        # Second conv + bn
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsample to identity if needed (for dimension matching)
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        # Add residual connection (skip connection)
        out += identity
        out = F.silu(out)  # <-- PERUBAHAN 2: ReLU diganti ke SiLU (Swish)
        
        return out

class ResNet34(nn.Module):
    """
    ResNet-34 Network with Swish activation.
    """
    
    def __init__(self, num_classes=5):
        super(ResNet34, self).__init__()
        
        # Initial convolutional layer (KERNEL DAN STRIDE DARI PERMINTAAN SEBELUMNYA)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=8, stride=4, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Plain block stages
        self.stage1 = self._make_stage(64, 64, 3, stride=1)
        self.stage2 = self._make_stage(64, 128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Catatan: Kaiming init dengan nonlinearity='relu' tetap merupakan
                #         pilihan yang baik dan umum untuk aktivasi sejenis Swish.
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial conv + bn + swish + maxpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.silu(x)      # <-- PERUBAHAN 3: ReLU diganti ke SiLU (Swish)
        x = self.maxpool(x)
        
        # Plain block stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Final classification layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def create_resnet34(num_classes=5):
    return ResNet34(num_classes=num_classes)

def test_model():
    print("Creating ResNet-34 model with Swish activation...")
    model = create_resnet34(num_classes=5)
    
    print("\n" + "="*50)
    print("RESNET-34 (SWISH) MODEL ARCHITECTURE SUMMARY")
    print("="*50)
    
    try:
        summary(model, input_size=(1, 3, 224, 224), verbose=1)
    except Exception as e:
        print(f"Error in torchinfo summary: {e}")
        # Manual test
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            output = model(test_input)
            print(f"Input shape: {test_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Model works correctly: {output.shape == (1, 5)}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

if __name__ == "__main__":
    model = test_model()
    print("\n" + "="*50)
    print("MODEL READY FOR TRAINING!")
    print("="*50)