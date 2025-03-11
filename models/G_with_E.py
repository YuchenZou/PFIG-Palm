import torch.nn as nn
import torch

class ResidualBlock_AdaIN(nn.Module):
    def __init__(self, in_features, style_dim, norm):
        super(ResidualBlock_AdaIN, self).__init__()
        self.adain1 = AdaIN(in_features, style_dim, norm)
        self.adain2 = AdaIN(in_features, style_dim, norm)
        conv_block1 = [ nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3)]
        conv_block2 = [ nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3)]
        self.conv_block1 = nn.Sequential(*conv_block1)
        self.conv_block2 = nn.Sequential(*conv_block2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, style):
        x2 = self.conv_block1(x)
        x2 = self.adain1(x2, style)
        x2 = self.relu(x2)
        x2 = self.conv_block2(x2)
        x2 = self.adain2(x2, style)
        return x + x2


class ResidualBlock(nn.Module):
    def __init__(self, in_features, norm):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class AdaIN(nn.Module):
    def __init__(self, channels, style_dim, norm):
        super(AdaIN, self).__init__()
        self.norm = norm(channels)
        self.liner2 = nn.Linear(style_dim, style_dim)
        self.style_scale = nn.Linear(style_dim, channels)
        self.style_shift = nn.Linear(style_dim, channels)

    def forward(self, x, style):
        normalized = self.norm(x)
        style = self.liner2(style)
        style_scale = self.style_scale(style).unsqueeze(2).unsqueeze(3)
        style_shift = self.style_shift(style).unsqueeze(2).unsqueeze(3)
        transformed = style_scale * normalized + style_shift
        return transformed


class DownsamplingBlock(nn.Module):
    def __init__(self, in_features, out_features, style_dim, norm):
        super(DownsamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.adain = AdaIN(out_features, style_dim, norm)
        self.to_noise = nn.Linear(1, out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, style, inoise):
        x = self.conv(x)
        x = self.adain(x, style)
        # inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        # noise = self.to_noise(inoise).permute((0, 3, 2, 1))
        # x = self.relu(x + noise)
        x = self.relu(x)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, in_features, out_features, style_dim, norm):
        super(UpsamplingBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1)
        self.adain = AdaIN(out_features, style_dim, norm)
        self.to_noise = nn.Linear(1, out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, style, inoise):
        x = self.deconv(x)
        x = self.adain(x, style)
        # inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        # noise = self.to_noise(inoise).permute((0, 3, 2, 1))
        # x = self.relu(x + noise)
        x = self.relu(x)
        return x


class StyleEncoder(nn.Module):
    def __init__(self, input_nc, style_dim, norm):
        super(StyleEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_nc, 32, kernel_size=3, stride=2, padding=1),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            norm(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            norm(64),
            nn.ReLU(inplace=True),
        )
        self.fc_mean = nn.Linear(64 * 8 * 8, style_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, style_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar



class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6, norm=nn.BatchNorm2d):
        super(Generator, self).__init__()

        style_dim = 8
        self.styleEncoder = StyleEncoder(input_nc, style_dim, norm=norm)
        style_dim2 = 128
        self.style_liner = nn.Sequential(nn.Linear(style_dim, style_dim2//4), nn.Linear(style_dim2//4, style_dim2))
        # Initial convolution block
        self.initial_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, 64, 3),
        )
        self.initial_adain = AdaIN(64, style_dim2, norm)
        self.initial_relu = nn.ReLU(inplace=True)

        # Downsampling
        in_features = 64
        out_features = in_features*2
        self.downsampling1 = DownsamplingBlock(in_features, out_features, style_dim2, norm=norm)

        in_features = out_features
        out_features = in_features*2
        self.downsampling2 = DownsamplingBlock(in_features, out_features, style_dim2, norm=norm)

        in_features = out_features
        out_features = in_features
        self.downsampling3 = DownsamplingBlock(in_features, out_features, style_dim2, norm=norm)

        # Residual blocks
        in_features = out_features
        self.residual_blocks = nn.ModuleList()
        for _ in range(n_residual_blocks):
            self.residual_blocks.append(ResidualBlock_AdaIN(in_features, style_dim2, norm=norm))

        self.upsampling1 = UpsamplingBlock(in_features, out_features, style_dim2, norm=norm)

        # Upsampling
        out_features = in_features // 2
        self.upsampling2 = UpsamplingBlock(in_features, out_features, style_dim2, norm=norm)

        in_features = out_features
        out_features = in_features // 2
        self.upsampling3 = UpsamplingBlock(in_features, out_features, style_dim2, norm=norm)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, output_nc, 3),
            nn.Tanh()
        )

    def forward(self, x, style_img, z=None):
        if z is None:
            z, mean, logvar = self.styleEncoder(style_img)
        else:
            mean, logvar = None, None
        z = self.style_liner(z)
        noise = torch.rand(x.size(0), 256, 256, 1, device=x.device)
        # Initial convolution block
        x = self.initial_conv(x)
        x = self.initial_adain(x, z)
        x = self.initial_relu(x)
        x = self.downsampling1(x, z, noise)
        x = self.downsampling2(x, z, noise)
        x = self.downsampling3(x, z, noise)
        for block in self.residual_blocks:
            x = block(x, z)
        x = self.upsampling1(x, z, noise)
        x = self.upsampling2(x, z, noise)
        x = self.upsampling3(x, z, noise)
        x = self.output_layer(x)
        return x, mean, logvar

