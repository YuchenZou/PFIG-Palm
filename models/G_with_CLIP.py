import torch.nn as nn
import torch
import clip
from PIL import Image

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


def image_noise(n, im_size, device='cuda'):
    return torch.rand(n, im_size, im_size, 1, device=device)


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


class CLIPStyleEncoder(nn.Module):
    def __init__(self, style_dim, device='cuda'):
        super(CLIPStyleEncoder, self).__init__()
        self.device = device
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Get CLIP embedding dimension (512 for ViT-B/32)
        clip_dim = self.clip_model.visual.output_dim  # 512
        
        # Project CLIP embeddings to style dimension with variational components
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, style_dim * 2)  # *2 for mean and logvar
        )
        
    def encode_image(self, images):
        """Encode images using CLIP visual encoder"""
        with torch.no_grad():
            # images should already be preprocessed and normalized
            image_features = self.clip_model.encode_image(images)
            image_features = image_features.float()
        
        # Project to style space
        projected = self.projection(image_features)
        mean = projected[:, :projected.size(1)//2]
        logvar = projected[:, projected.size(1)//2:]
        
        return mean, logvar
    
    def encode_text(self, text_prompts):
        """Encode text prompts using CLIP text encoder"""
        with torch.no_grad():
            # Tokenize text
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features.float()
        
        # Project to style space
        projected = self.projection(text_features)
        mean = projected[:, :projected.size(1)//2]
        logvar = projected[:, projected.size(1)//2:]
        
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6, norm=nn.BatchNorm2d, device='cuda'):
        super(Generator, self).__init__()
        self.device = device
        
        style_dim = 8
        self.clip_style_encoder = CLIPStyleEncoder(style_dim, device=device)
        
        style_dim2 = 128
        self.style_liner = nn.Sequential(
            nn.Linear(style_dim, style_dim2//4), 
            nn.Linear(style_dim2//4, style_dim2)
        )
        
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

    def forward(self, x, style_input=None, z=None, mode='image'):
        """
        Args:
            x: input image
            style_input: style image (for training) or text prompt (for testing)
            z: pre-computed style vector (optional)
            mode: 'image' for image input, 'text' for text input
        """
        if z is None:
            if mode == 'image' and style_input is not None:
                # Training: extract style from image
                mean, logvar = self.clip_style_encoder.encode_image(style_input)
                z = self.clip_style_encoder.reparameterize(mean, logvar)
            elif mode == 'text' and style_input is not None:
                # Testing: extract style from text
                mean, logvar = self.clip_style_encoder.encode_text(style_input)
                z = self.clip_style_encoder.reparameterize(mean, logvar)
            else:
                raise ValueError("Either z or style_input must be provided")
        else:
            mean, logvar = None, None
            
        z = self.style_liner(z)
        noise = image_noise(x.size(0), 256)
        
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


# Helper function for preprocessing images for CLIP
def preprocess_images_for_clip(images, clip_preprocess):
    """
    Convert tensor images to CLIP-compatible format
    Args:
        images: torch.Tensor of shape (B, C, H, W) in range [-1, 1] or [0, 1]
        clip_preprocess: CLIP preprocessing function
    """
    processed_images = []
    for img in images:
        # Convert from tensor to PIL Image
        if img.min() >= -1 and img.max() <= 1:
            # Assume range is [-1, 1], convert to [0, 1]
            img = (img + 1) / 2
        
        # Convert to PIL Image
        img_pil = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
        
        # Apply CLIP preprocessing
        processed_img = clip_preprocess(img_pil)
        processed_images.append(processed_img)
    
    return torch.stack(processed_images)


