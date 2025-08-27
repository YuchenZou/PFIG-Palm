#!/usr/bin/env python3
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.G_with_CLIP import Generator
from collections import OrderedDict

class PalmGenerator:
    def __init__(self, model_path, input_nc=3, output_nc=3, cuda=True):
        self.cuda = cuda and torch.cuda.is_available()
        
        self.netG = Generator(input_nc, output_nc)
        state_dict = torch.load(model_path, map_location='cpu' if not self.cuda else None)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        
        self.netG.load_state_dict(new_state_dict)
        
        if self.cuda:
            self.netG.cuda()
        self.netG.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 根据需要调整尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.unnormalize = transforms.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
        )
    
    def generate_palm(self, line_image_path, text_prompts):
        line_image = Image.open(line_image_path).convert('RGB')
        line_tensor = self.transform(line_image).unsqueeze(0)
        
        if self.cuda:
            line_tensor = line_tensor.cuda()
        
        generated_images = []
        
        with torch.no_grad():
            for text_prompt in text_prompts:
                fake_palm, _, _ = self.netG(line_tensor, [text_prompt], mode='text')
                
                fake_palm = self.unnormalize(fake_palm)
                fake_palm = torch.clamp(fake_palm, 0, 1)
                
                img = transforms.ToPILImage()(fake_palm.squeeze(0))
                generated_images.append(img)
        
        return generated_images

def generate_palm_images(line_image_path, text_prompts, model_path, output_dir=None):
    generator = PalmGenerator(model_path)
    
    generated_images = generator.generate_palm(line_image_path, text_prompts)
    
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for i, img in enumerate(generated_images):
            filename = f"generated_palm_{i:03d}.jpg"
            img.save(os.path.join(output_dir, filename))
            print(f"Saved: {filename}")
    
    return generated_images



# Usage example:

if __name__ == '__main__':

    line_image_path = "path/to/your/line_image.jpg"

    text_prompts = [
        "A palmprint in pale tone.",    
        "A palmprint in pale tone, obscured by shadowed regions.",    
        "A palmprint in pale tone with shallow crease.",   

        "A reddish palmprint",
        "A reddish palmprint with uneven illumination.",        
        "A reddish palmprint overexposed under strong lighting.",

        "A palmprint with yellowish tone.",
        "A palmprint with yellowish tone, obscured by blur.",
        "A palmprint with yellowish tone, captured in low light."            
        ]


    model_path = "path/to/your/netG_CLIP.pth"

    output_dir = "output_CLIP"
    
    # 生成图像
    generated_images = generate_palm_images(
        line_image_path=line_image_path,
        text_prompts=text_prompts,
        model_path=model_path,
        output_dir=output_dir
    )
    
    print(f"Generated {len(generated_images)} palm images successfully!")