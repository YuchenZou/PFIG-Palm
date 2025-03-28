import torch

def fourier_transform(image):
    f = torch.fft.fft2(image)
    magnitude = torch.abs(f)
    phase = torch.angle(f)
    return magnitude, phase


def ifft_transform(magnitude, phase):
    complex_tensor = magnitude * torch.exp(1j * phase)
    ifft_result = torch.fft.ifft2(complex_tensor)
    # ifft_result = torch.clamp(ifft_result.real, -1, 1)
    return ifft_result.real


def replace_phase(phase_spectrum1, phase_spectrum2, radius_ratio=0.99):
    batch_size, channels, height, width = phase_spectrum1.shape

    center_row = height // 2
    center_col = width // 2

    radius = int(round(radius_ratio*center_row))

    mask = torch.zeros_like(phase_spectrum1, dtype=torch.bool)
    mask[:, :, center_row - radius:center_row + radius,
         center_col - radius:center_col + radius] = True

    phase_remix = torch.where(mask, phase_spectrum1, phase_spectrum2)
    return phase_remix

def IDI(ID_img, style_img):

    magnitude_ID, phase_ID = fourier_transform(ID_img)
    magnitude_style, phase_style = fourier_transform(style_img)

    phase_remix = replace_phase(phase_ID, phase_style)
    new_img = ifft_transform(magnitude_style, phase_remix)
    return new_img
