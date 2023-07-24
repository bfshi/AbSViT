import torch

def unnormalize_image(data):
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda().view(3, 1, 1)
    img = (data.squeeze(0) * std + mean)
    return img.permute([1, 2, 0]).detach().cpu().numpy()
