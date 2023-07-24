import torch
import torch.distributed as dist
import torch.nn.functional as F

def unnormalize_image(data):
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda().view(3, 1, 1)
    img = (data.squeeze(0) * std + mean)
    return img.permute([1, 2, 0]).detach().cpu().numpy()

def clip_loss(image_embed, text_embed, clip_mask=None):
    logit_scale = 1 / 0.07
    local_batch_size = image_embed.size(0)

    labels = local_batch_size * get_rank() + torch.arange(
        local_batch_size, device=image_embed.device
    )

    # normalized features
    image_embed = F.normalize(image_embed, dim=-1, p=2)
    text_embed = F.normalize(text_embed, dim=-1, p=2)

    # gather features from all GPUs
    image_embed_all, text_embed_all = all_gather_batch([image_embed, text_embed])

    # cosine similarity as logits
    logits_per_image = logit_scale * image_embed @ text_embed_all.t()
    logits_per_text = logit_scale * text_embed @ image_embed_all.t()

    if clip_mask is None:
        loss = (F.cross_entropy(logits_per_image, labels) + \
                F.cross_entropy(logits_per_text, labels)) / 2
    else:
        loss = (F.cross_entropy(logits_per_image, labels, reduction='none') * clip_mask).sum() / clip_mask.sum() / 2 + \
               (F.cross_entropy(logits_per_text, labels, reduction='none') * clip_mask).sum() / clip_mask.sum() / 2

    return loss

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor
