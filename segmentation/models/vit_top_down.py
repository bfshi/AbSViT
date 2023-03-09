import math
import logging
from functools import partial
from functools import reduce
from operator import mul
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Dropout

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_  # ,PatchEmbed
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from torch.nn.modules.utils import _pair

from mmseg.ops import resize
from mmseg.utils import get_root_logger
from mmseg.models.utils.embed import AdaptivePadding
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
                         load_state_dict)

from ..builder import BACKBONES

_logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.adap_padding = AdaptivePadding(kernel_size=patch_size, stride=patch_size, dilation=1, padding='corner')
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.adap_padding(x)
        x = self.proj(x)
        out_size = (x.shape[2], x.shape[3])
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, out_size


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, td=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if td is not None:
            qkv_td = self.qkv(td).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            v = v + qkv_td[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, td=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), td)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class Decode_Block(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.linear = nn.Linear(inplanes, inplanes, bias=False)
        self.linear2 = nn.Linear(inplanes, inplanes, bias=False)

    def forward(self, x):
        x = self.linear(x)
        out = self.linear2(x)
        return x, out


@BACKBONES.register_module
class ViT_Top_Down(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, prompt_config, transfer_type='lienar', img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            weight_init='', embed_layer=PatchEmbed, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=None, block_fn=Block,
            truncate_embedding="none", out_indices=-1, init_cfg=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'

        # # Check prompt config parameters
        # assert prompt_config.LOCATION == "prepend"
        # assert prompt_config.INITIATION == "random"
        # assert prompt_config.NUM_DEEP_LAYERS is None
        # assert not prompt_config.DEEP_SHARED

        self.transfer_type = transfer_type
        self.prompt_config = prompt_config
        self.out_indices = out_indices if out_indices != -1 else [depth-1]
        self.init_cfg = init_cfg
        self.patch_size = patch_size
        self.img_size = img_size

        if transfer_type == 'prompt':
            print(self.prompt_config)
            num_tokens = self.prompt_config['num_tokens']
            self.num_tokens = num_tokens
            self.prompt_dropout = Dropout(self.prompt_config['dropout'])
            if truncate_embedding == "none":
                self.truncate_embedding = depth - 1
            else:
                assert truncate_embedding <= depth - 1, f"number of prompt truncated is {truncate_embedding}, depth is {depth}, prompts truncated exceeds prompt depth"
                self.truncate_embedding = truncate_embedding

            prompt_dim = embed_dim
            self.prompt_proj = nn.Identity()

            # prompt initiation
            patch_size_tuple = _pair(patch_size)
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size_tuple, 1) + prompt_dim))  # noqa
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config['deep']:  # noqa
                total_d_layer = depth - 1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.decoders = nn.ModuleList([Decode_Block(embed_dim) for _ in range(depth)])
        self.prompt = torch.nn.parameter.Parameter(torch.randn(self.embed_dim), requires_grad=True)

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        logger = get_root_logger()
        checkpoint = CheckpointLoader.load_checkpoint(
            self.init_cfg['checkpoint'], logger=logger, map_location='cpu')

        state_dict = checkpoint["model"]

        if 'pos_embed' in state_dict.keys():
            if self.pos_embed.shape != state_dict['pos_embed'].shape:
                logger.info(msg=f'Resize the pos_embed shape from '
                                f'{state_dict["pos_embed"].shape} to '
                                f'{self.pos_embed.shape}')
                h = w = self.img_size
                pos_size = int(
                    math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                state_dict['pos_embed'] = self.resize_pos_embed(
                    state_dict['pos_embed'],
                    (h // self.patch_size, w // self.patch_size),
                    (pos_size, pos_size), 'bicubic')

        load_state_dict(self, state_dict, strict=False, logger=logger)

        if self.transfer_type == 'linear':
            for k, p in self.named_parameters():
                p.requires_grad = False
        elif self.transfer_type == 'prompt':
            for k, p in self.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False
        elif self.transfer_type == 'top-down':
            for k, p in self.named_parameters():
                if "decoders" not in k and "prompt" not in k:
                    p.requires_grad = False
        elif self.transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.
        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        # keep dim for easy deployment
        cls_token_weight = pos_embed[:, 0:1]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def _pos_embed(self, x, hw_shape):
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        pos_embed = self.pos_embed
        x_len, pos_len = x.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size // self.patch_size) * (
                    self.img_size // self.patch_size) + 1:
                pos_h = self.img_size // self.patch_size
                pos_w = self.img_size // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w), 'bicubic')

        x = x + pos_embed
        return self.pos_drop(x)

    def feedback(self, x):
        td = []
        for depth in range(len(self.decoders) - 1, -1, -1):
            x, out = self.decoders[depth](x)
            td = [out] + td
        return td

    def forward(self, x, return_spatial_feature=False):
        x, hw_shape = self.patch_embed(x)
        x = self._pos_embed(x, hw_shape)
        B, _, __ = x.shape
        if self.transfer_type == 'prompt':
            x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        input = x

        # first feedforward
        for i, blk in enumerate(self.blocks):
            if self.transfer_type == 'prompt' and self.prompt_config['deep'] and i != 0:
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                    self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))
                x = torch.cat([x[:, :1, :], deep_prompt_emb, x[:, (1 + self.num_tokens):, :]], dim=1)
            x = blk(x)
        x = self.norm(x)

        # global prior
        cos_sim = F.normalize(x, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        x = x * mask
        td = self.feedback(x)

        # final feedforward
        x = input
        outs = []
        for i, blk in enumerate(self.blocks):
            if self.transfer_type == 'prompt' and self.prompt_config['deep'] and i != 0:
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                    self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))
                x = torch.cat([x[:, :1, :], deep_prompt_emb, x[:, (1 + self.num_tokens):, :]], dim=1)
            x = blk(x, td[i])

            if i == len(self.blocks) - 1:
                x = self.fc_norm(self.norm(x))

            if i in self.out_indices:
                out = x[:, (1 + self.num_tokens):] if self.transfer_type == 'prompt' else x[:, 1:]
                B, N, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()



def vit_tiny_patch16_224(pretrained=False, cfg=None, prompt_cfg=None, **kwargs):
    model = ViT_Top_Down(
        config=cfg, prompt_config=prompt_cfg,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model



def vit_small_patch16_224(pretrained=False, cfg=None, prompt_cfg=None, **kwargs):
    assert cfg is not None, "cfg cannot be None!"
    # assert prompt_cfg is not None, "prompt cfg cannot be None!"
    model = ViT_Top_Down(
        config=cfg, prompt_config=prompt_cfg,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes=-1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.load(cfg.MODEL.MODEL_ROOT, map_location="cpu")
        model.load_state_dict(state_dict["model"], strict=False)
    return model, model.embed_dim



def vit_base_patch16_224(pretrained=False, cfg=None, prompt_cfg=None, **kwargs):
    model = ViT_Top_Down(
        config=cfg, prompt_config=prompt_cfg,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes=-1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.load(cfg.MODEL.MODEL_ROOT, map_location="cpu")
        model.load_state_dict(state_dict["model"], strict=False)
    return model, model.embed_dim

