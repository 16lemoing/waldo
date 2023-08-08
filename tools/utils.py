import os
import math
import PIL
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import linalg
from collections import OrderedDict
import gzip, pickle, pickletools
from operator import mul
from functools import reduce

import torch
import torchvision
from torchvision import transforms

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def serialize(path, obj):
    with gzip.open(path, "wb") as f:
        pickled = pickle.dumps(obj)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)

def deserialize(path):
    with gzip.open(path, 'rb') as f:
        p = pickle.Unpickler(f)
        return p.load()

def get_vprint(verbose):
    if verbose:
        return lambda s: print(s)
    else:
        return lambda s: None

def to_cuda(tensor_dic, key, flatten_empty=True):
    if key in tensor_dic:
        if isinstance(tensor_dic[key], list):
            return [t.cuda() for t in tensor_dic[key]]
        else:
            if 0 in tensor_dic[key].size() and flatten_empty:
                tensor_dic[key] = torch.Tensor([])
            return tensor_dic[key].cuda()
    return torch.Tensor([])

def flatten(x, ndim):
    shape = None
    if x is not None:
        dim = x.ndim - ndim + 1
        assert dim > 0
        shape = x.shape[:dim]
        size = reduce(mul, shape)
        x = x.reshape(size, *x.shape[dim:])
    return x, shape

def unflatten(x, shape):
    if x is not None:
        return x.reshape(*shape, *x.shape[1:])
    else:
        return x

def flatten_vid(x, vid_ndim=5):
    vid_size = None
    if x.ndim == vid_ndim:
        vid_size = x.shape[:2]
        x = x.reshape(-1, *x.shape[2:])
    return x, vid_size

def unflatten_vid(x, vid_size):
    if vid_size is not None and x.size(0) != 0:
        B, T = vid_size
        return x.reshape(B, T, *x.shape[1:])
    else:
        return x

def to_ctx(tensor, ctx_mask):
    if torch.any(~ctx_mask):
        return tensor[ctx_mask]
    else:
        return tensor.reshape(-1, *tensor.shape[2:])

def from_ctx(ctx_tensor, ctx_mask, pad_value=0):
    if torch.any(~ctx_mask):
        tensor = torch.empty(*ctx_mask.shape, *ctx_tensor.shape[1:], device=ctx_tensor.device).fill_(pad_value)
        tensor[ctx_mask] = ctx_tensor
        return tensor
    else:
        return ctx_tensor.reshape(*ctx_mask.shape, *ctx_tensor.shape[1:])

def onehot(soft_code=None, straight_through=False):
    onehot_code = torch.zeros_like(soft_code)
    onehot_code.scatter_(-1, torch.argmax(soft_code, dim=-1, keepdim=True), 1)
    if straight_through:
        onehot_code = soft_code + (onehot_code - soft_code).detach()
    return onehot_code

def soften(code, codebook_size):
    soft_code = torch.zeros(*code.shape, codebook_size, device=code.device)
    soft_code.scatter_(-1, code.unsqueeze(-1), 1)
    return soft_code

def to_one_hot(code, size, dim):
    shape = code.shape.insert(dim, size)
    return torch.zeros(shape, device=code.device).scatter_(dim, code.unsqueeze(dim), 1)

def to_patch(x, patch_size):
    B, P = x.size(0), patch_size
    x, _ = flatten_vid(x)
    _, C, H, W = x.shape
    return x.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).contiguous().view(B, -1, C, P, P)

def from_patch(x, tgt_size):
    B, C, H, W = tgt_size[0], *tgt_size[-3:]
    P = x.size(-1)
    return x.view(B, -1, H // P, W // P, C, P, P).permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(tgt_size)

def requires_grad(net, flag=True):
    if net is not None:
        for p in net.parameters():
            p.requires_grad = flag

# Taken from https://github.com/mseitzer/pytorch-fid/
def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # msg = ('fid calculation produces singular product; '
        #        'adding %s to diagonal of cov estimates') % eps
        # print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def compute_frechet_distance_from_acts(acts_1, acts_2):
    m1 = np.mean(acts_1, axis=0)
    s1 = np.cov(acts_1, rowvar=False)
    m2 = np.mean(acts_2, axis=0)
    s2 = np.cov(acts_2, rowvar=False)
    fd_value = compute_frechet_distance(m1, s1, m2, s2)
    return fd_value

def load_model(model, path, remove_string=None):
    state_dict = torch.load(path)['state_dict']
    if remove_string is not None:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace(remove_string, "")] = v
    else:
        new_state_dict = state_dict
    model.load_state_dict(new_state_dict)

def color_transfer(im, colormap):
    im = im.cpu().numpy()
    im_new = torch.Tensor(im.shape[0], 3, im.shape[2], im.shape[3])
    newcmp = ListedColormap(colormap)
    for i in range(im.shape[0]):
        img = (im[i, 0, :, :]).astype('uint8')
        rgba_img = newcmp(img)
        rgb_img = PIL.Image.fromarray((255 * np.delete(rgba_img, 3, 2)).astype('uint8'))
        tt = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        rgb_img = tt(rgb_img)
        im_new[i, :, :, :] = rgb_img
    return im_new


class DummyOpt:
    def __init__(self):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass


class DummyDecorator:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass


class DummyScaler:
    def __init__(self):
        pass

    def scale(self, loss):
        return loss

def normalize(tensor, span):
    tensor = tensor.clamp(span[0], span[1])
    tensor = (tensor - span[0]) / (span[1] - span[0])
    return tensor

def dump_image(tensor, path, span=None):
    # Expect tensor of shape (3, H, W) with values in [-1, 1]
    if span is None:
        span = [-1, 1]
    tensor = normalize(tensor, span)
    torchvision.utils.save_image(tensor, path)

def dump_video(tensor, path, span=None, fps=4):
    # Expect tensor of shape (T, 3, H, W) with values in [-1, 1]
    if span is None:
        span = [-1, 1]
    tensor = normalize(tensor, span)
    tensor = (tensor.permute(0, 2, 3, 1) * 255).to(dtype=torch.uint8)
    torchvision.io.write_video(path, tensor, fps)

def dappend(d, val):
    for k in val:
        if k in d:
            d[k].append(val[k])
        else:
            d[k] = [val[k]]

def get_gaussian_kernel(k, sigma_div=6):
    x_cord = torch.arange(k)
    x_grid = x_cord.repeat(k).view(k, k)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (k - 1) / 2.
    sigma = k / sigma_div
    variance = sigma ** 2.
    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel

def get_grid(height, width):
    H, W = height, width
    x = torch.linspace(-1.0 + (1.0 / W), 1.0 - (1.0 / W), W).view(1, 1, -1, 1).expand(-1, H, -1, -1)
    y = torch.linspace(-1.0 + (1.0 / H), 1.0 - (1.0 / H), H).view(1, -1, 1, 1).expand(-1, -1, W, -1)
    return torch.cat([x, y], dim=-1)


def expand(mask, num=1, dir=None, soft=False, alpha=0.97):
    if soft:
        for _ in range(num):
            if not dir or dir == "south":
                mask[:, :, 1:, :] = torch.maximum(mask[:, :, 1:, :], alpha * mask[:, :, :-1, :])
            if not dir or dir == "north":
                mask[:, :, :-1, :] = torch.maximum(mask[:, :, :-1, :], alpha * mask[:, :, 1:, :])
            if not dir or dir == "east":
                mask[:, :, :, 1:] = torch.maximum(mask[:, :, :, 1:], alpha * mask[:, :, :, :-1])
            if not dir or dir == "west":
                mask[:, :, :, :-1] = torch.maximum(mask[:, :, :, :-1], alpha * mask[:, :, :, 1:])
    else:
        mask = mask.bool()
        for _ in range(num):
            if not dir or dir == "south":
                mask[:, :, 1:, :] = mask[:, :, 1:, :] | mask[:, :, :-1, :]
            if not dir or dir == "north":
                mask[:, :, :-1, :] = mask[:, :, :-1, :] | mask[:, :, 1:, :]
            if not dir or dir == "east":
                mask[:, :, :, 1:] = mask[:, :, :, 1:] | mask[:, :, :, :-1]
            if not dir or dir == "west":
                mask[:, :, :, :-1] = mask[:, :, :, :-1] | mask[:, :, :, 1:]
        mask = mask.float()
    return mask
