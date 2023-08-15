import sys
import numpy as np
import argparse
import math
import os
import os.path
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import lpips
from model import Generator
from bicubic import BicubicDownSample

class mask_function(nn.Module):
    def __init__(self,mask):
        super(mask_function,self).__init__()
        self.gen_image = None

        #initialize masks and opposition masks for inpainting
        self.mask1 = torch.from_numpy(mask).to("cuda")
        self.mask2 = self.mask1.reshape(256,4,256,4)
        self.mask2 = self.mask2.mean([1,3]).float()
        self.opp_mask1 = torch.ones((1024,1024)).to("cuda") - self.mask1
        self.opp_mask2 = torch.ones((256,256)).to("cuda")- self.mask2
    def set_gen(self,x):
        self.gen_image = x
    def forward(self,x):
        #handle 1024 and 256 dimension images
        if(x.size()[len(x.size())-1] == 1024):
            return torch.mul(self.mask1,x)
        elif(x.size()[len(x.size())-1] == 256):
            return torch.mul(self.mask2,x) + torch.mul(self.opp_mask2,self.gen_image)
            

#approximate the 8 layer neural network with one layer (works best for inversion)
class mapping_proxy(nn.Module):
    def __init__(self,gaussian_ft):
        super(mapping_proxy,self).__init__()
        self.mean = gaussian_ft["mean"]
        self.std = gaussian_ft["std"]
        self.lrelu = torch.nn.LeakyReLU(0.2)
    def forward(self,x):
        x = self.lrelu(self.std * x + self.mean)
        return x

#distance on the sphere
def loss_geocross(latent):
        if(latent.size() == (1,512)):
            return 0
        else:
            X = latent.view(-1, 1, 18, 512)
            Y = latent.view(-1, 18, 1, 512)
            A = ((X-Y).pow(2).sum(-1)+1e-9).sqrt()
            B = ((X+Y).pow(2).sum(-1)+1e-9).sqrt()
            D = 2*torch.atan2(A, B)
            D = ((D.pow(2)*512).mean((1, 2))/8.).sum()
            return D
        
#since prior is a bunch of gaussians in high dimensions, approximately lie on the unit sphere
class project_sphere():
    def __init__(self,params):
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt().to("cuda") for param in params}
        for i in self.radii:
            if(i.size() == (1,18,512)):
                tmp_radius = np.full(shape=(1,18,1),fill_value = 23.2)
                self.radii[i] = torch.from_numpy(tmp_radius).to("cuda")
    @torch.no_grad()
    def project(self):
        for param in self.params:
            if(param.size() == (1,18,512)):
                current_radius = param.pow(2).sum(tuple(range(2,param.ndim))).sqrt()
                for i in range(18):
                    if(current_radius[0][i] > self.radii[param][0][i][0]):
                        param[0][i].data.div_(current_radius[0][i])
                        param[0][i].mul_(self.radii[param][0][i][0])
            else:
                current_radius = param.pow(2).sum(tuple(range(2,param.ndim)))
                if(current_radius > self.radii[param]):
                    param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
                    param.mul_(self.radii[param])


class SphericalOptimizer(): 
    def __init__(self, params):
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt() for param in params}
    @torch.no_grad()
    def step(self, closure=None):
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)),keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii[param])


def compute_z_from_w(w,gaussian_ft):
    lrelu = torch.nn.LeakyReLU(5)
    z = lrelu(w) - gaussian_ft["mean"]
    z = z / gaussian_ft["std"]
    return z

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


def invert_depixelization():
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    args = parser.parse_args()

    n_mean_latent = 100000

    resize = min(args.size, 256)

    #lpips works best in 256x256 space, so best to downsample or upsample image to this space
    down_transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    or_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    if(args.mask != None):
        mask = np.load(args.mask)
    else:
        mask = np.ones((1024,1024))
    if(args.reconstruction == 'inpaint'):
        mm = mask_function(mask)
    else:
        mm = BicubicDownSample(factor = 1024//32)
    imgs = []
    originals = []
    for imgfile in args.files:
        img = down_transform(Image.open(imgfile).convert("RGB"))
        or_img = or_transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)
        originals.append(or_img)

    imgs = torch.stack(imgs, 0).to(device)
    originals = torch.stack(originals,0).to(device)
    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    steps = args.step
    lr_func = lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10)
    with torch.no_grad():
        if(args.space == 'w'):
            if(not os.path.isfile('./w_fit.pt')):
                noise_sample = torch.randn(n_mean_latent, 512, device=device)
                latent_out = g_ema.style(noise_sample)
                latent_mean = latent_out.mean(0)
                latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
                mapping_stats = {"mean":latent_mean,"std":latent_std}
                torch.save(mapping_stats,'w_fit.pt')
            else:
                w_fit = torch.load('./w_fit.pt')
                latent_mean = w_fit["mean"]
                latent_std = w_fit["std"]
        else:
            if(not os.path.isfile('./gaussian_fit.pt')):
                noise_sample = torch.randn(n_mean_latent, 512, device=device)
                latent_out = g_ema.style(noise_sample)
                inverse_lrelu = torch.nn.LeakyReLU(negative_slope = 5)
                latent_out = inverse_lrelu(latent_out)
                latent_mean = latent_out.mean(0)
                latent_std = latent_out.std(0)
                mapping_stats = {"mean":latent_mean,"std":latent_std}
                torch.save(mapping_stats,'gaussian_fit.pt')
            gf = torch.load('./gaussian_fit.pt')
            w_fit = torch.load('./w_fit.pt')
            w_mean = w_fit["mean"]
            #latent_mean = compute_z_from_w(w_mean,gf)/bi
            latent_mean = torch.randn((512)).to("cuda")
            latent_std = w_fit["std"]
            mapping_proxy_layer = mapping_proxy(torch.load('gaussian_fit.pt'))  
    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    if(args.space == 'w'):
        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)
        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
    else:
        latent_in = torch.randn(
                    (1, 18, 512), dtype=torch.float, requires_grad=True, device='cuda')
    latent_in.requires_grad = True

    for noise in noises[0:args.num_noises]:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises[0:args.num_noises], lr=args.lr)
    #ps = project_sphere([latent_in] + noises[0:args.num_noises])
    ps = SphericalOptimizer([latent_in] + noises[0:args.num_noises])
    pbar = tqdm(range(args.step))
    latent_path = []
    if(args.lr_scheduler != 'rampdown'):
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    mse_min = np.inf
    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        if(args.lr_scheduler != 'rampdown'):
            optimizer.param_groups[0]["lr"] = lr
        noise_strength = 0*latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            
        #latent_n = latent_noise(latent_in, noise_strength.item())
        if(args.space == 'z'):
            latent_n = mapping_proxy_layer(latent_in) 
        else:
            latent_n = latent_noise(latent_in, noise_strength.item())
    
        img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

        batch, channel, height, width = img_gen.shape
        mse_loss = F.mse_loss(mm(img_gen), originals)
        p_loss = percept(mm(img_gen),originals)
        n_loss = noise_regularize(noises)
        loss = args.pe * p_loss +  args.mse * mse_loss +  args.geocross* loss_geocross(latent_in)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(args.lr_scheduler != 'rampdown'):
            scheduler.step()
        if(args.project):
            ps.step()
        #noise_normalize_(noises[0:args.num_noises])
        if(mse_loss < mse_min):
        #if (i + 1) % 10 == 0:
            mse_min = mse_loss
            if(args.space == 'z'):
                #latent_path.append(latent_n)
                latent_path.append(mapping_proxy_layer(latent_in).detach().clone())
            else:
                latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
            )
        )
    img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)

    filename = os.path.splitext(os.path.basename(args.files[0]))[0] + ".pt"

    img_ar = make_image(img_gen)

    result_file = {}
    for i, input_name in enumerate(args.files):
        noise_single = []
        for noise in noises:
            noise_single.append(noise[i : i + 1])

        result_file[input_name] = {
            "img": img_gen[i],
            "latent": latent_in[i],
            "noise": noise_single,
        }

        img_name = os.path.splitext(os.path.basename(input_name))[0] + "-project.png"
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(img_name)
    torch.save(result_file, filename)
