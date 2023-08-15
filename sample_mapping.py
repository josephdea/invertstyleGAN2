import torchvision
from stylegan import G_synthesis,G_mapping
from dataclasses import dataclass
from SphericalOptimizer import SphericalOptimizer
from pathlib import Path
import numpy as np
import time
import torch
from loss import LossBuilder
from functools import partial
from drive import open_url
import sys
import scipy.linalg as la




def SampleStatistics():
    mapping = G_mapping().cuda()
    with open_url("https://drive.google.com/uc?id=14R6iHGf5iuVx3DMNsACAl7eBr7Vdpd0k", cache_dir='cache', verbose="false") as f:
        mapping.load_state_dict(torch.load(f))
    latent_in1 = torch.randn((200000,512)).to("cuda")
    mapping_out = mapping(latent_in1)
    np.save('z',latent_in1.to("cpu").data.numpy())
    np.save('mapping_z',mapping_out.to("cpu").data.numpy())
