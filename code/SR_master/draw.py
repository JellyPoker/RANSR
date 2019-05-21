import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

## x2
# models = ['RCAN_BIX2_G10R20P48', 'SURCAN_BIX2_G10R20P48', 'SURCANP_BIX2_G10R20P48']
# marks = ['RCAN', 'SURCAN', 'SURCANP']

## x4
models = ['WRANSR_ABIX2_G4R8P48', 'WRANSR_BBIX2_G4R8P48', 'WRANSR_CBIX2_G4R8P48']
marks = ['WRANSR_A', 'WRANSR_B', 'WRANSR_C']
epoch = 200
sep = 1

num = len(models)

psnr = np.ones([epoch, num], dtype=np.float32)

for i in range(num):
    dir = '../experiment/' + models[i]
    log = torch.load(dir + '/psnr_log.pt')
    y = log.numpy()
    y = y.squeeze()
    y = y[0:epoch]

    y = y[::sep]

    psnr[range(0, epoch, sep), i] = y

# save mat
dataNew = '../experiment/PSNR_x2.mat'
scio.savemat(dataNew, {'PSNR': psnr})







