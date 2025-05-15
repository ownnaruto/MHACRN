import numpy as np
import torch
import torch.nn as nn


def loss_select(name):
    name = name.upper()

    # default loss of each dataset
    if name in ("METRLA", "PEMSBAY", "PEMSD7M", "PEMSD7L"):
        return MaskedMAELoss
    elif name in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        return nn.HuberLoss
    elif name in (
        "ELECTRICITY",
        "EXCHANGE",
        "TRAFFIC",
        "WEATHER",
        "ILI",
        "ETTH1",
        "ETTH2",
        "ETTM1",
        "ETTM2",
    ):
        return nn.MSELoss

    elif name in (
        "MASKEDMAELOSS",
        "MASKED_MAE_LOSS",
        "MASKEDMAE",
        "MASKED_MAE",
        "MASKMAE",
        "MASK_MAE",
        "MMAE",
    ):
        return MaskedMAELoss
    elif name in (
        "HUBERLOSS",
        "HUBER_LOSS",
        "HUBER",
        "SMOOTHEDL1LOSS",
        "SMOOTHED_L1_LOSS",
        "SMOOTHEDL1",
        "SMOOTHED_L1",
    ):
        return nn.HuberLoss
    elif name in ("MAELOSS", "MAE_LOSS", "MAE", "L1LOSS", "L1_LOSS", "L1"):
        return nn.L1Loss
    elif name in ("MSELOSS", "MSE_LOSS", "MSE"):
        return nn.MSELoss

    elif name in ('MASKEDMSELOSS'):
        return MaskedMSELoss 
    
    elif name in ('MASKEDHUBERLOSS'):
        return MaskedHuberLoss  
    
    elif name in ('MYLOSS'):
        return MyLoss 
    else:
        raise NotImplementedError


def masked_mse_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # loss = torch.abs(preds - labels) 
    loss = 0.5 * torch.pow(preds - labels, 2)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_huber_loss(preds, labels, null_val=0.0, delta=1.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.where(torch.abs(preds - labels) < delta, 0.5 * torch.pow(preds - labels, 2), delta * torch.abs(preds - labels) - 0.5 * delta ** 2)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
    

class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val)
    

class MaskedMSELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mse_loss(preds, labels, null_val)
    

class MaskedHuberLoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0, delta=1.0):
        return masked_huber_loss(preds, labels, null_val, delta)


class MyLoss:
    def __init__(self, beta=0.01):
        self.beta = beta 
        
    def _get_name(self):
        return self.__class__.__name__  

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val) + self.beta * masked_mse_loss(preds, labels, null_val)

