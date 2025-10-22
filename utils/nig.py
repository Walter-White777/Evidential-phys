import torch
import math
import numpy as np


def compute_uncertainty(gamma, v, alpha, beta):
    """
    Input:
    ----------
        gamma: prediction (mean of mu)
        
    Return:
    ----------
        aleatoric: dataset uncertainty
        epistemic: model uncertainty
    
    """

    epislon = 1e-13
    aleatoric = beta / ((alpha - 1)+epislon)
    epistemic = aleatoric / (v+epislon)

    return aleatoric, epistemic
    

def moe_NIG(gamma1, v1, alpha1, beta1, gamma2, v2, alpha2, beta2):
    
    gamma = (v1 * gamma1 + v2 * gamma2) / (v1 + v2)
    v = v1 + v2
    alpha = alpha1 + alpha2 + 0.5
    beta = beta1 + beta2 + 0.5 * v1 * (gamma1 - gamma) **2 + 0.5 * v2 * (gamma2 - gamma) **2

    return gamma, v, alpha, beta
    
def mixture_NIG(gamma1, v1, alpha1, beta1, gamma2, v2, alpha2, beta2):
    sigma1 = compute_uncertainty(gamma1, v1, alpha1, beta1)[1]
    sigma2 = compute_uncertainty(gamma2, v2, alpha2, beta2)[1]
    gamma = (sigma1 * gamma1 + sigma2 * gamma2) / (sigma1 + sigma2)

    return gamma


def lossNIG(y, gamma, v, alpha, beta, lamda):
    """
    Input:
    ----------
        y: ground truth 
        St parameters m(gamma, v, alpha, beta)


    Return:
    ----------
        loss_NLL: negative log likelihood for model's fitting
        loss_R: minimize evidence on incorrect predictions
    
    """

    loss_NLL = torch.mean(1/2 * torch.log(math.pi / v) - alpha * torch.log(2*beta*(1+v)) +\
                (alpha + 1 / 2) * torch.log((y-gamma)**2 * v + (2*beta*(1+v))) +\
                torch.lgamma(alpha) - torch.lgamma(alpha + 1/2))
    
    loss_R = torch.mean(torch.abs(y - gamma) * (2* v + alpha))

    loss = loss_NLL + lamda * loss_R

    return loss, (loss_NLL, loss_R)




def lossNIG_2(y, gamma, v, alpha, beta, error, lamda):
    """
    Input:
    ----------
        y: ground truth 
        St parameters m(gamma, v, alpha, beta)
        error: error between gt and prediction(NPC, or L2)


    Return:
    ----------
        loss_NLL: negative log likelihood for model's fitting
        loss_R: minimize evidence on incorrect predictions
    
    """

    loss_NLL = torch.mean(1/2 * torch.log(math.pi / v) - alpha * torch.log(2*beta*(1+v)) +\
                (alpha + 1 / 2) * torch.log(error * v + (2*beta*(1+v))) +\
                torch.lgamma(alpha) - torch.lgamma(alpha + 1/2))
    
    loss_R = torch.mean(error * (2* v + alpha))

    loss = loss_NLL + lamda * loss_R


    # need regularization with coefficient lamda
    return loss, (loss_NLL, loss_R)

def quantify_uncertainty_mean(au, eu, mode = 'test'):

    
    if mode == 'test':
        sigma = np.sqrt(np.abs(eu))
    elif mode == 'train':
        sigma = torch.sqrt(torch.abs(eu))


    # epistemic uncertainty(mean value or divergence)

    return au.mean(), sigma.mean()
