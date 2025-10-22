import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from module import *

class RF_conv_encoder(torch.nn.Module):
    def __init__(self, channels=10):  
        super(RF_conv_encoder, self).__init__()
        
        self.ConvBlock1 = torch.nn.Sequential(
            torch.nn.Conv1d(channels, 32, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
        )

        self.ConvBlock2 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock4 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 256, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock5_mean = torch.nn.Sequential(
            torch.nn.Conv1d(256, 512, 7, stride=1, padding=3),
        )
        self.downsample1 = torch.nn.MaxPool1d(kernel_size=2)
        self.downsample2 = torch.nn.MaxPool1d(kernel_size=2)
        
    def forward(self, x_orig: torch.tensor) -> torch.tensor:
        x = self.ConvBlock1(x_orig)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.downsample1(x)
        x = self.ConvBlock4(x)
        x = self.downsample2(x)
        x_encoded_mean  = self.ConvBlock5_mean(x)

        return x_encoded_mean

class RF_conv_decoder(torch.nn.Module):
    def __init__(self, channels=10):  
        super(RF_conv_decoder, self).__init__()

        self.IQ_encoder = RF_conv_encoder(channels)
        
        self.ConvBlock1 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(512, 256, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(256, 128, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(128, 64, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock4 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(64, 32, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(32, 1, 1, stride=1, padding=0)
        )
        
    def forward(self, x_IQ: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        z_IQ = self.IQ_encoder(x_IQ)
        x = self.ConvBlock1(z_IQ)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x_decoded = self.ConvBlock5(x)        
        return x_decoded, z_IQ

class PhysNet(nn.Module):
    def __init__(self, S=2, in_ch=3):
        super().__init__()

        self.S = S # S is the spatial dimension of ST-rPPG block

        self.start = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )

        # 1x
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # encoder
        self.encoder1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        #
        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # decoder to reach back initial temporal length
        self.decoder1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )
                    

        self.end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        stds = torch.std(x, dim=(2, 3, 4), keepdim=True)
        x = (x - means) / stds # (B, C, T, 128, 128)

        [batch,channel,length,width,height] = x.shape
 
        parity = []
        x = self.start(x) # (B, C, T, 128, 128)
        x = self.loop1(x) # (B, 64, T, 64, 64)
        parity.append(x.size(2) % 2)
        x = self.encoder1(x) # (B, 64, T/2, 32, 32)
        parity.append(x.size(2) % 2)
        x = self.encoder2(x) # (B, 64, T/4, 16, 16)
        x = self.loop4(x) # (B, 64, T/4, 8, 8)

        x = F.interpolate(x, scale_factor=(2, 1, 1)) # (B, 64, T/2, 8, 8)
        x = self.decoder1(x) # (B, 64, T/2, 8, 8)
        x = F.pad(x, (0,0,0,0,0,parity[-1]), mode='replicate')
        x = F.interpolate(x, scale_factor=(2, 1, 1)) # (B, 64, T, 8, 8)
        x = self.decoder2(x) # (B, 64, T, 8, 8)
        x = F.pad(x, (0,0,0,0,0,parity[-2]), mode='replicate')
        x = self.end(x) # (B, 1, T, 1, 1), ST-rPPG block

        x = x.view(-1, length) # output rppg

        return x





class TrustFusion(nn.Module):
    def __init__(self, d_output=1):
        super(TrustFusion, self).__init__()
        
        self.backbone_rgb = PhysNet(in_ch=3, S=2)
        self.backbone_rf = RF_conv_decoder()
        
        self.d_output = d_output

        self.linear_v = nn.Linear(d_output, 4 * d_output)
        self.linear_r = nn.Linear(d_output, 4 * d_output)

    def evidence(self, x):
        return F.softplus(x)
    
    def get_parameters(self, gamma, v, alpha, beta):

        v = self.evidence(v)
        alpha = self.evidence(alpha) + 1
        beta = self.evidence(beta) 

        return gamma, v, alpha, beta

    def forward(self, v, r):

        """ 
        
        Input:
        ----------
            v: input video (B, C, T, H, W)
            r: input RF range matrix (B, 10, window_size, T)

            
        Return:
        ----------
            rPPG: raw rPPG signal (B, T)
            gamma: mean of NIG distribution (B, T, 1)
            v: variance of NIG distribution
        """


        # rPPG extraction
        rPPG_v = self.backbone_rgb(v) #(B, T)
        rPPG_r = self.backbone_rf(r)[0].squeeze(1) #(B, 1, T)

        # Transform
        output_v = rPPG_v.unsqueeze(-1) #(B, T, 1)
        output_v = self.linear_v(output_v) #(B, T, 4)

        output_r = rPPG_r.unsqueeze(-1) #(B, T, 1)
        output_r = self.linear_r(output_r) #(B, T, 4)
        
        # Uncertainty Head
        # get NIG parameters
        gamma_v, v_v, alpha_v, beta_v = torch.split(output_v, 1, dim=-1) #(B, T, 1)
        gamma_v, v_v, alpha_v, beta_v = self.get_parameters(gamma_v, v_v, alpha_v, beta_v)

        gamma_r, v_r, alpha_r, beta_r = torch.split(output_r, 1, dim=-1) #(B, T, 1)
        gamma_r, v_r, alpha_r, beta_r = self.get_parameters(gamma_r, v_r, alpha_r, beta_r)


        return rPPG_v, gamma_v, v_v, alpha_v, beta_v, rPPG_r, gamma_r, v_r, alpha_r, beta_r
    







class TrustAdaptiveFusionFT(nn.Module):
    def __init__(self, d_output=1):
        super(TrustAdaptiveFusionFT, self).__init__()
        
        self.backbone_rgb = PhysNet(in_ch=3, S=2)
        self.backbone_rf = RF_conv_decoder()
        
        self.d_output = d_output

        self.linear_v = nn.Linear(d_output, 4 * d_output)
        self.linear_r = nn.Linear(d_output, 4 * d_output)

        # 频域处理模块
        self.freq_enc_v = FrequencyEncoder(in_channels=1, out_channels=1)
        self.freq_enc_r = FrequencyEncoder(in_channels=1, out_channels=1)

        self.attention = nn.Sequential(
            nn.Linear(2*d_output+1, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

        self.TSF_v = TimeFreqInteraction(d = 1)
        self.TSF_r = TimeFreqInteraction(d = 1)

    

    def evidence(self, x):
        return F.softplus(x)
    
    def get_parameters(self, gamma, v, alpha, beta):
        v = self.evidence(v)
        alpha = self.evidence(alpha) + 1
        beta = self.evidence(beta) 
        return gamma, v, alpha, beta

    def forward(self, v, r):
        # 特征提取
        rPPG_v = self.backbone_rgb(v)            # (B, T)
        rPPG_r = self.backbone_rf(r)[0]  # (B, 1, T)

        
        # output_v_si = self._fuse_frequency(rPPG_v.unsqueeze(-1), self.freq_conv_v)  # (B, T, 1)
        # output_r_si = self._fuse_frequency(rPPG_r.unsqueeze(-1), self.freq_conv_r)  # (B, T, 1)
        psd_v, psd_r = compute_batch_psd(rPPG_v), compute_batch_psd(rPPG_r.squeeze(1)) # (B,  F)
        f_freq_v, f_freq_r = self.freq_enc_v(psd_v.unsqueeze(1)), self.freq_enc_r(psd_r.unsqueeze(1))

        # 维度的设置
        
        output_v = self.linear_v(rPPG_v.unsqueeze(1))     # (B, T, 4)
        output_r = self.linear_r(rPPG_r)     # (B, T, 4)

        # 参数分解
        gamma_v, v_v, alpha_v, beta_v = self.get_parameters(*torch.split(output_v, 1, dim=-1))
        gamma_r, v_r, alpha_r, beta_r = self.get_parameters(*torch.split(output_r, 1, dim=-1))

        
        # fusion features of every modality
        output_v_si = self.TSF_r(rPPG_v.unsqueeze(1), f_freq_v)
        output_r_si = self.TSF_v(rPPG_r, f_freq_r)


       
        score_v = self.attention(torch.cat([v_v, output_v_si, output_r_si], -1))
        score_r = self.attention(torch.cat([v_r, output_r_si, output_v_si], -1))
        a_weights = F.softmax(torch.cat([score_v, score_r], -1), dim=-1)

        # 自适应融合
        a_v, a_r = a_weights[:, 0], a_weights[:, 1]
        
        gamma = a_v*gamma_v + a_r*gamma_r
        return (rPPG_v, gamma_v, v_v, alpha_v, beta_v, 
                rPPG_r, gamma_r, v_r, alpha_r, beta_r,
                gamma)