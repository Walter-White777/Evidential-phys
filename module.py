import torch
import torch.nn as nn 
import torch.nn.functional as F




class FrequencyEncoder(nn.Module):
    """
    利用一维卷积对频率信号进行编码
    输入形状: (B, C, F) 其中 F 为频率分量数
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(FrequencyEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # x: (B, C, F)
        out = self.conv1(x)  # 卷积提取局部频率模式
        out = self.bn1(out)  # 批归一化
        out = self.relu(out) # 非线性激活
        return out



class TimeFreqInteraction(nn.Module):
    def __init__(self, d):
        super(TimeFreqInteraction, self).__init__()
        self.W_Ft = nn.Linear(d, d, bias=False)  # 时域投影
        self.W_Fs = nn.Linear(d, d, bias=False)  # 频域投影
        self.W_t = nn.Parameter(torch.ones(1))  # 交互强度
        self.W_s = nn.Parameter(torch.ones(1))

    def forward(self, Ft, Fs):
        """
        Ft: (B, T, d)  -> 时域特征
        Fs: (B, F, d)  -> 频域特征
        Returns:
        Ft_new: (B, T, d), Fs_new: (B, F, d)
        """
        B, T, d = Ft.shape
        _, F, _ = Fs.shape

        
        Ft_proj = self.W_Ft(Ft)  # (B, T, d)
        Fs_proj = self.W_Fs(Fs)  # (B, F, d)
        
        interaction_matrix = torch.bmm(Ft_proj, Fs_proj.transpose(1, 2))  # (B, T, F)

        interaction_matrix = interaction_matrix / (interaction_matrix.norm(dim=[1, 2], keepdim=True) + 1e-6)

        Ft_new = Ft + self.W_t * torch.bmm(interaction_matrix, Fs)  # (B, T, d)
        Fs_new = Fs + self.W_s * torch.bmm(interaction_matrix.transpose(1, 2), Ft)  # (B, F, d)

        return Ft_new, Fs_new
    

def compute_batch_psd(s, zero_pad, Fs, high_pass, low_pass):
    """
    Args:
        s:  (B, T)
        zero_pad: 
        Fs: 采样率
        high_pass: 
        low_pass: 
    Returns:
        psds: (B, N)
    """
    B, T = s.shape
    
    psd_list = []
    
    for b in range(B):
        # 取出第 b 个样本的时域信号
        x = s[b]  # (T, )
        # 去均值
        x = x - torch.mean(x)
        
        # 如果需要零填充，则在左右两侧各填充 zero_pad/2*L
        if zero_pad > 0:
            L = x.shape[-1]
            pad_size = int(zero_pad/2 * L)
            x = F.pad(x, (pad_size, pad_size), mode='constant', value=0)
        
        # 计算 FFT 的实部和虚部（采用 rfft 返回复数结果的实部和虚部）
        # 输出形状为 (N, 2) 其中 N = T_new//2 + 1
        fft_out = torch.view_as_real(torch.fft.rfft(x, norm='forward'))
        # 计算 PSD：实部和虚部平方和
        psd = fft_out[:, 0]**2 + fft_out[:, 1]**2  # (N, )
        
        # 计算对应的频率轴
        Fn = Fs / 2  # 奈奎斯特频率
        freqs = torch.linspace(0, Fn, psd.shape[0], device=psd.device)
        # 筛选心率相关的频率范围：将 high_pass 和 low_pass 从 bpm 转换为 Hz
        # 注意：1 bpm = 1/60 Hz
        use_freqs = torch.logical_and(freqs >= (high_pass / 60), freqs <= (low_pass / 60))
        psd = psd[use_freqs]
        
        # 归一化 PSD
        psd = psd / (torch.sum(psd) + 1e-8)
        
        psd_list.append(psd)
    
    # 由于每个样本筛选后的 PSD 长度可能相同，可以直接 stack 成为 (B, N)
    psds = torch.stack(psd_list, dim=0)
    return psds