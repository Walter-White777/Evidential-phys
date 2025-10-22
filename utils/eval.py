import os
import pickle
import numpy as np
import scipy.stats
import sklearn.metrics
import torch

from tqdm import tqdm

from rf import organizer as org
from rf.proc import create_fast_slow_matrix, find_range
from .errors import getErrors
from .utils import extract_video, pulse_rate_from_power_spectral_density
from .nig import *

def eval_uncertainty_model_CA(root_path, test_files, model, sequence_length = 128, 
                  adc_samples = 256, rf_window_size = 5, freq_slope=60.012e12, 
                  samp_f=5e6, sampling_ratio = 4, device=torch.device('cuda')):

    """
    Args:
        root_path: SIGRAPH_data/*
        test_files: ['v_1_1',....]
    
    """
    model.eval()
    video_samples = []
    for folder in tqdm(test_files, total=len(test_files)):
        # print(folder)
        rf_folder = folder[2:]
        rgb_folder = folder
        # load ground truth
        signal = np.load(f"{root_path}/rf_files/{rf_folder}/vital_dict.npy", allow_pickle=True).item()['rgbd']['NOM_PLETHWaveExport']

        # load frames
        video_path = os.path.join(root_path, 'rgb_files', rgb_folder)
        video = extract_video(path=video_path, file_str='rgbd_rgb')

        # load RF
        rf_fptr = open(os.path.join(root_path, 'rf_files', rf_folder, "rf.pkl"),'rb')
        s = pickle.load(rf_fptr)

        # Number of samples is set ot 256 for our experiments
        rf_organizer = org.Organizer(s, 1, 1, 1, 2*adc_samples) 
        frames = rf_organizer.organize()
        # The RF read adds zero alternatively to the samples. Remove these zeros.
        frames = frames[:,:,:,0::2] 

        data_f = create_fast_slow_matrix(frames)
        range_index = find_range(data_f, samp_f, freq_slope, adc_samples)
        temp_window = np.blackman(rf_window_size)
        raw_data = data_f[:, range_index-len(temp_window)//2:range_index+len(temp_window)//2 + 1]
        circ_buffer = raw_data[0:800]
        
        # Concatenate extra to generate ppgs of size 3600
        raw_data = np.concatenate((raw_data, circ_buffer))
        raw_data = np.array([np.real(raw_data),  np.imag(raw_data)])
        raw_data = np.transpose(raw_data, axes=(0,2,1))
        rf_data = raw_data

        rf_data = np.transpose(rf_data, axes=(2,0,1))
        cur_video_sample = {}

        cur_est_ppgs = None

        for cur_frame_num in range(video.shape[0]):
            # Preprocess
            # For rf
            cur_frame_rf = rf_data[cur_frame_num*sampling_ratio:(cur_frame_num+1)*sampling_ratio, :, :]
            cur_frame_rf = torch.tensor(cur_frame_rf).type(torch.float32)/1.255e5  # Normalize
            # For RGB
            cur_frame_rgb = video[cur_frame_num, :, :, :]
            cur_frame_cropped_rgb = torch.from_numpy(cur_frame_rgb.astype(np.uint8)).permute(2, 0, 1).float()
            cur_frame_cropped_rgb = cur_frame_cropped_rgb / 255  # Normalize
            # Add the T dim
            cur_frame_cropped_rgb = cur_frame_cropped_rgb.unsqueeze(0).to(device) 
            cur_frame_rf = cur_frame_rf.to(device)

            # Concat
            if cur_frame_num % sequence_length == 0:
                cur_cat_frames_rf = cur_frame_rf
                cur_cat_frames_rgb = cur_frame_cropped_rgb
            else:
                # assert cur_cat_frames_rf.shape == cur_frame_rf.shape, f'expected shape of {cur_cat_frames_rf.shape}, but got {cur_frame_rf.shape}.'
                cur_cat_frames_rf = torch.cat((cur_cat_frames_rf, cur_frame_rf), 0)
                cur_cat_frames_rgb = torch.cat((cur_cat_frames_rgb, cur_frame_cropped_rgb), 0)

            # Test the performance
            # assert cur_cat_frames_rf.shape[0] == sequence_length*sampling_ratio, f'Expected RF length:{sequence_length*sampling_ratio}, but get {cur_cat_frames_rf.shape[0]}.'
            # assert cur_cat_frames_rgb.shape[0] == sequence_length, f'Expected Video length:{sequence_length}, but get {cur_cat_frames_rgb.shape[0]}.'
            if cur_cat_frames_rf.shape[0] == sequence_length*sampling_ratio and cur_cat_frames_rgb.shape[0] == sequence_length:
                # DL
                with torch.no_grad():
                    # Add the B dim
                    cur_cat_frames_rf = cur_cat_frames_rf.unsqueeze(0)
                    cur_cat_frames_rf = torch.transpose(cur_cat_frames_rf, 1, 2)
                    cur_cat_frames_rf = torch.transpose(cur_cat_frames_rf, 2, 3)
                    IQ_frames = torch.reshape(cur_cat_frames_rf, (cur_cat_frames_rf.shape[0], -1, cur_cat_frames_rf.shape[3]))
                    # cur_est_ppg, _ = model(IQ_frames)
                    cur_cat_frames_rgb = cur_cat_frames_rgb.unsqueeze(0)
                    cur_cat_frames_rgb = torch.transpose(cur_cat_frames_rgb, 1, 2)

                    # uncertainty version
                    # cur_est_ppg = model(IQ_frames)[0]
                    # cur_est_ppg = model(IQ_frames)[1] # use gamma(mean) as rppg
                    rPPG_fusion, rPPG_v, gamma_v, v_v, alpha_v, beta_v, rPPG_r, gamma_r, v_r, alpha_r, beta_r = model(cur_cat_frames_rgb, IQ_frames)
                    
                    # Fusion based on Epistemic Uncertainty
                    # cur_est_ppg = moe_NIG(gamma_v, v_v, alpha_v, beta_v, gamma_r, v_r, alpha_r, beta_r)[0]
                    # cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()
                    cur_est_ppg = rPPG_fusion.squeeze().cpu().numpy()
                    
                # First seq
                if cur_est_ppgs is None: 
                    cur_est_ppgs = cur_est_ppg
                else:
                    cur_est_ppgs = np.concatenate((cur_est_ppgs, cur_est_ppg), -1)
    
        # Save
        # print(cur_est_ppgs.shape)
        cur_video_sample['est_ppgs'] = cur_est_ppgs[0:900]
        cur_video_sample['gt_ppgs'] = signal[25:]
        video_samples.append(cur_video_sample)
    print('All finished!')

    # Estimate using waveforms

    hr_window_size = 300
    stride = 128
    mae_list = []
    all_hr_est = []
    all_hr_gt = []
    for index, cur_video_sample in enumerate(video_samples):
        cur_est_ppgs = cur_video_sample['est_ppgs']
        # Load GT
        cur_gt_ppgs = cur_video_sample['gt_ppgs']
        cur_est_ppgs = (cur_est_ppgs - np.mean(cur_est_ppgs)) / np.std(cur_est_ppgs)
        cur_gt_ppgs = (cur_gt_ppgs - np.mean(cur_gt_ppgs)) / np.std(cur_gt_ppgs)

        # Get est HR for each window
        hr_est_temp = []
        hr_gt_temp = []
        for start in range(0, len(cur_est_ppgs) - hr_window_size, stride):
            ppg_est_window = cur_est_ppgs[start:start + hr_window_size]
            ppg_gt_window = cur_gt_ppgs[start:start + hr_window_size]

            # Normalize PPG
            ppg_est_window = (ppg_est_window - np.mean(ppg_est_window)) / np.std(ppg_est_window)
            ppg_gt_window = (ppg_gt_window - np.mean(ppg_gt_window)) / np.std(ppg_gt_window)
            hr_est_temp.append(pulse_rate_from_power_spectral_density(
                ppg_est_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
            hr_gt_temp.append(pulse_rate_from_power_spectral_density(
                ppg_gt_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))

        hr_est_windowed = np.array([hr_est_temp])
        hr_gt_windowed = np.array(hr_gt_temp)

        all_hr_est.append(hr_est_temp)
        all_hr_gt.append(hr_gt_temp)


        # Errors
        _, MAE, _, _ = getErrors(hr_est_windowed, hr_gt_windowed)
        mae_list.append(MAE)

    print('Mean MAE:', np.mean(np.array(mae_list)))
    return np.array(mae_list), (all_hr_est, all_hr_gt)



def eval_uncertainty_model_additon(root_path, test_files, model, sequence_length = 128, 
                  adc_samples = 256, rf_window_size = 5, freq_slope=60.012e12, 
                  samp_f=5e6, sampling_ratio = 4, device=torch.device('cuda')):

    """
    Args:
        root_path: SIGRAPH_data/*
        test_files: ['v_1_1',....]
    
    """
    model.eval()
    video_samples = []
    for folder in tqdm(test_files, total=len(test_files)):
        # print(folder)
        rf_folder = folder[2:]
        rgb_folder = folder
        # load ground truth
        signal = np.load(f"{root_path}/rf_files/{rf_folder}/vital_dict.npy", allow_pickle=True).item()['rgbd']['NOM_PLETHWaveExport']

        # load frames
        video_path = os.path.join(root_path, 'rgb_files', rgb_folder)
        video = extract_video(path=video_path, file_str='rgbd_rgb')

        # load RF
        rf_fptr = open(os.path.join(root_path, 'rf_files', rf_folder, "rf.pkl"),'rb')
        s = pickle.load(rf_fptr)

        # Number of samples is set ot 256 for our experiments
        rf_organizer = org.Organizer(s, 1, 1, 1, 2*adc_samples) 
        frames = rf_organizer.organize()
        # The RF read adds zero alternatively to the samples. Remove these zeros.
        frames = frames[:,:,:,0::2] 

        data_f = create_fast_slow_matrix(frames)
        range_index = find_range(data_f, samp_f, freq_slope, adc_samples)
        temp_window = np.blackman(rf_window_size)
        raw_data = data_f[:, range_index-len(temp_window)//2:range_index+len(temp_window)//2 + 1]
        circ_buffer = raw_data[0:800]
        
        # Concatenate extra to generate ppgs of size 3600
        raw_data = np.concatenate((raw_data, circ_buffer))
        raw_data = np.array([np.real(raw_data),  np.imag(raw_data)])
        raw_data = np.transpose(raw_data, axes=(0,2,1))
        rf_data = raw_data

        rf_data = np.transpose(rf_data, axes=(2,0,1))
        cur_video_sample = {}

        cur_est_ppgs = None

        for cur_frame_num in range(video.shape[0]):
            # Preprocess
            # For rf
            cur_frame_rf = rf_data[cur_frame_num*sampling_ratio:(cur_frame_num+1)*sampling_ratio, :, :]
            cur_frame_rf = torch.tensor(cur_frame_rf).type(torch.float32)/1.255e5  # Normalize
            # For RGB
            cur_frame_rgb = video[cur_frame_num, :, :, :]
            cur_frame_cropped_rgb = torch.from_numpy(cur_frame_rgb.astype(np.uint8)).permute(2, 0, 1).float()
            cur_frame_cropped_rgb = cur_frame_cropped_rgb / 255  # Normalize
            # Add the T dim
            cur_frame_cropped_rgb = cur_frame_cropped_rgb.unsqueeze(0).to(device) 
            cur_frame_rf = cur_frame_rf.to(device)

            # Concat
            if cur_frame_num % sequence_length == 0:
                cur_cat_frames_rf = cur_frame_rf
                cur_cat_frames_rgb = cur_frame_cropped_rgb
            else:
                # assert cur_cat_frames_rf.shape == cur_frame_rf.shape, f'expected shape of {cur_cat_frames_rf.shape}, but got {cur_frame_rf.shape}.'
                cur_cat_frames_rf = torch.cat((cur_cat_frames_rf, cur_frame_rf), 0)
                cur_cat_frames_rgb = torch.cat((cur_cat_frames_rgb, cur_frame_cropped_rgb), 0)

            # Test the performance
            # assert cur_cat_frames_rf.shape[0] == sequence_length*sampling_ratio, f'Expected RF length:{sequence_length*sampling_ratio}, but get {cur_cat_frames_rf.shape[0]}.'
            # assert cur_cat_frames_rgb.shape[0] == sequence_length, f'Expected Video length:{sequence_length}, but get {cur_cat_frames_rgb.shape[0]}.'
            if cur_cat_frames_rf.shape[0] == sequence_length*sampling_ratio and cur_cat_frames_rgb.shape[0] == sequence_length:
                # DL
                with torch.no_grad():
                    # Add the B dim
                    cur_cat_frames_rf = cur_cat_frames_rf.unsqueeze(0)
                    cur_cat_frames_rf = torch.transpose(cur_cat_frames_rf, 1, 2)
                    cur_cat_frames_rf = torch.transpose(cur_cat_frames_rf, 2, 3)
                    IQ_frames = torch.reshape(cur_cat_frames_rf, (cur_cat_frames_rf.shape[0], -1, cur_cat_frames_rf.shape[3]))
                    # cur_est_ppg, _ = model(IQ_frames)
                    cur_cat_frames_rgb = cur_cat_frames_rgb.unsqueeze(0)
                    cur_cat_frames_rgb = torch.transpose(cur_cat_frames_rgb, 1, 2)

                    # uncertainty version
                    # cur_est_ppg = model(IQ_frames)[0]
                    # cur_est_ppg = model(IQ_frames)[1] # use gamma(mean) as rppg
                    rPPG_fusion, rPPG_v, gamma_v, v_v, alpha_v, beta_v, rPPG_r, gamma_r, v_r, alpha_r, beta_r = model(cur_cat_frames_rgb, IQ_frames)
                    
                    # Fusion based on Epistemic Uncertainty
                    # cur_est_ppg = moe_NIG(gamma_v, v_v, alpha_v, beta_v, gamma_r, v_r, alpha_r, beta_r)[0]
                    # cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()
                    cur_est_ppg = rPPG_fusion.squeeze().cpu().numpy()
                    
                # First seq
                if cur_est_ppgs is None: 
                    cur_est_ppgs = cur_est_ppg
                else:
                    cur_est_ppgs = np.concatenate((cur_est_ppgs, cur_est_ppg), -1)
    
        # Save
        # print(cur_est_ppgs.shape)
        cur_video_sample['est_ppgs'] = cur_est_ppgs[0:900]
        cur_video_sample['gt_ppgs'] = signal[25:]
        video_samples.append(cur_video_sample)
    print('All finished!')

    # Estimate using waveforms

    hr_window_size = 300
    stride = 128
    mae_list = []
    all_hr_est = []
    all_hr_gt = []
    for index, cur_video_sample in enumerate(video_samples):
        cur_est_ppgs = cur_video_sample['est_ppgs']
        # Load GT
        cur_gt_ppgs = cur_video_sample['gt_ppgs']
        cur_est_ppgs = (cur_est_ppgs - np.mean(cur_est_ppgs)) / np.std(cur_est_ppgs)
        cur_gt_ppgs = (cur_gt_ppgs - np.mean(cur_gt_ppgs)) / np.std(cur_gt_ppgs)

        # Get est HR for each window
        hr_est_temp = []
        hr_gt_temp = []
        for start in range(0, len(cur_est_ppgs) - hr_window_size, stride):
            ppg_est_window = cur_est_ppgs[start:start + hr_window_size]
            ppg_gt_window = cur_gt_ppgs[start:start + hr_window_size]

            # Normalize PPG
            ppg_est_window = (ppg_est_window - np.mean(ppg_est_window)) / np.std(ppg_est_window)
            ppg_gt_window = (ppg_gt_window - np.mean(ppg_gt_window)) / np.std(ppg_gt_window)
            hr_est_temp.append(pulse_rate_from_power_spectral_density(
                ppg_est_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
            hr_gt_temp.append(pulse_rate_from_power_spectral_density(
                ppg_gt_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))

        hr_est_windowed = np.array([hr_est_temp])
        hr_gt_windowed = np.array(hr_gt_temp)

        all_hr_est.append(hr_est_temp)
        all_hr_gt.append(hr_gt_temp)

        # print(hr_est_windowed, hr_gt_windowed)

        # Errors
        _, MAE, _, _ = getErrors(hr_est_windowed, hr_gt_windowed)
        mae_list.append(MAE)

    print('Mean MAE:', np.mean(np.array(mae_list)))
    return np.array(mae_list), (all_hr_est, all_hr_gt)



def eval_uncertainty_model(root_path, test_files, model, sequence_length = 128, 
                  adc_samples = 256, rf_window_size = 5, freq_slope=60.012e12, 
                  samp_f=5e6, sampling_ratio = 4, device=torch.device('cuda')):

    """
    Args:
        root_path: SIGRAPH_data/*
        test_files: ['v_1_1',....]
    
    """
    model.eval()
    video_samples = []
    for folder in tqdm(test_files, total=len(test_files)):
        # print(folder)
        rf_folder = folder[2:]
        rgb_folder = folder
        # load ground truth
        signal = np.load(f"{root_path}/rf_files/{rf_folder}/vital_dict.npy", allow_pickle=True).item()['rgbd']['NOM_PLETHWaveExport']

        # load frames
        video_path = os.path.join(root_path, 'rgb_files', rgb_folder)
        video = extract_video(path=video_path, file_str='rgbd_rgb')

        # load RF
        rf_fptr = open(os.path.join(root_path, 'rf_files', rf_folder, "rf.pkl"),'rb')
        s = pickle.load(rf_fptr)

        # Number of samples is set ot 256 for our experiments
        rf_organizer = org.Organizer(s, 1, 1, 1, 2*adc_samples) 
        frames = rf_organizer.organize()
        # The RF read adds zero alternatively to the samples. Remove these zeros.
        frames = frames[:,:,:,0::2] 

        data_f = create_fast_slow_matrix(frames)
        range_index = find_range(data_f, samp_f, freq_slope, adc_samples)
        temp_window = np.blackman(rf_window_size)
        raw_data = data_f[:, range_index-len(temp_window)//2:range_index+len(temp_window)//2 + 1]
        circ_buffer = raw_data[0:800]
        
        # Concatenate extra to generate ppgs of size 3600
        raw_data = np.concatenate((raw_data, circ_buffer))
        raw_data = np.array([np.real(raw_data),  np.imag(raw_data)])
        raw_data = np.transpose(raw_data, axes=(0,2,1))
        rf_data = raw_data

        rf_data = np.transpose(rf_data, axes=(2,0,1))
        cur_video_sample = {}

        cur_est_ppgs = None

        for cur_frame_num in range(video.shape[0]):
            # Preprocess
            # For rf
            cur_frame_rf = rf_data[cur_frame_num*sampling_ratio:(cur_frame_num+1)*sampling_ratio, :, :]
            cur_frame_rf = torch.tensor(cur_frame_rf).type(torch.float32)/1.255e5  # Normalize
            # For RGB
            cur_frame_rgb = video[cur_frame_num, :, :, :]
            cur_frame_cropped_rgb = torch.from_numpy(cur_frame_rgb.astype(np.uint8)).permute(2, 0, 1).float()
            cur_frame_cropped_rgb = cur_frame_cropped_rgb / 255  # Normalize
            # Add the T dim
            cur_frame_cropped_rgb = cur_frame_cropped_rgb.unsqueeze(0).to(device) 
            cur_frame_rf = cur_frame_rf.to(device)

            # Concat
            if cur_frame_num % sequence_length == 0:
                cur_cat_frames_rf = cur_frame_rf
                cur_cat_frames_rgb = cur_frame_cropped_rgb
            else:
                # assert cur_cat_frames_rf.shape == cur_frame_rf.shape, f'expected shape of {cur_cat_frames_rf.shape}, but got {cur_frame_rf.shape}.'
                cur_cat_frames_rf = torch.cat((cur_cat_frames_rf, cur_frame_rf), 0)
                cur_cat_frames_rgb = torch.cat((cur_cat_frames_rgb, cur_frame_cropped_rgb), 0)

            # Test the performance
            # assert cur_cat_frames_rf.shape[0] == sequence_length*sampling_ratio, f'Expected RF length:{sequence_length*sampling_ratio}, but get {cur_cat_frames_rf.shape[0]}.'
            # assert cur_cat_frames_rgb.shape[0] == sequence_length, f'Expected Video length:{sequence_length}, but get {cur_cat_frames_rgb.shape[0]}.'
            if cur_cat_frames_rf.shape[0] == sequence_length*sampling_ratio and cur_cat_frames_rgb.shape[0] == sequence_length:
                # DL
                with torch.no_grad():
                    # Add the B dim
                    cur_cat_frames_rf = cur_cat_frames_rf.unsqueeze(0)
                    cur_cat_frames_rf = torch.transpose(cur_cat_frames_rf, 1, 2)
                    cur_cat_frames_rf = torch.transpose(cur_cat_frames_rf, 2, 3)
                    IQ_frames = torch.reshape(cur_cat_frames_rf, (cur_cat_frames_rf.shape[0], -1, cur_cat_frames_rf.shape[3]))
                    # cur_est_ppg, _ = model(IQ_frames)
                    cur_cat_frames_rgb = cur_cat_frames_rgb.unsqueeze(0)
                    cur_cat_frames_rgb = torch.transpose(cur_cat_frames_rgb, 1, 2)

                    # uncertainty version
                    # cur_est_ppg = model(IQ_frames)[0]
                    # cur_est_ppg = model(IQ_frames)[1] # use gamma(mean) as rppg
                    rPPG_v, gamma_v, v_v, alpha_v, beta_v, rPPG_r, gamma_r, v_r, alpha_r, beta_r = model(cur_cat_frames_rgb, IQ_frames)

                    
                    
                    # Fusion based on Epistemic Uncertainty
                    cur_est_ppg = moe_NIG(gamma_v, v_v, alpha_v, beta_v, gamma_r, v_r, alpha_r, beta_r)[0]
                    cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()

                # First seq
                if cur_est_ppgs is None: 
                    cur_est_ppgs = cur_est_ppg
                else:
                    cur_est_ppgs = np.concatenate((cur_est_ppgs, cur_est_ppg), -1)
    
        # Save
        cur_video_sample['est_ppgs'] = cur_est_ppgs[0:900]
        cur_video_sample['gt_ppgs'] = signal[25:]
        video_samples.append(cur_video_sample)
    print('All finished!')




    # Estimate using waveforms
    hr_window_size = 300
    stride = 128
    mae_list = []
    all_hr_est = []
    all_hr_gt = []
    for index, cur_video_sample in enumerate(video_samples):
        cur_est_ppgs = cur_video_sample['est_ppgs']
        # Load GT
        cur_gt_ppgs = cur_video_sample['gt_ppgs']
        cur_est_ppgs = (cur_est_ppgs - np.mean(cur_est_ppgs)) / np.std(cur_est_ppgs)
        cur_gt_ppgs = (cur_gt_ppgs - np.mean(cur_gt_ppgs)) / np.std(cur_gt_ppgs)

        # Get est HR for each window
        hr_est_temp = []
        hr_gt_temp = []
        for start in range(0, len(cur_est_ppgs) - hr_window_size, stride):
            ppg_est_window = cur_est_ppgs[start:start + hr_window_size]
            ppg_gt_window = cur_gt_ppgs[start:start + hr_window_size]

            # Normalize PPG
            ppg_est_window = (ppg_est_window - np.mean(ppg_est_window)) / np.std(ppg_est_window)
            ppg_gt_window = (ppg_gt_window - np.mean(ppg_gt_window)) / np.std(ppg_gt_window)
            hr_est_temp.append(pulse_rate_from_power_spectral_density(
                ppg_est_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
            hr_gt_temp.append(pulse_rate_from_power_spectral_density(
                ppg_gt_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))

        hr_est_windowed = np.array([hr_est_temp])
        hr_gt_windowed = np.array(hr_gt_temp)

        all_hr_est.append(hr_est_temp)
        all_hr_gt.append(hr_gt_temp)

        # Errors
        _, MAE, _, _ = getErrors(hr_est_windowed, hr_gt_windowed)
        mae_list.append(MAE)

    print('Mean MAE:', np.mean(np.array(mae_list)))
    return np.array(mae_list), (all_hr_est, all_hr_gt)


def eval_performance(hr_est, hr_gt):
    print(hr_est.shape, hr_gt.shape)
    
    hr_est = np.reshape(hr_est, (-1))
    hr_gt  = np.reshape(hr_gt, (-1))
    r = scipy.stats.pearsonr(hr_est, hr_gt)
    mae = np.sum(np.abs(hr_est - hr_gt))/len(hr_est)
    hr_std = np.std(hr_est - hr_gt)
    hr_rmse = np.sqrt(np.sum(np.square(hr_est-hr_gt))/len(hr_est))
    hr_mape = sklearn.metrics.mean_absolute_percentage_error(hr_est, hr_gt)

    return mae, hr_mape, hr_rmse, hr_std, r[0]

def eval_performance_bias(hr_est, hr_gt, fitz_labels_path, session_names):
    l_m_d_arr = get_mapped_fitz_labels(fitz_labels_path , session_names)
    l_m_d_arr = np.array(l_m_d_arr)

    general_performance = eval_performance(hr_est, hr_gt)
    l_p = np.array(eval_performance(hr_est[np.where(l_m_d_arr == 1)], hr_gt[np.where(l_m_d_arr == 1)]))
    d_p = np.array(eval_performance(hr_est[np.where(l_m_d_arr == 2)], hr_gt[np.where(l_m_d_arr == 2)]))

    performance_diffs = np.array([l_p-d_p])
    performance_diffs = np.abs(performance_diffs)
    performance_max_diffs = performance_diffs.max(axis=0)

    print("General Performance")
    print(general_performance)
    print("Performance Max Differences")
    print(performance_max_diffs)
    print("Performance By Skin Tone")
    print("Light - ", l_p)
    print("Dark - ", d_p)

    return general_performance, performance_max_diffs

def eval_clinical_performance(hr_est, hr_gt, fitz_labels_path, session_names):
    l_m_d_arr = get_mapped_fitz_labels(fitz_labels_path , session_names)
    l_m_d_arr = np.array(l_m_d_arr)
    #absolute percentage error
    # print(hr_gt.shape, hr_est.shape)
    apes = np.abs(hr_gt - hr_est)/hr_gt*100
    # print(apes)
    l_apes = np.reshape(apes[np.where(l_m_d_arr==1)], (-1))
    d_apes = np.reshape(apes[np.where(l_m_d_arr==2)], (-1))

    l_5 = len(l_apes[l_apes <= 5])/len(l_apes)*100 
    d_5 = len(d_apes[d_apes <= 5])/len(d_apes)*100
    
    l_10 = len(l_apes[l_apes <= 10])/len(l_apes)*100
    d_10 = len(d_apes[d_apes <= 10])/len(d_apes)*100

    print("AAMI Standard - L,D")
    print(l_10, d_10)
# Test Functions
def get_mapped_fitz_labels(fitz_labels_path, session_names):
    with open(fitz_labels_path, "rb") as fpf:
        out = pickle.load(fpf)

    #mae_list
    #session_names
    sess_w_fitz = []
    fitz_dict = dict(out)
    l_m_d_arr = []
    for i, sess in enumerate(session_names):
        pid = sess.split("_")
        pid = pid[0] + "_" + pid[1]
        fitz_id = fitz_dict[pid]
        if(fitz_id < 3):
            l_m_d_arr.append(1)
        elif(fitz_id < 5):
            l_m_d_arr.append(-1)
        else:
            l_m_d_arr.append(2)
    return l_m_d_arr