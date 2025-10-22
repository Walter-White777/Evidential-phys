import os
import pickle
import numpy as np 
import imageio
import scipy.signal as sig
from torch.utils.data import Dataset
from scipy.io import loadmat
import h5py

import rf.organizer as org
from rf.proc import create_fast_slow_matrix, find_range


def normalize(data:list):
    data = np.array(data)
    # normalized_data = (data -  np.mean(data)) / np.std(data)
    data_mean = np.mean(data)
    data_std = np.std(data)
    normalized_data = (data - data_mean) / data_std
    return normalized_data


class MultiModalityDatasetUCLA(Dataset):
    def __init__(self, data_path, split, frame_length = 128, fs_ppg=30, video_length = 900, \
                 fs_rf = 120, window_size = 5, freq_slope=60.012e12, sample_rate_rf=5e6, chirp_samples=256):
        
        self.data_path = data_path
        self.split = split

        # There is an offset in capturing the signals w.r.t the ground truth.
        self.ppg_offset = 25

        # number of frame for input of model
        self.frame_length = frame_length
        self.video_length = video_length
        # number of samples in one trial.
        self.num_samples = 30
        
        self.rgb_path = os.path.join(data_path, 'rgb_files')
        self.rf_path = os.path.join(data_path, 'rf_files')
        


        # ppg configs
        self.ppg_list = []

        # RF configs
        self.window_size = window_size
        self.fs_rf = fs_rf
        self.freq_slope = freq_slope
        self.sample_rate_rf = sample_rate_rf
        self.sample_ratio = int(fs_rf / fs_ppg)  # fs along slow time axis
        self.chirp_samples = chirp_samples

        

        # load ppgs
        for subject in self.split:

            file_path = os.path.join(self.rgb_path, subject)

            ppg = np.load(os.path.join(file_path, 'rgbd_ppg.npy'))
            self.ppg_list.append(normalize(ppg))


        # normalize PPGs
        self.ppgs = self.ppg_list


        # load RFs and create fast-slow matrix
        self.rfs = []
        for subject in self.split:

            subject_rf = subject[2:]

            file_path = os.path.join(self.rf_path, subject_rf)

            rf_fptr = open(os.path.join(file_path, 'rf.pkl'), 'rb')
            s = pickle.load(rf_fptr)

            # Organize the raw data from the RF.
            # Number of samples is set ot 256 for our experiments.
            rf_organizer = org.Organizer(s, 1, 1, 1, 2*self.chirp_samples)
            frames = rf_organizer.organize()

            # The RF read adds zero alternatively to the samples. Remove these zeros.
            frames = frames[:,:,:,0::2]

            # Process the organized RF data
            data_f = create_fast_slow_matrix(frames)
            range_index = find_range(data_f, self.sample_rate_rf, self.freq_slope, self.chirp_samples)
            # Get the windowed raw data for the network
            raw_data = data_f[:, range_index-self.window_size//2:range_index+self.window_size//2 + 1]  #(rf_length:512, window_size:25cm)
            # Note that item is a complex number due to the nature of the algorithm used to extract and process the pickle file.
            # Hence for simplicity we separate the real and imaginary parts into 2 separate channels.
            raw_data = np.array([np.real(raw_data),  np.imag(raw_data)]) # (2, rf_length, window_size)
            raw_data = np.transpose(raw_data, axes=(0,2,1))  # (2, window_size, rf_length)

            self.rfs.append(raw_data)
        
        # all possible sampling combinations.
        self.all_idxs = []

        for i in range(len(self.split)):

            cur_frames = np.random.randint(low=0, high = self.video_length - self.frame_length - self.ppg_offset, size=self.num_samples)

            rf_cur_frames = cur_frames * self.sample_ratio

            for rf_cur_frame, cur_frame in zip(rf_cur_frames, cur_frames):

                self.all_idxs.append((i, cur_frame, rf_cur_frame))


    def __len__(self):

        return len(self.all_idxs)
    

    def __getitem__(self, idx):


        # video and ppg are synchronized
        index, frame_start, rf_start= self.all_idxs[idx]
        

        # load video
        video = []

        for img_idx in range(self.frame_length):
            image_path = os.path.join(self.rgb_path, 
                                      str(self.split[index]),
                                      f'rgbd_rgb_{frame_start+img_idx}.png'
                                      )
            
            video.append(imageio.imread(image_path))
        video = np.array(video)

        if(len(video.shape)<4):
            video = np.expand_dims(video, axis=3)

        # transpose to (C, T, H, W)
        video = np.transpose(video, axes=(3,0,1,2))

        # video = self.videos[index][frame_start:frame_start+self.frame_length]
        ppg = self.ppgs[index][frame_start:frame_start+self.frame_length]
        rf = self.rfs[index][:,:,rf_start:rf_start+self.sample_ratio*self.frame_length]


        if video.dtype == np.uint16:
            video = video.astype(np.int32)

        assert self.frame_length == len(ppg), f'Expected signal of length {self.frame_length}, but got signal of length {len(ppg)}'

        return np.array(video), np.array(ppg), np.array(rf).astype('float32')
    

class MultiModalityDatasetUCLA_ZJU(Dataset):
    def __init__(self, data_path, split, frame_length = 300, fs_ppg=30, video_length = 5100, \
                 fs_rf = 120, window_size = 5, freq_slope=60.012e12, sample_rate_rf=6e6, chirp_samples=256):
        
        self.data_path = data_path
        self.split = split

        # There is an offset in capturing the signals w.r.t the ground truth.
        self.ppg_offset = 0

        # number of frame for input of model
        self.frame_length = frame_length
        self.video_length = video_length
        # number of samples in one trial.
        self.num_samples = 30
        
        self.rgb_path = os.path.join(data_path, 'rgb_h5_0')
        self.rf_path = os.path.join(data_path, 'rf_files_refined')
        self.ppg_path = os.path.join(data_path, 'ppg')
        


        # ppg configs
        self.ppg_list = []

        # RF configs
        self.window_size = window_size
        self.fs_rf = fs_rf
        self.freq_slope = freq_slope
        self.sample_rate_rf = sample_rate_rf
        self.sample_ratio = int(fs_rf / fs_ppg)  # fs along slow time axis
  

        

        # load ppgs
        for file in self.split:
            "subject: 45_2"
            subject = file.split('_')[0]

            ppg = np.load(os.path.join(self.ppg_path, subject, file+'.npy'))
            self.ppg_list.append(normalize(ppg))


        # normalize PPGs
        self.ppgs = self.ppg_list


        # load RFs and create fast-slow matrix
        self.rfs = []
        self.all_idxs = []

        index = 0 # rf mat file index
        for i in range(len(self.split)):
            subject = self.split[i]
            file_path = os.path.join(self.rf_path, subject)

            time_stamp = np.load(os.path.join(file_path, 'rf_time.npy'))


            for idx, mat_file in enumerate(sorted(os.listdir(file_path))[:-1]):
                cur_time_str = time_stamp[idx]
                h, m, s = cur_time_str.strip().split(':')
                cur_time = int(h)*3600 + int(m)*60 + int(s)

                cur_frame = cur_time * 30

                raw_data = loadmat(os.path.join(file_path, mat_file))['window_profile'][:] #(window_size, T)
                raw_data = raw_data[:, ::85]
            
                raw_data = np.array([np.real(raw_data),  np.imag(raw_data)]) # (2, window_size, T)
                
            # raw_data = np.transpose(raw_data, axes=(0,2,1))  # (2, window_size, rf_length)
                # cur_rf_frame = cur_frame * self.sample_ratio
                self.all_idxs.append(((index, i), (0, cur_frame)))

                self.rfs.append(raw_data)

                index += 1
        
        
    def __len__(self):

        return len(self.all_idxs)
    

    def __getitem__(self, idx):


        # video and ppg are synchronized
        (rf_index, frame_index),  (rf_start, frame_start) = self.all_idxs[idx]
        

        # load video
 

        with h5py.File(os.path.join(self.rgb_path, str(self.split[frame_index])+'.h5'), 'r') as f:
            video = f['imgs'][frame_start:frame_start+self.frame_length]

        # for img_idx in range(self.frame_length):
        #     image_path = os.path.join(self.rgb_path, 
        #                               str(self.split[index]),
        #                               f'rgbd_rgb_{frame_start+img_idx}.png'
        #                               )
            
        #     video.append(imageio.imread(image_path))
        video = np.array(video)

        if(len(video.shape)<4):
            video = np.expand_dims(video, axis=3)

        # transpose to (C, T, H, W)
        video = np.transpose(video, axes=(3,0,1,2))

        # video = self.videos[index][frame_start:frame_start+self.frame_length]
        ppg = self.ppgs[frame_index][frame_start:frame_start+self.frame_length]
        rf = self.rfs[rf_index][:,:,rf_start:rf_start+self.sample_ratio*self.frame_length]


        if video.dtype == np.uint16:
            video = video.astype(np.int32)

        assert self.frame_length == len(ppg), f'Expected signal of length {self.frame_length}, but got signal of length {len(ppg)}'

        if video.shape[1] < self.frame_length:
            print(self.split[frame_index])
        return np.array(video), np.array(ppg), np.array(rf).astype('float32')

