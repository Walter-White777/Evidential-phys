import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
from PIL import Image
import cv2
import pickle
import matplotlib.pyplot as plt
import glob


# visualize waveform

def data_normalize(x):
    
    return (x - np.mean(x)) / np.std(x)
def visualize_wf(gt, prediction, outputpath):
    """
    visulize the waveform 
    Input:
        numpy array (T,)

    Output:
        saved img
    """
    [T] = gt.shape
    vis_length = 30 * 10 # 10s
    assert gt.shape == prediction.shape 

    start = np.random.choice(T-vis_length)
    end = start + vis_length

    # plt.figure('res', figsize=(8, 3))

    plt.figure('res', figsize=(8, 3))
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.plot(data_normalize(gt[start:end]), label ='GT')
    plt.plot(data_normalize(prediction[start:end]), label ='rPPG')
    plt.legend()

    plt.savefig(outputpath)
    plt.show()

# visualize the similarity between features
def FeatureVisualization(features):

    vis = features.cpu().data.numpy()*1e6 #[B, C, T, H, W]
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    cv2.imwrite('./assets/vis.jpg', vis)



# augmentation for image sequence
def gaussian_blur(images, sigma):

    images = images + np.random.normal(0, 2, images.shape)

    return images


def flip(images):
    mode = np.random.randint(0, 1)
    images = [cv2.flip(i, mode) for i in images]

    return images
    
def rotate(images):
    angle = np.random.choice([0, 90, 180, 270])

    if angle == 0 :
        return images
    if angle == 90:
        images = [cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE) for i in images]
    if angle == 180:
        images = [cv2.rotate(i, cv2.ROTATE_180) for i in images]
    if angle == 270:
        images = [cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE) for i in images]

    return images




def random_crop(images, output_size=64):
    H, W, _ = images[0].shape

    # clip position
    x = np.random.choice(W-output_size)
    y = np.random.choice(H-output_size)


    return images[:, y:y+output_size, x:x+output_size,:]



def random_brightness(images, max_delta = 0.3):
    factor = np.random.uniform(
        np.max(1.0-max_delta, 0), 1.0+max_delta)
    images = images * factor

    images = np.clip(images, 0.0, 1.0)

    images = np.uint8(images * 255.0)

    return images

# augment in a composition
def augment_sequence(frames, aug):


    if 'flip' in aug:
        augmented_seq = flip(frames)
    if 'blur' in aug:
        augmented_seq = gaussian_blur(frames)

    if 'crop' in aug:
        augmented_seq = random_crop(frames)

    if 'brightness' in aug:
        augmented_seq = random_brightness(frames)

    return augmented_seq


def MMSE_split_percentage(k=5, idx=0):

    f_sub_num = list(range(5,20)) + list(range(21,28))
    m_sub_num = list(range(1, 18))

    sub = np.array(['F%03d'%n for n in f_sub_num]+['M%03d'%n for n in m_sub_num])

    rng = np.random.default_rng(12345)
    sub = rng.permutation(sub)

    val_len = len(sub)//k
    sub_val = sub[idx*val_len+1:(idx+1)*val_len+1]
    
    all_files_list = glob.glob('/home/data/rppg/h5_mmse/*h5')

    train_list = []
    val_list = []

    for f_name in all_files_list:
        sub = f_name.split('/')[-1][:4]
        if sub in sub_val:
            val_list.append(f_name)
        else:
            train_list.append(f_name)

    return train_list, val_list


def UBFC_LU_split():
    # split UBFC dataset into training and testing parts
    # the function returns the file paths for the training set and test set.
    # TODO: if you want to train on another dataset, you should define new train-test split function.
    
    h5_dir = '/home/data/rppg/UBFC_h5'
    train_list = []
    val_list = []

    val_subject = [49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38]

    for subject in range(1,50):
        if os.path.isfile(h5_dir+'/%d.h5'%(subject)):
            if subject in val_subject:
                val_list.append(h5_dir+'/%d.h5'%(subject))
            else:
                train_list.append(h5_dir+'/%d.h5'%(subject))

    return train_list, val_list    

def PURE_split():

    h5_dir = '/home/data/rppg/h5_pure'
    train_list = []
    val_list = []

    val_subject = [6, 8, 9, 10]

    for subject in range(1,11):
        for sess in [1,2,3,4,5,6]:
            if os.path.isfile(h5_dir+'/%02d-%02d.h5'%(subject, sess)):
                if subject in val_subject:
                    val_list.append(h5_dir+'/%02d-%02d.h5'%(subject, sess))
                else:
                    train_list.append(h5_dir+'/%02d-%02d.h5'%(subject, sess))
    return train_list, val_list  

def dataset_split(dataset_name):
    """
    return dataset split according to dataset_name
    Args:
        datasetname: str (UBFC, PURE, MMSE, VIPL, OBF)
    """
    
    if dataset_name == 'UBFC':

        return UBFC_LU_split()
    
    if dataset_name == 'PURE':

        return PURE_split()
    
    if dataset_name == 'MMSE':

        return MMSE_split_percentage()


def get_experiment_dir(root):

    if not os.path.exists(root):
        os.makedirs(root)

    dirs = sorted(os.listdir(root))
    if len(dirs) > 0:
        last_number = int(dirs[-1].split('_')[-1])
        last_number += 1
    else:
        last_number = 0
    experiment_dir = os.path.join(root, 'exper_%04d' % last_number)
    os.makedirs(experiment_dir, exist_ok=False)
    return experiment_dir


class H5Dataset(Dataset):

    def __init__(self, train_list, T, augmentation=None, darkness=None):
        self.train_list = train_list # list of .h5 file paths for training
        self.T = T # video clip length
        self.augmentation = augmentation
        self.darkness = darkness
        self.gt_path = '/8T-1/gjy/ZJU_train/ppg'
        self.data_path = '/8T-1/gjy/ZJU_train/rgb_h5'

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):

        session = self.train_list[idx]
        subject = session.split('_')[0]
        gt = np.load(os.path.join(self.gt_path, subject, session+'.npy'))
        gt = data_normalize(gt)
        with h5py.File(os.path.join(self.data_path, session+'.h5'), 'r') as f:
            img_length = np.min([f['imgs'].shape[0], gt.shape[0]])

            idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            
            bvp = gt[idx_start:idx_end].astype('float32')
            
            img_seq = f['imgs'][idx_start:idx_end]
            

            # augment for training, needless in test
            # if self.augmentation!=None and len(self.augmentation)>0:
            #     img_seq_q = augment_sequence(img_seq, aug = self.augmentation)
            #     img_seq_k = augment_sequence(img_seq, aug = self.augmentation)

            # darken the video:
            # if self.darkness != None:
            #     img_seq = self.darkness * img_seq


            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32') # (C, T, H, W)
            
        return img_seq, bvp  # supervised
        # return img_seq_q, img_seq_k # CLR


def loads_data(buf):
    return pickle.loads(buf)

