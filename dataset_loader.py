import ast

#import tensorflow as tf
import random
import scipy.signal
import os
import numpy as np
import soundfile as sf
import math
import pandas as pd
import glob
from tqdm import tqdm
import torch

#generator function. It reads the csv file with pandas and loads the largest audio segments from each recording. If extend=False, it will only read the segments with length>length_seg, trim them and yield them with no further processing. Otherwise, if the segment length is inferior, it will extend the length using concatenative synthesis.



class TestDataset(torch.utils.data.Dataset):
    def __init__(self, path_music,  fs=22050, seg_len=5):
                
        test_samples=glob.glob(os.path.join(path_music,"*.wav"))

        self.records=[]
        seg_len=int(seg_len)
        pointer=int(fs*5) #starting at second 5 by default
        for i in tqdm(range(len(test_samples))):
            data, sr=sf.read(test_samples[i])
            if len(data.shape)>1 and not(stereo):
                data=np.mean(data,axis=1)
            if sr !=fs: 
                
                print("resampling", sr, fs, test_samples[i])
                data=scipy.signal.resample(data, int(len(data)*fs/sr))

            #normalize
            data=0.9*(data/np.max(np.abs(data)))

            segment=data[pointer:pointer+seg_len]

            self.records.append(segment.astype("float32"))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


        
#Train dataset object

class TrainDataset (torch.utils.data.IterableDataset):
    def __init__(self, path_music,  fs=22050, seg_len=5,seed=42 ):
        super(TrainDataset).__init__()
        random.seed(seed)
        np.random.seed(seed)
        self.train_samples=[]
        for path in path_music:
            self.train_samples.extend(glob.glob(os.path.join(path ,"*.wav")))
       
        self.seg_len=int(seg_len)
        self.fs=fs

    def __iter__(self):
        while True:
            num=random.randint(0,len(self.train_samples)-1)
            #for file in self.train_samples:  
            file=self.train_samples[num]
            data, samplerate = sf.read(file)
            assert(samplerate==self.fs, "wrong sampling rate")
            data_clean=data
            #Stereo to mono
            if len(data.shape)>1 :
                data_clean=np.mean(data_clean,axis=1)
    
            #normalize
            data_clean=data_clean/np.max(np.abs(data_clean))
         
            #framify data clean files
            num_frames=np.floor(len(data_clean)/self.seg_len) 
            if num_frames>1:
                idx=np.random.randint(0,len(data_clean)-self.seg_len)
                segment=data_clean[idx:idx+self.seg_len]
                segment=segment.astype('float32')
                rms=np.sqrt(np.var(segment))
                segment= (0.1/rms)*segment #default rms  of 0.1. Is this scaling correct??
            
                #scale=np.random.uniform(-6,4)
                #segment=10.0**(scale/10.0) *segment

                yield  segment
            else:
                pass
         


             


