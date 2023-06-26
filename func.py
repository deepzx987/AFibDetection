import numpy as np
import scipy as sp
from scipy import signal
import operator
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.signal import butter, lfilter, freqz, iirnotch, medfilt, iirfilter

def mmf(signal,alpha=0.2):
    return (1-alpha)*np.median(signal) + alpha*np.mean(signal)

def mean_median_filter(signal,window,alpha=0.6):
    
    # Always odd  window 
    if window%2 == 0:
        window = window-1
    
    # Appended signal
    new_signal = []
    for i in range((window-1)//2):
        new_signal.append(signal[0])
    for i in range(len(signal)):
        new_signal.append(signal[i])
    for i in range((window-1)//2):
        new_signal.append(signal[-1])
    
    # Windowing
    mmfoutput = []
    for i in range(len(signal)):
        mmfoutput.append(mmf(new_signal[i:i+window],alpha))
        
    return mmfoutput

def mean_median_filt(signal):
    baseline = mean_median_filter(signal,250,0.6)
    baseline = mean_median_filter(baseline,600,0.6)
    baseline = np.array(baseline)
    return baseline

def Notch_Filter(data, freq_to_remove, sample_freq, order=5, filter_type='butter'):
    fs   = sample_freq
    nyq  = fs/2.0
    low  = freq_to_remove - 1.0
    high = freq_to_remove + 1.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], btype='bandstop',
                     analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def cleaning_signal(path):
    name = path[-7:-4]+'_signal'
    if os.path.isfile(name+'.npy'):
        return np.load(name+'.npy')
    else:
        df = pd.read_csv(path)
        data = (df.iloc[:,1]).values
        data_to_process = data
        data_norm = data_to_process - np.mean(data_to_process)
        pli = Notch_Filter(data=data_norm, freq_to_remove=50.0)
        baseline = mean_median_filter(pli,300,0.6)
        baseline = mean_median_filter(baseline,600,0.6)
        baseline = np.array(baseline)
        clean_sig = pli - baseline
        np.save(name,clean_sig)
        return clean_sig

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff=100.0, fs=360.0, order=5,btype='low'):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

