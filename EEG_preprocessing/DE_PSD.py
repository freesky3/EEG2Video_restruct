'''
compute the differential entropy(DE) and power spectral density(PSD) of EEG signal

'''


# define stardard frequency bands for calculating PSD
freq_band = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 99)]
'''
1-4Hz corresponds to delta band
4-8Hz corresponds to theta band
8-14Hz corresponds to alpha band
14-31Hz corresponds to beta band
31-99Hz corresponds to gamma band
'''

import numpy as np

def DE_PSD(data, time_win, fre=200, STFTN = 200, freq_band = freq_band): 
    '''
    compute the differential entropy(DE) and power spectral density(PSD) of EEG signal
    
    args:
    data: EEG signal, (num_electrodes=62, num_samples)
    fre: sampling frequency
    time_win: time window for calculating PSD
    STFTN: number of points for Short-Time Fourier, which will influence the frequency resolution
    freq_band: frequency bands for calculating PSD

    
    returns:
    DE: differential entropy, (num_electrodes=62, len(freq_band)=5)
    PSD: power spectral density, (num_electrodes=62, len(freq_band)=5)
    '''

    # turn freq_band into np array
    fStart, fEnd = list(zip(*freq_band))
    fStart, fEnd = np.array(fStart), np.array(fEnd)
    fStartNum = (fStart/fre*STFTN).astype(int)
    fEndNum = (fEnd/fre*STFTN).astype(int)

    Hlength = int(time_win*fre)
    Hwindow = np.hanning(Hlength)

    num_electrodes = data.shape[0]
    psd = np.zeros((num_electrodes, len(freq_band)))
    de = np.zeros((num_electrodes, len(freq_band)))

    for i in range(num_electrodes):
        data_i = data[i,:]
        Hdata = data_i*Hwindow
        FFTdata = np.fft.fft(Hdata, n=STFTN)
        # input of fft is real signal, so only need to calculate half of the spectrum
        magFFTdata = abs(FFTdata[0:int(STFTN/2)])

        for p in range(0,len(freq_band)):
            E = 0
            for p0 in range(fStartNum[p],fEndNum[p]+1):
                E=E+magFFTdata[p0]*magFFTdata[p0]
            E = E/(fEndNum[p]-fStartNum[p]+1)
            psd[i][p] = E
            de[i][p] = np.log2(100*E)
        
    return de, psd
    