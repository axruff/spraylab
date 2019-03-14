import numpy as np
import math

from numba import vectorize
from numba import jit


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    
    #return y
    return y[(int(window_len/2)-1):-(int(window_len/2))]

def count_zero(a):  
    eps = 1e-5
    #vals = a[a < eps]
    vals = np.where(a < min_vel_in_pixels, 1, 0)
    
    if len(vals) == 0:
        return 0.0
    else:
        return np.sum(vals)
   
def max_outliers(a):  
    eps = 1e-5
    #vals = a[a < eps]
    vals = np.where(a > max_vel_in_pixels, a, 0)
    
    if len(vals) == 0:
        return 0.0
    else:
        return np.max(vals)

def high_corr(a):  
    eps = 1e-5
    #vals = a[a < eps]
    vals = np.where(a > min_corr_value, a, 0)
    
    if len(vals) == 0:
        return 0.0
    else:
        return np.mean(vals)
    
def low_corr(a):  
    eps = 1e-5
    #vals = a[a < eps]
    vals = np.where(a < min_corr_value, a, 0)
    
    if len(vals) == 0:
        return 0.0
    else:
        return np.mean(vals)
    
    
def avg_nonzero(a): 
    
    non_zero_vals = a[a > 0]
    
    if len(non_zero_vals) == 0:
        return 0.0
    else:
        return np.mean(non_zero_vals)
    
def std_nonzero(a): 
    
    non_zero_vals = a[a != 0]
    
    if len(non_zero_vals) == 0:
        return 0.0
    else:
        return np.std(non_zero_vals)
    
@jit
def avg_non_zero_numba(a):
    num = 0
    sum_val = 0
    
    for i in range(len(a)):
        if a[i] > 0:
            sum_val += a[i]
            num += 1
    if num == 0:
        return 0
        
    return sum_val / num

@jit
def avg_non_zero_full(a):
    
    nz = a.shape[0]
    ny = a.shape[1]
    nx = a.shape[2]
    
    res = np.zeros_like(a[0])
    
    for i in range(ny):
        for j in range(nx):
             
            num = 0
            sum_val = 0
            
            for k in range(nz):

                val = a[k,i,j]
                
                if val != 0:
                    sum_val += val                    
                    num += 1
                    
            if num == 0:
                res[i,j] = 0
            else:
                res[i,j] = sum_val / num 
                
    return res
  
def mul3(a, b, c):
    return a*b*c

def mul4(a, b, c, d):
    return a*b*c*d

vec_mul3 = vectorize('float64(float64, float64, float64)', target='parallel')(mul3)
vec_mul4 = vectorize('float64(float64, float64, float64, float64)', target='parallel')(mul4)