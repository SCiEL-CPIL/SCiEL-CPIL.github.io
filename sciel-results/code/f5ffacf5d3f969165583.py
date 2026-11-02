"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600
# Interval width on [-1/4, 1/4]
h = (1/4 - (-1/4)) / N  # 1/2 / 600 = 1/1200

# EVOLVE-BLOCK-START
def _autoconvolution_max(f, h):
    """Compute the maximum of the autoconvolution (f * f) at the grid points."""
    c = np.convolve(f, f, mode='full')
    return np.max(c) * h

def _score(f):
    """Compute the normalized maximum of the autoconvolution."""
    s = np.sum(f) * h
    if s == 0:
        return np.inf
    max_conv = _autoconvolution_max(f, h)
    return max_conv / (s * s)

def _smooth_max_surrogate(c, beta):
    """Log-sum-exp approximation of the max."""
    c_max = np.max(c)
    return c_max + np.log(np.sum(np.exp(beta * (c - c_max)))) / beta

def _objective(z, beta):
    """Objective function: smooth max of autoconvolution of softmax(z)."""
    z_max = np.max(z)
    f = np.exp(z - z_max)
    f_sum = np.sum(f)
    f = f / f_sum
    
    c = np.convolve(f, f, mode='full')
    
    return _smooth_max_surrogate(c, beta)

def _objective_grad(z, beta):
    """Gradient of the smooth max objective w.r.t z."""
    z_max = np.max(z)
    f = np.exp(z - z_max)
    f_sum = np.sum(f)
    f = f / f_sum
    
    c = np.convolve(f, f, mode='full')
    
    c_max = np.max(c)
    exp_bc = np.exp(beta * (c - c_max))
    sum_exp = np.sum(exp_bc)
    p = exp_bc / sum_exp
    
    # Compute G_f = 2 * correlate(p, f, 'full')[N-1 : N-1+N]
    # Using FFT for speed
    L = 1 << (2*N).bit_length()
    F_p = np.fft.rfft(p, L)
    F_f = np.fft.rfft(f, L)
    # Correlation is IFFT(FFT(p) * conj(FFT(f)))
    F_corr = F_p * np.conj(F_f)
    corr = np.fft.irfft(F_corr, L)
    
    # Extract the central part
    # The full correlation length is 2N-1.
    # The lag 0 is at index N-1 in the 'full' output of np.correlate?
    # Let's verify.
    # np.correlate([1,2], [3,4], 'full') -> [4, 10, 6]
    # Lag -1: 1*4=4. Lag 0: 1*3+2*4=11? No.
    # Definition: sum a[n+k] b[n].
    # k=-1: a[0]b[1] = 4.
    # k=0: a[0]b[0] + a[1]b[1] = 3+8=11.
    # k=1: a[1]b[0] = 6.
    # So lags are -1, 0, 1.
    # Indices in 'full' are 0, 1, 2.
    # Lag 0 is at index len(b)-1 = 1.
    # We want lags 0 to N-1.
    # So we take indices N-1 to N-1+N-1.
    
    G_f = 2 * corr[N-1 : N-1+N]
    
    grad_z = f * G_f
    grad_z -= f * np.dot(f, G_f)
    
    return grad_z

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    np.random.seed(42)
    
    z = np.zeros(N)
    lr = 0.1
    beta = 50.0
    
    best_score = _score(np.exp(z) / np.sum(np.exp(z)))
    best_z = z.copy()
    
    max_iter = 1000
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        grad = _objective_grad(z, beta)
        
        z_new = z - lr * grad
        
        f_new = np.exp(z_new) / np.sum(np.exp(z_new))
        score_new = _score(f_new)
        
        if score_new < best_score:
            best_score = score_new
            best_z = z_new.copy()
            z = z_new
            if i % 100 == 0 and beta < 200:
                beta *= 1.5
        else:
            lr *= 0.9
            if score_new < best_score * 1.01:
                z = z_new
            else:
                z = z

    f_final = np.exp(best_z) / np.sum(np.exp(best_z))
    return f_final.tolist()
# EVOLVE-BLOCK-END
