"""
Multi-resolution PSD-domain optimization for minimizing max(|autoconv|)/(sum x)^2.
Uses coarse-to-fine spectral interpolation to avoid local minima.
"""
import numpy as np
import time
from scipy.optimize import minimize

N_FINAL = 400

START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    N = len(x)
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _psd_objective_and_grad(psd, beta=100.0):
    """
    Smooth objective and gradient for PSD optimization.
    Minimizes smooth_max(|ifft(psd)|) / psd[0]
    """
    L_fft = len(psd) * 2 - 1
    # psd is length L_fft//2 + 1
    # Reconstruct full length for IFFT
    # Actually, irfft takes length L_fft//2+1 and outputs L_fft
    
    # Compute autoconvolution from PSD
    # Note: psd here is P[k] = |X[k]|^2
    # C[k] = IFFT(P)
    ac = np.fft.irfft(psd, n=L_fft)
    abs_ac = np.abs(ac)
    
    dc = psd[0]
    if dc < 1e-15:
        return 1e10, np.zeros_like(psd)
    
    # Smooth max via log-sum-exp
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        smooth_max = max_val
        weights = np.ones_like(abs_ac) / len(abs_ac)
    else:
        shifted = abs_ac - max_val
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        smooth_max = (max_val + np.log(sum_exp)) / beta
        weights = exp_shifted / sum_exp
    
    val = smooth_max / dc
    
    # Gradient computation
    # d(smooth_max)/dpsd[k]
    # c = weights * sign(ac)
    c = weights * np.sign(ac)
    
    # grad_smooth_max_psd = FFT(c).conj() / L_fft ?
    # C = IFFT(P). dC/dP = IFFT basis.
    # d(smooth_max)/dP = sum_k (d(smooth_max)/dC_k) * (dC_k/dP)
    # dC_k/dP_j = delta_{k,j} (in frequency domain sense? No)
    # C is in time domain (lags), P is in freq domain.
    # C[l] = (1/L) sum_m P[m] exp(2*pi*i*l*m/L)
    # dC[l]/dP[m] = (1/L) exp(2*pi*i*l*m/L)
    # d(smooth_max)/dP[m] = sum_l (d(smooth_max)/dC[l]) * (1/L) exp(2*pi*i*l*m/L)
    # This is (1/L) * FFT( d(smooth_max)/dC )[m]
    
    # d(smooth_max)/dC[l] = weights[l] * sign(C[l]) = c[l]
    
    fft_c = np.fft.fft(c, n=L_fft)
    grad_smooth_max_psd = np.real(fft_c.conj()) / L_fft
    
    # Gradient of val = smooth_max / dc
    # dc = psd[0]
    grad = grad_smooth_max_psd / dc
    # d/dc (smooth_max / dc) = -smooth_max / dc^2
    # d/dpsd[0] (smooth_max / psd[0]) = (grad_sm[0] * psd[0] - smooth_max) / psd[0]^2
    # d/dpsd[k>0] = grad_sm[k] / psd[0]
    
    grad[0] = (grad_smooth_max_psd[0] * dc - smooth_max) / (dc * dc)
    if len(grad) > 1:
        grad[1:] = grad_smooth_max_psd[1:] / dc
    
    return val, grad

def _psd_to_signal(psd):
    """Reconstruct real time-domain signal from PSD with zero phase."""
    L_fft = len(psd) * 2 - 1
    N = L_fft // 2
    # X[k] = sqrt(PSD[k])
    X = np.zeros(L_fft, dtype=complex)
    X[:len(psd)] = np.sqrt(np.maximum(psd, 0))
    for k in range(1, len(psd)):
        X[L_fft-k] = X[k]
    
    x = np.fft.ifft(X).real[:N]
    return x

def _upsample_psd(x_small, N_large):
    """Upsample signal by interpolating PSD."""
    L_small = 2 * len(x_small) - 1
    L_fft_small = 1
    while L_fft_small < L_small:
        L_fft_small *= 2
        
    X_small = np.fft.rfft(x_small, n=L_fft_small)
    psd_small = np.abs(X_small)**2
    
    L_large = 2 * N_large - 1
    L_fft_large = 1
    while L_fft_large < L_large:
        L_fft_large *= 2
        
    n_psd_small = L_fft_small // 2 + 1
    n_psd_large = L_fft_large // 2 + 1
    
    # Interpolate PSD
    freqs_small = np.fft.rfftfreq(L_fft_small)
    freqs_large = np.fft.rfftfreq(L_fft_large)
    
    # Ensure monotonic increasing frequencies for interp
    psd_large = np.interp(freqs_large, freqs_small, psd_small)
    
    # Reconstruct signal
    X_large = np.zeros(L_fft_large, dtype=complex)
    X_large[:n_psd_large] = np.sqrt(np.maximum(psd_large, 0))
    for k in range(1, n_psd_large):
        X_large[L_fft_large-k] = X_large[k]
        
    x_large = np.fft.ifft(X_large).real[:N_large]
    return x_large

def _optimize_psd(x_init, N, beta=100.0, maxiter=200):
    """Optimize PSD for given N."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_psd = L_fft // 2 + 1
    
    # Initialize PSD from x_init
    X_init = np.fft.rfft(x_init, n=L_fft)
    psd_init = np.abs(X_init)**2
    
    # Ensure psd[0] is not zero
    if psd_init[0] < 1e-10:
        psd_init[0] = 1.0
        
    def _obj_grad(p):
        return _psd_objective_and_grad(p, beta=beta)
    
    try:
        result = minimize(_obj_grad, psd_init, method='L-BFGS-B',
                         jac=True, options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-12})
        
        psd_opt = np.maximum(result.x, 0)
        x_opt = _psd_to_signal(psd_opt)
        
        # Normalize to sum 1 (optional, but helps stability)
        s = np.sum(x_opt)
        if abs(s) > 1e-10:
            x_opt = x_opt / s
            
        return x_opt
    except Exception:
        return x_init

def construct():
    # Multi-resolution stages
    stages = [50, 100, 200, 400]
    
    x_current = np.ones(50)
    
    for N_stage in stages:
        if _check_deadline():
            break
            
        if N_stage > 50:
            x_current = _upsample_psd(x_current, N_stage)
            
        # Optimize at this resolution
        # Use a few betas to escape local minima
        best_x = x_current
        best_val = _objective(x_current)
        
        for beta in [50.0, 100.0, 200.0]:
            if _check_deadline():
                break
                
            x_opt = _optimize_psd(x_current, N_stage, beta=beta, maxiter=100)
            val_opt = _objective(x_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_x = x_opt
                x_current = x_opt.copy()
                
    return best_x
