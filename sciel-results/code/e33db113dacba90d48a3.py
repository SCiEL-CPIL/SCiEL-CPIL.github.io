"""
Strategy: SPECTRAL-DOMAIN PARAMETRIC OPTIMIZATION + SPECTRAL REFINEMENT

The current approach operates in the time domain using structured basis functions 
and coordinate descent. This new strategy reframes the problem in the frequency 
domain. 

Key Insight:
The objective is to minimize max_k (f * f)[k] / (sum f)^2.
By the Wiener-Khinchin theorem, the autoconvolution is the inverse FFT of the 
power spectrum |F|^2. To minimize the peak of the inverse FFT (the time-domain 
autocorrelation), we want to make the power spectrum as flat as possible. 

A flat power spectrum corresponds to a "white noise" like time-domain signal, 
which has a low peak factor. By parameterizing the magnitude of the Fourier 
coefficients and optimizing them to be uniform (while maintaining non-negativity 
in the time domain via projection), we can achieve a much flatter autocorrelation 
peak than time-domain basis functions.

Method:
1. Parameterize the signal by its Fourier magnitude vector M (length N/2+1) 
   and phase vector P.
2. Initialize M to be roughly uniform (flat spectrum) and P to be random.
3. Use Nelder-Mead to optimize the parameters. The objective is the normalized 
   peak autocorrelation. To enforce non-negativity, we project the time-domain 
   signal to be non-negative (clip at 0) before scoring.
4. Perform a fast spectral-domain refinement: for each frequency bin, adjust the 
   magnitude to reduce the peak autocorrelation, using the fact that the 
   contribution of each bin to the time-domain peak is known.

This is a structural change from "time-domain basis + coordinate descent" to 
"frequency-domain parametric + spectral refinement".
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

def compute_score_fft(f):
    """Compute the normalized maximum of the autoconvolution using FFT for speed."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)[:2 * N - 1]
    return np.max(conv) / (s * s)

def compute_score_exact(f):
    """Compute the normalized maximum of the autoconvolution exactly."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def spectral_to_time(params, n_bins):
    """
    Convert spectral parameters (magnitudes and phases) to a nonnegative time-domain signal.
    params: array of length 2 * n_bins
    First n_bins are magnitudes, next n_bins are phases.
    """
    mags = np.maximum(params[:n_bins], 0.0)
    phases = params[n_bins:]
    
    # Construct complex spectrum
    F = mags * np.exp(1j * phases)
    
    # Inverse FFT to get time domain
    f = np.fft.irfft(F, n=N)
    
    # Enforce non-negativity
    f = np.maximum(f, 0.0)
    
    return f

def objective_spectral(params, n_bins):
    """Objective function for spectral optimization."""
    f = spectral_to_time(params, n_bins)
    s = np.sum(f)
    if s < 1e-12:
        return 1e6
    return compute_score_fft(f)

def spectral_refinement(f, max_iter=10, time_limit=10):
    """
    Refine the signal in the frequency domain by adjusting magnitudes to flatten the peak.
    """
    start_time = time.time()
    current_f = f.copy()
    current_score = compute_score_fft(current_f)
    best_f = current_f.copy()
    best_score = current_score
    
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
        
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        F = np.fft.rfft(current_f, n=n_fft)
        mags = np.abs(F)
        phases = np.angle(F)
        n_bins = len(mags)
        
        improved = False
        for k in range(n_bins):
            if time.time() - start_time > time_limit:
                break
            
            # Try scaling the magnitude of bin k
            # A flat spectrum is desired. If bin k is too large relative to others, reduce it.
            # If it's too small, increase it.
            # Simple heuristic: try to make |F[k]| closer to the mean magnitude.
            mean_mag = np.mean(mags)
            target_mag = mean_mag
            
            candidates = [mags[k] * 0.5, mags[k] * 0.8, mags[k], mags[k] * 1.2, mags[k] * 1.5, target_mag]
            best_val = current_score
            best_mag = mags[k]
            
            for new_mag in candidates:
                if new_mag == mags[k]:
                    continue
                F[k] = new_mag * np.exp(1j * phases[k])
                f_new = np.fft.irfft(F, n=N)
                f_new = np.maximum(f_new, 0.0)
                s_new = np.sum(f_new)
                if s_new < 1e-12:
                    continue
                score_new = compute_score_fft(f_new)
                if score_new < best_val:
                    best_val = score_new
                    best_mag = new_mag
            
            if best_mag != mags[k]:
                mags[k] = best_mag
                F[k] = best_mag * np.exp(1j * phases[k])
                improved = True
                current_score = best_val
                if current_score < best_score:
                    best_score = current_score
                    best_f = np.fft.irfft(F, n=N)
                    best_f = np.maximum(best_f, 0.0)
        
        if not improved:
            break
            
    return best_f, best_score

def construct():
    start_time = time.time()
    
    n_bins = N // 2 + 1
    
    # Initialize magnitudes to be uniform (flat spectrum)
    # and phases to be random.
    # We use a softplus-like initialization to ensure smoothness.
    mags_init = np.ones(n_bins)
    phases_init = np.random.uniform(-np.pi, np.pi, n_bins)
    
    init_params = np.concatenate([mags_init, phases_init])
    
    best_f = spectral_to_time(init_params, n_bins)
    best_score = compute_score_fft(best_f)
    
    # Nelder-Mead optimization on spectral parameters
    # This is a high-dimensional problem (601 params), but Nelder-Mead can handle it 
    # if we are careful. However, it might be slow. 
    # Let's use a smaller subset of parameters or a different optimizer.
    # Actually, let's just try Nelder-Mead with a limited number of iterations.
    
    def obj_wrapper(params):
        return objective_spectral(params, n_bins)
    
    try:
        res = minimize(obj_wrapper, init_params, method='Nelder-Mead',
                       options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-5, 'disp': False})
        if res.fun < best_score:
            best_score = res.fun
            best_f = spectral_to_time(res.x, n_bins)
    except Exception as e:
        pass
    
    # Check time
    if time.time() - start_time > TIME_LIMIT - 30:
        return [float(x) for x in best_f]
    
    # Try a few different initializations
    # 1. White noise in time domain (flat spectrum in frequency)
    f_white = np.random.rand(N)
    best_f_white = f_white
    best_score_white = compute_score_fft(f_white)
    if best_score_white < best_score:
        best_f = best_f_white
        best_score = best_score_white
        
    # 2. Box function in time domain
    f_box = np.zeros(N)
    f_box[N//4:3*N//4] = 1.0
    best_score_box = compute_score_fft(f_box)
    if best_score_box < best_score:
        best_f = f_box
        best_score = best_score_box
        
    # Spectral refinement on the best f
    if time.time() - start_time < TIME_LIMIT - 10:
        refined_f, refined_score = spectral_refinement(best_f, max_iter=20, time_limit=TIME_LIMIT - time.time() - 5)
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Final exact check
    final_score = compute_score_exact(best_f)
    if final_score < best_score:
        best_score = final_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    start = time.time()
    f = construct()
    elapsed = time.time() - start
    score = compute_score_exact(f)
    print(f"Score: {score}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
    print(f"Time: {elapsed:.2f}s")
