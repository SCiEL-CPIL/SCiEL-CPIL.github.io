"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Discrete spectral search.
Parameterize the step function by its DFT coefficients.
Enforce non-negativity via a soft-plus projection.
Directly minimize the maximum value of the inverse DFT of the squared magnitude spectrum
using coordinate descent on the frequency domain.
This decouples the peak constraint from the time-domain shape search.

Key insight: The autoconvolution in the time domain corresponds to the squared
magnitude of the DFT in the frequency domain (up to scaling). By working in the
frequency domain, we can directly control the "energy distribution" of the
autoconvolution. The maximum of the autoconvolution is related to the peak of
the inverse DFT of |F|^2. We use coordinate descent on the frequency coefficients
to flatten this peak, while projecting back to the time domain to enforce
non-negativity.
"""

import numpy as np
import time
from scipy.optimize import minimize, Bounds

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def soft_plus(x, beta=1.0):
    """Soft approximation to max(x, 0) using log(1 + exp(beta*x))/beta."""
    x = np.asarray(x, dtype=np.float64)
    # Numerically stable softplus
    return np.logaddexp(0, beta * x) / beta

def dft_to_time(F):
    """Convert DFT coefficients to time-domain step function with non-negativity."""
    # F is the DFT of the time-domain signal
    # We want a real, non-negative time-domain signal
    # Take inverse DFT
    f = np.real(np.fft.ifft(F))
    # Project to non-negative using soft-plus (differentiable)
    # But for the actual score, we need hard non-negativity
    # For optimization, we use soft-plus to guide the search
    f_soft = soft_plus(f, beta=5.0)
    # For the actual evaluation, we use hard projection
    f_hard = np.maximum(f, 0.0)
    return f_hard

def time_to_dft(f):
    """Convert time-domain step function to DFT coefficients."""
    return np.fft.fft(f)

def objective_from_F(F):
    """Compute the score from DFT coefficients.
    
    The autoconvolution in time domain is the inverse DFT of |F|^2 (scaled).
    We want to minimize the maximum of this autoconvolution, normalized by (sum f)^2.
    """
    # |F|^2 is the power spectrum
    power = np.abs(F) ** 2
    # Inverse DFT of power gives the autoconvolution (up to a scaling factor)
    # For a signal f of length N, the full autoconvolution has length 2N-1
    # The DFT of the circular autoconvolution is |F|^2
    # The linear autoconvolution is slightly different, but for large N they are similar
    # Let's compute the circular autoconvolution for speed
    conv_circ = np.real(np.fft.ifft(power))
    # The circular autoconvolution is the same as linear for the middle part
    # For the peak, we look at the circular version
    s = np.sum(np.real(np.fft.ifft(F)))  # sum of f
    if s < 1e-15:
        return np.inf
    # The peak of the circular autoconvolution is at lag 0 (or N)
    # But we want the maximum over all lags
    max_conv = np.max(conv_circ)
    return max_conv / (s * s)

def coordinate_descent(F, start_time, max_time, n_iter=5000):
    """Coordinate descent on DFT coefficients to minimize the peak of autoconvolution."""
    best_F = F.copy()
    best_score = objective_from_F(F)
    current_F = F.copy()
    
    # We optimize the real and imaginary parts of F
    # To maintain a real time-domain signal, F must satisfy conjugate symmetry
    # F[k] = conj(F[N-k])
    # So we only need to optimize k = 0, 1, ..., N/2
    
    half = N // 2
    step_size = 0.1
    patience = 100
    no_improve = 0
    
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        # Iterate over frequency bins
        for k in range(half + 1):
            # Try small perturbations in real and imaginary parts
            for delta in [-step_size, step_size]:
                # Perturb real part
                F_trial = current_F.copy()
                F_trial[k] += delta
                if k != 0 and k != half:
                    F_trial[N - k] += delta  # conjugate symmetry (real part same)
                # For conjugate symmetry: F[N-k] = conj(F[k])
                # If we change F[k] by delta (real), F[N-k] should change by delta (real)
                # If we change F[k] by i*delta (imag), F[N-k] should change by -i*delta
                
                score_trial = objective_from_F(F_trial)
                if score_trial < best_score:
                    best_score = score_trial
                    best_F = F_trial.copy()
                    current_F = F_trial.copy()
                    no_improve = 0
                    break
                else:
                    no_improve += 1
            
            if no_improve > patience:
                break
        
        if no_improve > patience:
            # Reduce step size
            step_size *= 0.9
            no_improve = 0
            if step_size < 1e-8:
                break
    
    return best_F, best_score

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Initialize DFT coefficients
    # Start with a few different initializations
    initializations = [
        np.fft.fft(np.ones(N)),
        np.fft.fft(np.cos(2 * np.pi * np.arange(N) / N) + 1),
        np.fft.fft(np.sin(2 * np.pi * np.arange(N) / N) + 1.5),
        np.fft.fft(np.exp(-np.abs(np.arange(N) - N/2) / 100.0)),
    ]
    
    for F_init in initializations:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        # Ensure conjugate symmetry for real signal
        F_sym = F_init.copy()
        half = N // 2
        for k in range(1, half + 1):
            F_sym[N - k] = np.conj(F_sym[k])
        
        # Coordinate descent on frequency domain
        F_opt, score_opt = coordinate_descent(F_sym, start_time, TIME_LIMIT - 5, n_iter=3000)
        
        # Convert back to time domain
        f_opt = np.real(np.fft.ifft(F_opt))
        f_opt = np.maximum(f_opt, 0.0)
        
        # Compute actual score
        actual_score = compute_score(f_opt)
        
        if actual_score < best_score:
            best_score = actual_score
            best_f = f_opt
    
    # Additional local refinement in time domain
    if time.time() - start_time < TIME_LIMIT - 10:
        # Simple gradient-free refinement
        h = best_f.copy()
        alpha = 0.02
        for _ in range(200):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            s = np.sum(h)
            if s < 1e-12:
                break
            h_norm = h * (N / s)
            
            conv = np.convolve(h_norm, h_norm, mode='full')
            max_c = np.max(conv)
            current_val = max_c / (N * N)
            
            # Find peak lags
            threshold = max_c * 0.9
            peak_indices = np.where(conv >= threshold)[0]
            
            # Sensitivity
            sensitivity = np.zeros(N)
            for m in peak_indices:
                j_min = max(0, m - (N - 1))
                j_max = min(N - 1, m)
                for j in range(j_min, j_max + 1):
                    sensitivity[j] += h_norm[m - j]
            
            sens_mean = np.mean(sensitivity)
            sens_std = np.std(sensitivity) + 1e-12
            sens_norm = (sensitivity - sens_mean) / sens_std
            
            h_new = h * (1.0 - alpha * sens_norm)
            h_new = np.maximum(h_new, 0.0)
            
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                break
            h_new = h_new * (N / s_new)
            
            conv_new = np.convolve(h_new, h_new, mode='full')
            val_new = np.max(conv_new) / (N * N)
            
            if val_new < current_val:
                h = h_new
            else:
                alpha *= 0.9
        
        h = np.maximum(h, 0.0)
        score_h = compute_score(h)
        if score_h < best_score:
            best_score = score_h
            best_f = h
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
