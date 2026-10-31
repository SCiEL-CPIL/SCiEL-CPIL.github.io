import numpy as np
import time

N = 600
# Interval [-1/4, 1/4]
x = np.linspace(-0.25, 0.25, N)
# dx is implicit in the convolution if we just use the array values, 
# but for "normalized" metrics like ratio, dx cancels out.

def compute_sidelobe_ratio(f):
    """
    Computes the ratio of the maximum sidelobe to the peak of the autoconvolution.
    Assumes f is non-negative.
    """
    # Use FFT for linear convolution
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    C = np.fft.irfft(F * F, n=n_fft)
    
    # The autoconvolution of a real sequence is symmetric.
    # The peak should be at the center.
    # For linear convolution of two length N sequences, the result length is 2N-1.
    # np.fft.irfft with n=2N gives a circular convolution result padded to 2N.
    # The linear convolution result is in C[0:N-1] and C[N:2N-1] (wrapped).
    # Actually, it's easier to just take the max of the whole C, 
    # but we need to exclude the peak.
    
    peak_val = np.max(C)
    peak_idx = np.argmax(C)
    
    # Create a mask to exclude the peak index and its immediate neighbors if needed?
    # No, just exclude the peak index.
    C_masked = C.copy()
    C_masked[peak_idx] = 0.0
    
    max_sidelobe = np.max(C_masked)
    
    if peak_val == 0:
        return 0.0
        
    return max_sidelobe / peak_val

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin for 150s limit
    
    best_score = float('inf')
    best_f = np.ones(N)  # Fallback to constant
    
    # Candidate 1: Constant
    f_const = np.ones(N)
    score_const = compute_sidelobe_ratio(f_const)
    best_score = score_const
    best_f = f_const
    
    # Candidate 2: Gaussians
    for alpha in np.linspace(1, 50, 50):
        if time.time() - start_time > timeout:
            break
        f_gauss = np.exp(-alpha * x**2)
        # Normalize to sum=1? No, ratio is scale invariant.
        score = compute_sidelobe_ratio(f_gauss)
        if score < best_score:
            best_score = score
            best_f = f_gauss
    
    # Candidate 3: Cosine (Hann-like)
    f_hann = 0.5 * (1 + np.cos(2 * np.pi * (x + 0.25))) # Shifted to be symmetric around 0?
    # x is [-0.25, 0.25]. Center is 0.
    # Hann window on [0, 1] is 0.5(1+cos(2pi t)).
    # Map x to t in [0, 1]: t = (x + 0.25) / 0.5 = 2x + 0.5.
    t = 2 * x + 0.5
    f_hann = 0.5 * (1 + np.cos(2 * np.pi * t))
    # Ensure non-negative (it is)
    score = compute_sidelobe_ratio(f_hann)
    if score < best_score:
        best_score = score
        best_f = f_hann

    # Candidate 4: Super-Gaussians
    for beta in np.linspace(2, 20, 20):
        if time.time() - start_time > timeout:
            break
        f_sup = np.exp(-np.abs(x)**beta)
        score = compute_sidelobe_ratio(f_sup)
        if score < best_score:
            best_score = score
            best_f = f_sup
            
    # Candidate 5: Polynomial window (Raised Cosine generalization)
    for k in np.linspace(1, 10, 10):
        if time.time() - start_time > timeout:
            break
        # (1 - (x/0.25)^k)^2 ?
        # Let's use (1 - (2x)^k)^2 for k>0
        term = np.abs(2 * x)**k
        f_poly = np.power(1 - term, 2)
        f_poly[f_poly < 0] = 0
        if np.sum(f_poly) == 0:
            continue
        score = compute_sidelobe_ratio(f_poly)
        if score < best_score:
            best_score = score
            best_f = f_poly

    # Return the best found function
    # The interface expects a sequence of nonnegative step heights
    return list(best_f)
