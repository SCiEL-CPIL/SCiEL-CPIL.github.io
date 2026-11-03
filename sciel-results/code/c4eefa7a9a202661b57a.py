import numpy as np
import time

N = 600
# Domain [-0.25, 0.25]
x = np.linspace(-0.25, 0.25, N)

def compute_psl(f):
    # Ensure non-negative
    f = np.abs(f)
    if np.sum(f) == 0:
        return float('inf')
    # Zero pad for linear convolution
    n_fft = 2 * N
    F = np.fft.fft(f, n=n_fft)
    R = np.fft.ifft(np.abs(F)**2).real
    # R is linear autoconvolution
    # Peak is at index 0 (or N if shifted, but ifft puts 0 at 0)
    # Wait, ifft of |F|^2 gives circular convolution.
    # With zero padding to 2N, the linear convolution is in R[0:N] and R[N:2N] (wrapped).
    # Actually, standard practice:
    # R_lin = np.fft.ifft(np.abs(np.fft.fft(f, n_fft))**2).real
    # The peak is at index 0.
    peak = R[0]
    if peak == 0:
        return float('inf')
    # Exclude the main peak.
    # How many lags to exclude?
    # If we exclude only k=0, then k=1 is included.
    # For a smooth function, R[1] is close to R[0].
    # "Sidelobes" usually refer to lobes outside the main lobe.
    # However, "normalized maximum of its autoconvolution" might just mean max(R) / R[0].
    # If it's max(R)/R[0], it's always 1.
    # So it must be max(R[1:]) / R[0] or similar.
    # Let's assume max(R[1:]) / R[0].
    
    # There is a symmetry. R[k] = R[-k] = R[N-k].
    # We just need to check the first half.
    # Exclude k=0.
    max_side = np.max(R[1:N]) # Assuming the rest is small or wrapped
    # Actually, for linear conv with zero padding, R[0:N] is the first half, R[N:2N] is the second half (negative lags).
    # So R[1:N] covers lags 1 to N-1.
    return max_side / peak

def construct():
    start_time = time.time()
    deadline = start_time + 100 # Safety margin
    
    best_f = None
    best_score = float('inf')
    
    # Candidate 1: Hamming
    # Hamming window formula: 0.54 - 0.46 * cos(2*pi*i/(N-1))
    i = np.arange(N)
    f_hamming = 0.54 - 0.46 * np.cos(2 * np.pi * i / (N - 1))
    # Ensure non-negative (it is)
    score_h = compute_psl(f_hamming)
    best_f = f_hamming
    best_score = score_h
    
    # Candidate 2: Hanning
    f_hanning = 0.5 * (1 - np.cos(2 * np.pi * i / (N - 1)))
    score_ha = compute_psl(f_hanning)
    if score_ha < best_score:
        best_f = f_hanning
        best_score = score_ha
        
    # Candidate 3: Blackman
    f_blackman = 0.42 - 0.5 * np.cos(2 * np.pi * i / (N - 1)) + 0.08 * np.cos(4 * np.pi * i / (N - 1))
    score_b = compute_psl(f_blackman)
    if score_b < best_score:
        best_f = f_blackman
        best_score = score_b
        
    # Candidate 4: Optimized Gaussian
    # f = exp(-x^2 / (2*sigma^2))
    # Search sigma
    sigmas = np.logspace(-3, -1, 50)
    for sigma in sigmas:
        if time.time() > deadline:
            break
        f_gauss = np.exp(-x**2 / (2 * sigma**2))
        score_g = compute_psl(f_gauss)
        if score_g < best_score:
            best_f = f_gauss
            best_score = score_g
            
    # Return list of floats
    return list(best_f)
