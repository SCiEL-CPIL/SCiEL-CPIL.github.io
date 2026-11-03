import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_score(x):
    """
    Compute the normalized maximum of the autoconvolution of a step function
    with heights x on N equal intervals over [-1/4, 1/4].
    
    The score is max(c) / (sum(x)^2 * h) where c is the discrete autoconvolution
    and h = 1/(2N). For constant x, this gives 2.0.
    """
    if np.allclose(x, 0):
        return 0.0
    c = np.convolve(x, x, mode='full')
    max_c = np.max(c)
    sum_x = np.sum(x)
    if sum_x == 0:
        return 0.0
    h = 1.0 / (2 * N)
    score = max_c / (sum_x * sum_x * h)
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # The problem is to minimize the peak of the autoconvolution.
    # By Parseval's theorem, the peak of the autoconvolution is related to the
    # L2 norm of the Fourier transform squared.
    # 
    # Key insight: We want to minimize ||f*f||_∞ / ||f||_1^2.
    # This is equivalent to making the Fourier transform as flat as possible.
    # 
    # Strategy: Use iterative reweighting in the frequency domain.
    # 1. Start with uniform initialization.
    # 2. Compute FFT.
    # 3. Apply a "flattening" operation to the magnitude spectrum.
    # 4. Project back to time domain with non-negativity constraint.
    # 5. Repeat until convergence.
    
    best_x = np.ones(N)
    best_score = compute_score(best_x)
    
    # Frequency domain optimization using iterative reweighting
    # The idea: minimize max|F(f)|^2 where F is the DFT of x
    # Subject to x >= 0 and sum(x) = const
    
    x = np.ones(N)
    
    # Precompute FFT size for efficiency
    fft_size = 2 * N  # Use zero-padded FFT for linear convolution
    
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        # Compute FFT of current x
        X = np.fft.rfft(x, n=fft_size)
        
        # The autoconvolution peak is related to the maximum of |X|^2
        # We want to flatten |X|^2
        mag = np.abs(X)
        
        # Apply soft thresholding / flattening in frequency domain
        # Reduce high-frequency components that contribute to peaks
        # Use a power-law flattening: new_mag = mag^alpha where alpha < 1
        # This reduces the dynamic range of the spectrum
        
        # Adaptive alpha based on current score
        current_score = compute_score(x)
        
        # Try different flattening strengths
        for alpha in [0.8, 0.9, 0.95]:
            if time.time() > deadline - 5:
                break
            
            # Flatten the magnitude spectrum
            mag_flat = mag ** alpha
            
            # Preserve the DC component (total mass)
            mag_flat[0] = mag[0]
            
            # Reconstruct with flattened magnitude and original phase
            X_new = mag_flat * (X / (mag + 1e-12))
            
            # Transform back to time domain
            x_candidate = np.fft.irfft(X_new, n=fft_size)[:N]
            
            # Project to non-negative
            x_candidate = np.maximum(x_candidate, 0)
            
            # Normalize to preserve L1 norm
            if np.sum(x_candidate) > 0:
                x_candidate = x_candidate / np.sum(x_candidate) * np.sum(x)
            
            # Evaluate
            score_candidate = compute_score(x_candidate)
            
            if score_candidate < current_score:
                x = x_candidate
                current_score = score_candidate
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
        
        # Also try a time-domain smoothing step
        # Apply a Gaussian-like smoothing to reduce local peaks
        if iteration % 10 == 0:
            # Simple boxcar smoothing
            kernel = np.ones(3) / 3
            x_smooth = np.convolve(x, kernel, mode='same')
            x_smooth = np.maximum(x_smooth, 0)
            if np.sum(x_smooth) > 0:
                x_smooth = x_smooth / np.sum(x_smooth) * np.sum(x)
            score_smooth = compute_score(x_smooth)
            if score_smooth < current_score:
                x = x_smooth
                current_score = score_smooth
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
    
    # Final local refinement using coordinate descent
    # Try adjusting individual elements to reduce the peak
    x = best_x.copy()
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        
        improved = False
        c = np.convolve(x, x, mode='full')
        max_idx = np.argmax(c)
        
        # The peak is at lag max_idx - (N-1)
        lag = max_idx - (N + 1)  # Adjust for 0-indexing
        
        # Try reducing elements that contribute to the peak
        # and increasing elements that don't
        for i in range(0, N, 20):  # Sample some indices
            if time.time() > deadline - 2:
                break
            
            # Try moving mass from i to i+1
            if i < N - 1 and x[i] > 0:
                x_trial = x.copy()
                delta = x[i] * 0.1
                x_trial[i] -= delta
                x_trial[i+1] += delta
                x_trial = np.maximum(x_trial, 0)
                
                score_trial = compute_score(x_trial)
                if score_trial < compute_score(x):
                    x = x_trial
                    improved = True
        
        if not improved:
            break
    
    final_score = compute_score(x)
    if final_score < best_score:
        best_score = final_score
        best_x = x.copy()
    
    return list(best_x)
