import time
import numpy as np
from scipy.fft import fft, ifft

N = 600
DEADLINE_FRACTION = 0.85

def _autoconv_max(h):
    """Compute max of autoconvolution of h using FFT."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    H = fft(h, n=size)
    autoconv = ifft(H * H, n=size).real
    return np.max(autoconv)

def _score(h):
    """Compute the normalized maximum of autoconvolution."""
    s = np.sum(h)
    if s == 0:
        return float('inf')
    h_norm = h / s
    max_ac = _autoconv_max(h_norm)
    return max_ac * N

def construct():
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRACTION
    
    # Strategy: Frequency-domain spectral shaping
    # The autoconvolution in frequency domain is |H|^2, so the peak of 
    # autoconvolution is related to the L2 norm of |H|^2 in time domain.
    # We want to minimize max(autoconv) subject to h >= 0 and sum(h) = 1.
    # 
    # Key insight: The uniform distribution has flat spectrum, but we can
    # shape the spectrum to reduce the peak of |H|^2 in time domain.
    # 
    # Approach: Start with uniform, then iterate:
    # 1. Compute current spectrum H
    # 2. Shape |H|^2 to be more uniform (reduce peaks)
    # 3. Project back to time domain ensuring non-negativity and normalization
    # 
    # This is a different design family: we optimize in frequency domain
    # with projection constraints, rather than coordinate descent in time domain.
    
    # Initialize with uniform distribution
    h = np.ones(N) / N
    best_h = h.copy()
    best_score = _score(h)
    
    # Frequency domain optimization
    size = 1
    while size < 2 * N - 1:
        size *= 2
    
    # Number of iterations based on time budget
    max_iter = 5000
    lr = 0.01
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute spectrum
        H = fft(h, n=size)
        
        # The autoconvolution is ifft(|H|^2)
        # We want to minimize the peak of this
        # Gradient of max(autoconv) w.r.t. h is complex, so we use a heuristic:
        # Shape |H|^2 to be more uniform
        
        ac_mag = np.abs(H) ** 2
        
        # Target: make ac_mag more uniform (reduce dynamic range)
        # Use a soft thresholding approach
        mean_ac = np.mean(ac_mag)
        std_ac = np.std(ac_mag)
        
        if std_ac < 1e-10:
            break
        
        # Shape: compress the dynamic range
        # New target magnitude: sqrt(mean + beta * (ac_mag - mean))
        beta = 0.5
        target_ac_mag = mean_ac + beta * (ac_mag - mean_ac)
        target_ac_mag = np.maximum(target_ac_mag, 0)
        
        # New magnitude spectrum
        target_mag = np.sqrt(target_ac_mag)
        
        # Keep phase from current H
        phase = np.angle(H)
        
        # Reconstruct
        H_new = target_mag * np.exp(1j * phase)
        
        # Transform back to time domain
        h_new = ifft(H_new, n=size).real[:N]
        
        # Project: ensure non-negativity and normalize
        h_new = np.maximum(h_new, 0)
        s = np.sum(h_new)
        if s == 0:
            break
        h_new = h_new / s
        
        # Compute score
        new_score = _score(h_new)
        
        # Accept if better
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
            h = h_new.copy()
        else:
            # If not better, dampen the update
            h = 0.9 * h + 0.1 * h_new
            h = np.maximum(h, 0)
            s = np.sum(h)
            if s > 0:
                h = h / s
    
    # Try a few random restarts with different initial phases if time allows
    if time.time() < deadline - 10:
        for restart in range(3):
            if time.time() > deadline:
                break
            
            # Random initial spectrum with flat magnitude
            h = np.ones(N) / N
            H = fft(h, n=size)
            
            # Add random phase perturbation
            random_phase = np.random.uniform(-np.pi, np.pi, size)
            H = np.abs(H) * np.exp(1j * random_phase)
            
            h = ifft(H, n=size).real[:N]
            h = np.maximum(h, 0)
            s = np.sum(h)
            if s == 0:
                continue
            h = h / s
            
            current_score = _score(h)
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
            
            # Iterate from this start
            for it in range(1000):
                if time.time() > deadline:
                    break
                
                H = fft(h, n=size)
                ac_mag = np.abs(H) ** 2
                mean_ac = np.mean(ac_mag)
                std_ac = np.std(ac_mag)
                
                if std_ac < 1e-10:
                    break
                
                beta = 0.5
                target_ac_mag = mean_ac + beta * (ac_mag - mean_ac)
                target_ac_mag = np.maximum(target_ac_mag, 0)
                target_mag = np.sqrt(target_ac_mag)
                phase = np.angle(H)
                H_new = target_mag * np.exp(1j * phase)
                h_new = ifft(H_new, n=size).real[:N]
                h_new = np.maximum(h_new, 0)
                s = np.sum(h_new)
                if s == 0:
                    break
                h_new = h_new / s
                
                new_score = _score(h_new)
                if new_score < best_score:
                    best_score = new_score
                    best_h = h_new.copy()
                    h = h_new.copy()
                else:
                    h = 0.9 * h + 0.1 * h_new
                    h = np.maximum(h, 0)
                    s = np.sum(h)
                    if s > 0:
                        h = h / s
    
    return [float(x) for x in best_h]
