"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Spectral design: represent h as 0.5 + x where x is zero-mean.
    Construct x via inverse DFT with controlled phase spectrum to minimize
    peak sidelobe of autocorrelation, then project to [0,1] and adjust sum.
    """
    # We want to minimize max_k C_k where C_k = N/2 - autocorr_h(k)
    # Equivalent to maximizing min_k autocorr_h(k)
    # autocorr_h(k) = N/4 + autocorr_x(k) where x = h - 0.5
    
    # Strategy: use a flat spectrum for x to make autocorr_x(k) ≈ 0 for k≠0
    # This gives autocorr_h(k) ≈ N/4 for all k, so C_k ≈ N/4
    
    # But we can try to make autocorr_x(k) slightly positive for some k
    # by shaping the spectrum
    
    # Construct x via DFT:
    # X[0] = 0 (zero mean)
    # X[k] for k=1..N-1: choose amplitudes and phases
    
    # Start with uniform amplitude, random phases
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = np.inf
    
    # Try multiple random phase assignments
    for trial in range(50):
        # Random phases for DFT components
        phases = np.random.uniform(0, 2*np.pi, N)
        
        # Amplitudes: try different profiles
        # Flat spectrum
        amps = np.ones(N)
        amps[0] = 0
        
        # Construct complex spectrum
        X = amps * np.exp(1j * phases)
        X[0] = 0
        
        # Inverse DFT
        x = np.fft.ifft(X).real
        
        # Scale x to fit in [-0.5, 0.5]
        # We want h = 0.5 + x to be in [0, 1], so x in [-0.5, 0.5]
        max_abs = np.max(np.abs(x))
        if max_abs > 0:
            x = x / max_abs * 0.5
        
        # Adjust to ensure sum(x) = 0
        x = x - np.mean(x)
        
        # Clip to [-0.5, 0.5]
        x = np.clip(x, -0.5, 0.5)
        
        # Re-adjust sum to exactly 0
        x = x - np.mean(x)
        
        # Final clip (may introduce small sum error)
        x = np.clip(x, -0.5, 0.5)
        
        # Correct sum exactly
        sum_error = np.sum(x)
        if abs(sum_error) > 1e-10:
            # Distribute correction evenly
            x = x - sum_error / N
            x = np.clip(x, -0.5, 0.5)
        
        h = 0.5 + x
        
        # Verify constraints
        if np.any(h < -1e-10) or np.any(h > 1 + 1e-10):
            h = np.clip(h, 0, 1)
        
        # Compute score: max_k C_k
        # C_k = N/2 - autocorr_h(k)
        # autocorr_h(k) = sum_i h_i * h_{i+k mod N}
        
        # Use FFT for efficient computation
        H = np.fft.fft(h)
        autocorr = np.fft.ifft(np.abs(H)**2).real
        
        C_vals = N/2 - autocorr
        score = np.max(C_vals)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Final correction to ensure sum is exactly N/2
    sum_h = np.sum(best_h)
    if abs(sum_h - N/2) > 1e-10:
        best_h = best_h - (sum_h - N/2) / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
