"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

def compute_score(heights):
    """Compute the normalized maximum of the autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    n = len(heights)
    # Compute autoconvolution via FFT
    f = np.fft.fft(heights)
    ac = np.fft.ifft(f * np.conj(f)).real
    s = np.sum(heights)
    if s < 1e-12:
        return 0.0
    return np.max(ac) * N / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize h
    # Start with uniform, but perturb to escape symmetry
    h = np.ones(N)
    h += 0.01 * np.random.randn(N)
    h = np.maximum(h, 0)
    h /= np.linalg.norm(h)
    
    best_h = h.copy()
    best_score = compute_score(best_h)
    
    # Softmax temperature
    T = 1.0 
    # Learning rate
    lr = 0.1
    
    for i in range(5000):
        if time.time() > deadline:
            break
        
        # Compute ac
        f = np.fft.fft(h)
        ac = np.fft.ifft(f * np.conj(f)).real
        
        # Compute weights for soft-max
        # ac values are positive. Scale them for softmax stability.
        # Max ac is around 1/N if ||h||_2=1? No, for uniform ||h||_2=1, ac[0]=1.
        # So ac[0] is the largest.
        ac_scaled = ac / T
        
        # Softmax weights
        ac_max = np.max(ac_scaled)
        exp_ac = np.exp(ac_scaled - ac_max)
        w = exp_ac / np.sum(exp_ac)
        
        # Gradient of sum(w_k * ac_k) wrt h
        # grad = 2 * ifft(conj(fft(w)) * fft(h))
        fw = np.fft.fft(w)
        fh = np.fft.fft(h)
        grad = 2.0 * np.fft.ifft(np.conj(fw) * fh).real
        
        # Update h
        h_new = h - lr * grad
        
        # Enforce non-negativity
        h_new = np.maximum(h_new, 0)
        
        # Normalize to unit L2 norm
        norm = np.linalg.norm(h_new)
        if norm < 1e-12:
            # If it died, restart with uniform
            h_new = np.ones(N)
            h_new += 0.01 * np.random.randn(N)
            h_new = np.maximum(h_new, 0)
            norm = np.linalg.norm(h_new)
        h_new /= norm
        
        # Check score
        score = compute_score(h_new)
        if score < best_score:
            best_score = score
            best_h = h_new.copy()
            h = h_new.copy()
            # Maybe decrease T to sharpen the penalty on the max?
            T *= 0.95
            if T < 0.01:
                T = 0.01
        else:
            # If no improvement, adjust learning rate
            lr *= 0.9
            if lr < 1e-5:
                break
                
    return best_h.tolist()

# EVOLVE-BLOCK-END
