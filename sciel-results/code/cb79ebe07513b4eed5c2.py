"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autocorr(h):
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac

def _max_sidelobe(ac):
    return np.max(np.abs(ac[1:N]))

def _delta_autocorr(h, i):
    delta = np.zeros(N)
    hi = h[i]
    # k ranges 1 to N-1
    # Part 1: i+k < N => k < N-i
    k_max_1 = N - i - 1
    if k_max_1 > 0:
        delta[1:k_max_1+1] += -2 * hi * h[i+1:i+k_max_1+1]
        
    # Part 2: i-k >= 0 => k <= i
    k_max_2 = i
    if k_max_2 > 0:
        delta[1:k_max_2+1] += -2 * h[i-k_max_2:i] * hi
        
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    np.random.seed(42)
    h = np.random.choice([-1, 1], size=N)
    ac = _compute_autocorr(h)
    current_max = _max_sidelobe(ac)
    
    best_h = h.copy()
    best_val = current_max
    
    while time.time() < deadline:
        best_gain = 0
        best_i = -1
        best_delta = None
        
        for i in range(N):
            delta = _delta_autocorr(h, i)
            new_ac = ac + delta
            new_max = np.max(np.abs(new_ac[1:N]))
            
            gain = current_max - new_max
            if gain > best_gain:
                best_gain = gain
                best_i = i
                best_delta = delta
        
        if best_i != -1:
            h[best_i] *= -1
            ac += best_delta
            current_max = _max_sidelobe(ac)
            
            if current_max < best_val:
                best_val = current_max
                best_h = h.copy()
        else:
            # No improvement, do a random kick
            # Flip 10 random bits
            indices = np.random.choice(N, size=10, replace=False)
            h[indices] *= -1
            ac = _compute_autocorr(h)
            current_max = _max_sidelobe(ac)
            
            if current_max < best_val:
                best_val = current_max
                best_h = h.copy()
                
    return best_h
# EVOLVE-BLOCK-END
