"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(s):
    """Compute max |autoconvolution| for signed sequence s (values in {-1,1}).
    Autoconvolution: (s * s)[k] = sum_i s[i] * s[i+k]
    We want to minimize the upper bound on C3, which relates to max |autocorrelation|.
    """
    # Use FFT for efficient computation
    n = len(s)
    # Zero-pad to avoid circular convolution issues
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(s, size)
    ac = np.fft.irfft(f * np.conj(f), size)[:2 * n - 1]
    # The autoconvolution values
    return np.max(np.abs(ac))


def _energy(s):
    """Energy function: max |autoconvolution|"""
    return _autoconv_max_abs(s)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Initialize with random signs
    np.random.seed(42)
    s = np.random.choice([-1, 1], size=N)
    
    # Also try all-ones as baseline
    s_ones = np.ones(N)
    e_ones = _energy(s_ones)
    
    e = _energy(s)
    if e_ones < e:
        s = s_ones.copy()
        e = e_ones
    
    best_s = s.copy()
    best_e = e
    
    # Simulated annealing with block flips
    T = 10.0  # initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline and T > T_min:
        # Choose block size: mix of small and large flips
        r = np.random.random()
        if r < 0.3:
            block_size = 1
        elif r < 0.6:
            block_size = np.random.randint(2, 8)
        elif r < 0.85:
            block_size = np.random.randint(8, 20)
        else:
            block_size = np.random.randint(20, 50)
        
        if block_size >= N:
            block_size = N - 1
        
        # Choose random position
        pos = np.random.randint(0, N - block_size + 1)
        
        # Try flipping the block
        s_new = s.copy()
        s_new[pos:pos + block_size] *= -1
        
        e_new = _energy(s_new)
        delta_e = e_new - e
        
        # Accept if better, or with probability exp(-delta_e / T)
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            s = s_new
            e = e_new
            if e < best_e:
                best_e = e
                best_s = s.copy()
        
        # Cool down
        T *= cooling_rate
        
        # Check deadline more frequently
        if time.time() > deadline - 5.0:
            break
    
    # Final local search polish with single flips
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(N):
                if time.time() > deadline - 1.0:
                    break
                s_trial = best_s.copy()
                s_trial[i] *= -1
                e_trial = _energy(s_trial)
                if e_trial < best_e:
                    best_s = s_trial
                    best_e = e_trial
                    improved = True
    
    return best_s
# EVOLVE-BLOCK-END
