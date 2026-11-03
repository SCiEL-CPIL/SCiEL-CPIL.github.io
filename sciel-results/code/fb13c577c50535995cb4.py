"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autocorr_peak_sidelobe(f):
    """Compute the max |R(k)| for k != 0 using FFT-based autocorrelation."""
    n = len(f)
    nfft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(f, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:2*n-1]
    # acf has length 2*n-1, with acf[n-1] being R(0)
    left = acf[0:n-1]   # R(-k) for k=1..n-1
    right = acf[n:2*n-1] # R(k) for k=1..n-1
    return max(np.max(np.abs(left)), np.max(np.abs(right)))

def _eval(f):
    return _autocorr_peak_sidelobe(f)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Use a fixed seed for reproducibility but allow some randomness
    random.seed(42)
    np.random.seed(42)
    
    best_f = None
    best_val = float('inf')
    
    # Multiple restarts with different strategies
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Different initialization strategies
        if restart == 0:
            # All ones (seed)
            f = np.ones(N, dtype=np.float64)
        elif restart == 1:
            # Random signs
            f = np.random.choice([-1.0, 1.0], size=N)
        elif restart == 2:
            # Alternating pattern
            f = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
        elif restart == 3:
            # Block pattern: +++++-----+++++-----...
            block_len = 10
            f = np.repeat(np.array([1.0]*5 + [-1.0]*5, dtype=np.float64), N // (2*block_len) + 1)[:N]
        elif restart == 4:
            # Longer blocks
            block_len = 20
            f = np.repeat(np.array([1.0]*10 + [-1.0]*10, dtype=np.float64), N // (2*block_len) + 1)[:N]
        elif restart == 5:
            # Random with bias toward long runs
            f = np.ones(N, dtype=np.float64)
            pos = 0
            while pos < N:
                run_len = random.choice([3, 5, 8, 12, 16])
                val = 1.0 if random.random() < 0.5 else -1.0
                f[pos:pos+run_len] = val
                pos += run_len
        elif restart == 6:
            # Based on low-discrepancy sequence (van der Corput-like)
            f = np.zeros(N, dtype=np.float64)
            for i in range(N):
                # Use bit-reversal to create pseudo-random but structured pattern
                x = i
                b = 0
                while x > 0:
                    b = (b << 1) | (x & 1)
                    x >>= 1
                f[i] = 1.0 if b % 2 == 0 else -1.0
        elif restart == 7:
            # Sine-modulated signs
            f = np.sign(np.sin(2 * np.pi * np.arange(N) / 37.0))
            f[f == 0] = 1.0
        elif restart == 8:
            # Another structured pattern
            f = np.ones(N, dtype=np.float64)
            for i in range(N):
                if (i // 8) % 3 == 1:
                    f[i] = -1.0
        else:
            # Random
            f = np.random.choice([-1.0, 1.0], size=N)
        
        # Evaluate initial
        val = _eval(f)
        if val < best_val:
            best_val = val
            best_f = f.copy()
        
        # Simulated annealing with adaptive block sizes
        T = 5.0
        T_min = 0.05
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 5000
        
        while T > T_min and no_improve < max_no_improve:
            if time.time() > deadline - 5.0:
                break
            
            # Adaptive block size selection
            # Larger blocks when stuck, smaller when improving
            if no_improve > 1000:
                bs = random.choice([16, 24, 32, 48, 64])
            elif no_improve > 500:
                bs = random.choice([8, 16, 24, 32])
            else:
                bs = random.choice([4, 8, 12, 16, 24])
            
            start = random.randint(0, N - bs)
            
            # Flip the block
            f[start:start+bs] *= -1
            
            new_val = _eval(f)
            delta = new_val - best_val
            
            if delta < 0:
                # Improvement
                if new_val < best_val:
                    best_val = new_val
                    best_f = f.copy()
                    no_improve = 0
                T = max(T * cooling, T_min)
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < np.exp(-delta / T):
                    T = max(T * cooling, T_min)
                else:
                    # Reject
                    f[start:start+bs] *= -1
                    no_improve += 1
                T = max(T * cooling, T_min)
        
        # Greedy polish: single-bit flips
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5.0:
                    break
                f[i] *= -1
                new_val = _eval(f)
                if new_val < best_val:
                    best_val = new_val
                    best_f = f.copy()
                    improved = True
                else:
                    f[i] *= -1
    
    return best_f
# EVOLVE-BLOCK-END
