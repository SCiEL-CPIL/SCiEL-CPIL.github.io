"""Strategy: Directed Simulated Annealing for Autoconvolution Minimization."""
import time
import numpy as np
import random
from scipy.signal import fftconvolve

N = 600
DEADLINE = 145.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    
    # Precompute FFT size for efficient convolution
    # For linear convolution of two length-N arrays, we need length >= 2N-1.
    # Next fast FFT size is 2^10 = 1024 for N=600? No, 2^10=1024. 
    # Actually 2N-1 = 1199. Next power of 2 is 2048.
    # Or use scipy.fft.next_fast_len
    n_fft = 2048
    
    def compute_obj(x):
        """Compute max(autoconv) / sum(x^2)."""
        x = np.maximum(x, 0)
        norm = np.sum(x * x)
        if norm < 1e-15:
            return 1e15
        # Use FFT for convolution
        X = np.fft.fft(x, n=n_fft)
        conv = np.fft.ifft(np.abs(X)**2).real
        # The linear convolution is in the first 2N-1 elements
        # Note: ifft gives circular convolution. Since we padded to 2N-1 > N, 
        # the linear part is conv[0:2N-1] if we aligned properly?
        # Actually, fftconvolve is safer for linear conv, but slower.
        # Let's stick to fftconvolve for correctness and speed is okay for N=600.
        # Wait, the current program uses manual fft. Let's optimize that.
        # If we pad x to length L >= 2N-1, the linear conv is the first 2N-1 coeffs 
        # of the circular conv ONLY IF the wrap-around doesn't interfere.
        # With L=2048 and 2N-1=1199, there is plenty of space.
        # The linear conv result is indeed conv[0:2N-1] in the ifft output?
        # Yes, if the signal is zero-padded to length L, the circular convolution 
        # of length L matches linear convolution for indices 0 to L-1 
        # provided no aliasing. Since support is 0..N-1, conv support is 0..2N-2.
        # So conv[0:2N-1] is correct.
        
        return np.max(conv[:2*N-1]) / norm

    # Initial state: Uniform distribution
    # Uniform x gives a "sinc-like" autoconvolution.
    # We want to minimize the peak.
    best_x = np.ones(N)
    best_val = compute_obj(best_x)
    
    # Simulated Annealing
    # We want to reduce the peak. 
    # Moves: 
    # 1. Move mass from high-value bins to low-value bins (flattening)
    # 2. Random perturbation
    
    T_start = 0.1 * best_val
    T_end = 1e-6 * best_val
    # Number of iterations: We have ~145s.
    # compute_obj takes ~0.1-0.5ms? 
    # Let's estimate. FFT of 2048 is fast. 
    # Let's run as many as possible.
    
    current_x = best_x.copy()
    current_val = best_val
    
    # Precompute some indices for speed? 
    # We will pick random indices.
    
    iters = 0
    time_budget = 140.0  # Leave 5s for cleanup
    
    while time.time() - start_time < time_budget:
        # Cooling schedule
        # Exponential cooling
        # We don't know total iters, so use time-based or simple decay per iter
        # Let's decay T by a factor each iteration
        T = T_start * (T_end / T_start) ** (iters / 50000) 
        if T < T_end:
            T = T_end
            
        # Generate move
        # Strategy: 
        # Pick two indices i, j.
        # Move delta mass from i to j.
        # If x[i] is large and x[j] is small, this might reduce peak (by spreading).
        # If x[i] is small and x[j] is large, this might increase peak.
        
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i == j:
            continue
            
        # Amount to move
        # Scale with T and current values
        # Max possible move from i is x[i]
        max_move = current_x[i]
        if max_move < 1e-9:
            continue
            
        # Propose a move size. 
        # Larger moves early, smaller later.
        # delta = max_move * random.uniform(0, 1) * (T / T_start + 0.1)
        # Let's just try a fraction of the mass
        frac = random.uniform(0.1, 0.9)
        delta = max_move * frac
        
        # Apply move
        new_x = current_x.copy()
        new_x[i] -= delta
        new_x[j] += delta
        
        # Ensure non-negativity (already ensured by delta <= x[i])
        
        new_val = compute_obj(new_x)
        
        # Acceptance criterion
        if new_val < current_val:
            # Improvement
            current_x = new_x
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_x = current_x.copy()
        else:
            # Worse, accept with probability exp(-delta_E / T)
            delta_E = new_val - current_val
            # T is in same units as delta_E
            prob = np.exp(-delta_E / T)
            if random.random() < prob:
                current_x = new_x
                current_val = new_val
                
        iters += 1
        
        # Check time every 1000 iters to avoid overhead of time.time() every loop
        if iters % 1000 == 0 and time.time() - start_time > time_budget:
            break

    # Final polish: Local search around the best found
    # Try small adjustments to the best_x
    if time.time() - start_time < DEADLINE - 5:
        # Gradient-free local search: 
        # For each index, try moving a small amount to a neighbor
        # This is expensive, so do it sparsely.
        
        # Actually, let's just do a few more SA iterations with very low T
        T = T_end
        for _ in range(5000):
            if time.time() - start_time > DEADLINE - 5:
                break
            i = random.randint(0, N-1)
            j = random.randint(0, N-1)
            if i == j: continue
            max_move = current_x[i]
            if max_move < 1e-9: continue
            # Small move
            delta = max_move * random.uniform(0.01, 0.1)
            new_x = current_x.copy()
            new_x[i] -= delta
            new_x[j] += delta
            new_val = compute_obj(new_x)
            if new_val < current_val:
                current_x = new_x
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_x = current_x.copy()
            else:
                delta_E = new_val - current_val
                if random.random() < np.exp(-delta_E / T):
                    current_x = new_x
                    current_val = new_val

    best_x = np.maximum(best_x, 0)
    return list(best_x)
