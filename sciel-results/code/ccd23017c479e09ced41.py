import numpy as np
import time

N = 400

def construct():
    """
    Constructs a signed step function (sequence of +/- 1) of length N
    to minimize the peak autocorrelation sidelobe.
    
    Uses Iterated Local Search (ILS) with greedy single-bit flips 
    and random block perturbations.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit (evaluator kills at 150s, but we should be safe)
    
    # FFT size for linear autocorrelation (next power of 2 >= 2*N - 1)
    fft_size = 1 << (2 * N - 1).bit_length()
    
    # Helper to compute peak sidelobe
    def peak_sidelobe(seq):
        # Zero pad
        xp = np.zeros(fft_size, dtype=np.float64)
        xp[:N] = seq
        # FFT
        X = np.fft.fft(xp)
        # Power spectrum
        P = X * np.conj(X)
        # Inverse FFT to get autocorrelation
        R = np.fft.ifft(P).real
        # R[0] is N. R[1:N] are positive lags.
        # We want max absolute value of R[1:N]
        # Note: For real sequences, R is real and symmetric.
        return np.max(np.abs(R[1:N]))
    
    # Initialize with random sequence
    x = np.random.choice([-1.0, 1.0], size=N)
    
    best_x = x.copy()
    best_cost = peak_sidelobe(best_x)
    
    # Parameters for ILS
    # Perturbation block size range
    min_block = 5
    max_block = 15
    
    while time.time() - start_time < time_limit:
        # 1. Local Search: Greedy Single-Bit Flips
        # We iterate until no single bit flip improves the cost.
        improved = True
        while improved:
            improved = False
            current_cost = peak_sidelobe(x)
            
            # If we already beat the best, update best
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x.copy()
            
            # Check time frequently in local search
            if time.time() - start_time > time_limit - 5:
                break
                
            # Try all single bit flips
            best_flip_idx = -1
            best_flip_cost = current_cost
            
            # Optimization: We can update the autocorrelation incrementally, 
            # but for N=400, full recompute is fast enough and simpler.
            # However, to speed up, we can check a random subset or all?
            # All is better for greedy convergence.
            
            for i in range(N):
                x[i] *= -1.0
                new_cost = peak_sidelobe(x)
                
                if new_cost < best_flip_cost:
                    best_flip_cost = new_cost
                    best_flip_idx = i
                else:
                    # Revert immediately if it's not the best candidate yet? 
                    # No, we need to compare against the *current* best flip.
                    # But we must revert before trying the next i.
                    x[i] *= -1.0
                    continue
                
                # If this is the best so far, keep it flipped for now?
                # No, standard greedy: evaluate all, pick best, apply.
                # To save time, if we find a cost of 0, we can stop early? Unlikely.
                # Let's just evaluate all.
                x[i] *= -1.0 # Revert to original state before next iteration
                
            if best_flip_idx != -1:
                # Apply the best flip
                x[best_flip_idx] *= -1.0
                improved = True
            else:
                improved = False
        
        # 2. Perturbation: Flip a random block
        block_len = np.random.randint(min_block, max_block + 1)
        start_idx = np.random.randint(0, N - block_len + 1)
        x[start_idx:start_idx + block_len] *= -1.0
        
        # Check if perturbed solution is better (rare, but possible if local search stuck)
        perturbed_cost = peak_sidelobe(x)
        if perturbed_cost < best_cost:
            best_cost = perturbed_cost
            best_x = x.copy()
            
        # Small sleep to avoid 100% CPU if it's too fast? 
        # No, we want to do as many iterations as possible.
        
    return best_x

# The evaluator expects the function to return the sequence.
# The seed program returned `best_x`.
# We keep the interface exactly as `construct` returning the sequence.
