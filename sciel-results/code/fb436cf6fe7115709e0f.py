import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    # Initialize with random signs
    h = np.random.choice([-1, 1], size=N)
    
    # Function to compute PSL (Peak Sidelobe Level)
    # Autocorrelation R[k] = sum(h[j] * h[j+k])
    # We want max(|R[k]|) for k != 0
    def compute_psl(h):
        # FFT based autocorrelation is faster for large N, but N=400 is small
        # Direct calculation or FFT?
        # FFT: R = ifft(|fft(h)|^2).real
        # This gives circular autocorrelation.
        # For linear autocorrelation, we need to zero-pad or handle carefully.
        # However, "step function on [-1/4, 1/4]" with 400 intervals usually implies
        # the sequence is treated as a vector. The autocorrelation definition 
        # in these combinatorial problems is usually linear (sum over valid j).
        # Let's use linear autocorrelation.
        
        # Efficient linear autocorrelation using FFT with zero padding
        n_fft = 2 * N
        H = np.fft.fft(h, n=n_fft)
        R = np.fft.ifft(np.abs(H)**2).real
        # R[0:N] contains the linear autocorrelation for lags 0 to N-1
        # R[N:2N] contains the negative lags (conjugate symmetric)
        # We only care about lags 1 to N-1
        return np.max(np.abs(R[1:N]))

    current_psl = compute_psl(h)
    best_h = h.copy()
    best_psl = current_psl
    
    # Simulated Annealing
    T = 10.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 1e-6
    
    # Pre-compute indices for potential updates? 
    # Actually, just iterating over all k for a flipped i is O(N).
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random index to flip
        i = np.random.randint(N)
        
        # Try flipping h[i]
        old_val = h[i]
        h[i] = -old_val
        
        # Update Autocorrelation R
        # We need to update R efficiently.
        # Instead of recomputing full FFT, let's maintain R explicitly?
        # Updating R explicitly for one flip is O(N).
        # Let's maintain R as a numpy array.
        
        # Wait, I haven't initialized R in the loop yet. Let's restructure.
        
        # For now, let's just recompute PSL via FFT. 
        # N=400, FFT is very fast. 
        # Let's benchmark: 1e6 FFTs of size 800?
        # np.fft.fft is C-optimized. It should handle thousands per second.
        # 115 seconds * 1000 steps/sec = 115,000 steps. This is decent.
        
        new_psl = compute_psl(h)
        
        if new_psl < current_psl:
            current_psl = new_psl
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
            T = max(T * cooling_rate, min_T)
        else:
            # Accept with probability exp(-(new_psl - current_psl) / T)
            delta = new_psl - current_psl
            if np.random.rand() < np.exp(-delta / T):
                current_psl = new_psl
                T = max(T * cooling_rate, min_T)
            else:
                # Reject
                h[i] = old_val
                T = max(T * cooling_rate, min_T)
                
    return best_h
