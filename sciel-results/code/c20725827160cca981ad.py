"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with a random ±1 sequence
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    # Compute initial autocorrelation
    def compute_acf(x):
        n = len(x)
        f = np.fft.fft(x, n=2*n)
        acf = np.fft.ifft(f * np.conj(f)).real[:n]
        return acf
    
    acf = compute_acf(x)
    best_score = np.max(np.abs(acf[1:]))
    best_x = x.copy()
    
    # Local search with incremental updates
    # When we flip x[i], the change in acf[k] is:
    # For k=0: no change (x[i]*x[i] is still 1)
    # For k>0: acf[k] changes by -2*x[i]*x[i+k] if i+k < N, and -2*x[i]*x[i-k] if i-k >= 0
    # Actually, acf[k] = sum_{j=0}^{N-1-k} x[j]*x[j+k]
    # If we flip x[i], then for each k where i is in the range [0, N-1-k],
    # the term x[i]*x[i+k] (if i+k < N) or x[i-k]*x[i] (if i-k >= 0) changes.
    
    # This is complex. Let's just recompute acf each time for simplicity,
    # but limit the number of iterations.
    
    max_iters = 50000
    iter_count = 0
    no_improve = 0
    patience = 5000
    
    while time.time() < deadline and iter_count < max_iters and no_improve < patience:
        idx = np.random.randint(N)
        x_flipped = x.copy()
        x_flipped[idx] *= -1
        
        acf_f = compute_acf(x_flipped)
        score = np.max(np.abs(acf_f[1:]))
        
        if score < best_score:
            best_score = score
            best_x = x_flipped
            x = x_flipped
            acf = acf_f
            no_improve = 0
        else:
            no_improve += 1
        
        iter_count += 1
    
    return best_x
# EVOLVE-BLOCK-END
