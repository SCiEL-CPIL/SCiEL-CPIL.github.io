"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Local search to minimize max|f*f| / (sum f)^2
    # Using O(N) update for convolution
    
    rng = np.random.default_rng(42)
    
    # Initialize with biased random sequence to ensure non-zero sum
    # Bias towards 1 to keep S large
    p = 0.7
    f = rng.choice([-1, 1], size=N, p=[1-p, p])
    
    # Ensure sum is not zero
    s = np.sum(f)
    if abs(s) < 1e-10:
        f = np.ones(N)
        s = N
    
    # Compute initial convolution
    c = np.convolve(f, f, mode='full')
    max_abs = np.max(np.abs(c))
    score = max_abs / (s * s)
    
    best_f = f.copy()
    best_score = score
    
    # Local search
    # Number of iterations
    n_iters = 20000
    
    for _ in range(n_iters):
        i = rng.integers(0, N)
        old_val = f[i]
        new_val = -old_val
        d = new_val - old_val # +/- 2
        
        # Update convolution
        # Delta C[k] = 2 * d * f[k-i] + d^2 * delta[k, 2i]
        # Note: indices in c are 0 to 2N-2.
        # f[k-i] is valid for k-i in [0, N-1] -> k in [i, i+N-1]
        
        # Create a temporary copy to check improvement
        c_new = c.copy()
        
        # The term 2 * d * f[k-i]
        # We need to add this to c_new[i : i+N]
        # The values are 2 * d * f[0 : N]
        c_new[i : i+N] += 2 * d * f
        
        # The term d^2 * delta[k, 2i]
        if 2 * i < len(c_new):
            c_new[2 * i] += d * d
            
        # Update sum
        s_new = s + d
        
        if abs(s_new) < 1e-10:
            continue # Skip if sum becomes zero
            
        max_abs_new = np.max(np.abs(c_new))
        score_new = max_abs_new / (s_new * s_new)
        
        if score_new < score:
            # Accept
            f[i] = new_val
            c = c_new
            s = s_new
            score = score_new
            
            if score < best_score:
                best_score = score
                best_f = f.copy()
        else:
            # Revert (no change to f, c, s, score)
            pass
            
    return best_f
# EVOLVE-BLOCK-END
