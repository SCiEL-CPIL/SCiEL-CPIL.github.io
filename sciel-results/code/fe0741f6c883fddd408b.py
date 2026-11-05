import time
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    Vectorized using NumPy for speed.
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    max_abs = 0
    
    # We iterate over step sizes d.
    # For a given d, the progressions are determined by start index i (0 <= i < d).
    # An AP starting at i with step d is s[i], s[i+d], s[i+2d], ...
    # We want max |sum_{k=0}^{L-1} s[i + k*d]| for all valid i, L.
    
    # Optimization:
    # For a fixed d, we can process all starting indices i in [0, d) simultaneously.
    # The maximum length for any start is max_len = (n - 1) // d + 1.
    # We create a matrix M where M[i, k] = s[i + k*d] if valid, else 0.
    # Then prefix sums along axis 1 give us sums of APs starting at i.
    # Specifically, prefix_sum[i, j] = sum_{k=0}^j M[i, k].
    # The max absolute sum for start i is max_j |prefix_sum[i, j]|.
    # The overall max is max_i max_j |prefix_sum[i, j]|.
    
    for d in range(1, n + 1):
        max_len = (n - 1) // d + 1
        
        # Create indices for the matrix M
        # i = 0..d-1, k = 0..max_len-1
        # Index in s: i + k*d
        i = np.arange(d)[:, None]
        k = np.arange(max_len)[None, :]
        idx = i + k * d
        
        # Mask invalid indices (where i + k*d >= n)
        valid = idx < n
        
        # Safe gather:
        # We need to extract s[idx] but only where valid.
        # To avoid out-of-bounds errors, we can clamp indices to n-1 for gathering,
        # and then zero out the invalid positions.
        # Alternatively, use np.take with mode='clip' if available, or manual masking.
        # Manual masking is safer and fast enough for n=256.
        
        # Create a zero-padded array or just use np.where on indices?
        # Actually, we can just create a flat index array and use np.take,
        # but we must ensure indices are within [0, n-1].
        # Let's clamp indices to [0, n-1] for the gather operation.
        idx_clipped = np.clip(idx, 0, n - 1)
        
        # Gather values
        vals = s[idx_clipped]
        
        # Zero out invalid positions
        vals[~valid] = 0
        
        # Compute prefix sums along the length axis (axis=1)
        cumsum = np.cumsum(vals, axis=1)
        
        # Find max absolute value in this block
        current_max = np.max(np.abs(cumsum))
        if current_max > max_abs:
            max_abs = current_max
            
        # Early termination if we hit a theoretical lower bound? 
        # Theoretical min for n=256 is likely small (e.g., 2 or 3).
        # If max_abs == 1, we can't do better than 1 (since singletons have sum 1).
        # Actually, singleton APs have sum +/-1, so max_abs >= 1 always.
        # If max_abs == 1, we found the optimal solution.
        if max_abs == 1:
            return 1
            
    return int(max_abs)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe (evaluator kills at 90s)
    
    n = 256
    rng = np.random.default_rng(12345)
    
    # Initial random signs
    signs = rng.choice([-1, 1], size=n).astype(np.int16)
    
    # Evaluate initial
    current_disc = _max_ap_discrepancy(signs)
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Greedy Hill Climbing
    # In each iteration, try flipping every bit and pick the one that minimizes discrepancy.
    # If no flip improves, stop (local minimum).
    # To escape local minima, we can use random restarts or simulated annealing steps,
    # but pure greedy is often strong for this kind of structured problem.
    # Given the time budget, we will do a few random restarts if we get stuck early.
    
    # However, a full greedy step (256 flips * eval) might be slow if eval is slow.
    # Let's estimate eval speed.
    # n=256. Loop d=1..256.
    # d=1: max_len=256. Matrix 1x256.
    # d=256: max_len=1. Matrix 256x1.
    # Total ops ~ sum_{d=1}^{256} d * (n/d) = n^2 = 65536 elements processed in cumsum.
    # NumPy cumsum is very fast. Likely < 1ms per eval.
    # 256 flips * 1ms = 256ms per full greedy step.
    # 80s / 0.256s = ~300 full greedy steps.
    # This is a very strong search.
    
    # We will implement a "best-first" greedy descent.
    # We also add a small amount of stochasticity (random restarts) if we plateau.
    
    max_restarts = 5
    current_signs = signs.copy()
    current_disc = _max_ap_discrepancy(current_signs)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # If this is not the first restart, randomize the current signs
        if restart > 0:
            # Random perturbation: flip 10% of bits
            flip_mask = rng.random(n) < 0.1
            current_signs[flip_mask] *= -1
            current_disc = _max_ap_discrepancy(current_signs)
            
        # Greedy descent
        improved = True
        while improved:
            if time.time() > deadline:
                break
                
            improved = False
            best_flip_idx = -1
            best_new_disc = current_disc
            
            # Try flipping each bit
            for i in range(n):
                # Flip bit i
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                
                new_disc = _max_ap_discrepancy(current_signs)
                
                # Revert
                current_signs[i] = old_sign
                
                if new_disc < best_new_disc:
                    best_new_disc = new_disc
                    best_flip_idx = i
                    
            if best_flip_idx != -1:
                # Apply the best flip
                current_signs[best_flip_idx] *= -1
                current_disc = best_new_disc
                improved = True
                
                # Update best
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = current_signs.copy()
                    
                # Early exit if optimal
                if best_disc == 1:
                    return best_signs.tolist()
            else:
                # No improvement found, local minimum
                break
                
    return best_signs.tolist()
