import numpy as np
import scipy.optimize
import time
import itertools

N = 14
D = 3

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def objective(x):
        # x is a flat array of length N*D
        pts = x.reshape(N, D)
        # Calculate pairwise squared distances
        # Using broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        # Mask diagonal (self-distance)
        np.fill_diagonal(dist_sq, np.inf)
        
        min_dist_sq = np.min(dist_sq)
        max_dist_sq = np.max(dist_sq)
        
        # Avoid division by zero
        if min_dist_sq < 1e-12:
            return 1e12
        
        ratio_sq = max_dist_sq / min_dist_sq
        return ratio_sq

    def halton_sequence(n, base):
        seq = []
        f = 1.0
        r = 0.0
        for i in range(n):
            f = 1.0 / base
            while i > 0:
                i, rem = divmod(i, base)
                r += rem * f
                f /= base
            seq.append(r)
        return seq

    best_score = float('inf')
    best_points = None

    # Try a few different initializations
    # 1. Halton sequence scaled to a cube
    # 2. Random points in a cube
    # 3. Random points on a sphere
    
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        if restart == 0:
            # Halton
            h1 = halton_sequence(N, 2)
            h2 = halton_sequence(N, 3)
            h3 = halton_sequence(N, 5)
            x0 = np.column_stack([h1, h2, h3])
        else:
            # Random uniform in cube [-1, 1]
            x0 = np.random.uniform(-1, 1, size=(N, D))
            
        x0_flat = x0.flatten()
        
        # Optimize
        try:
            res = scipy.optimize.minimize(
                objective, 
                x0_flat, 
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                if res.fun < best_score:
                    best_score = res.fun
                    best_points = res.x.reshape(N, D)
        except Exception:
            continue
            
    # If no optimization succeeded, return the initial best or a default
    if best_points is None:
        # Fallback: Halton
        h1 = halton_sequence(N, 2)
        h2 = halton_sequence(N, 3)
        h3 = halton_sequence(N, 5)
        best_points = np.column_stack([h1, h2, h3])

    return best_points
