# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Precompute indices for each value of x-r*y
    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    # Note: P[y, x] is the standard layout, but meshgrid with indexing='ij' gives X[i,j]=i, Y[i,j]=j.
    # The problem states P[y, x]. So row is y, col is x.
    # Let's stick to P[y, x] = P[row, col].
    # X[row, col] = col, Y[row, col] = row.
    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy') # X[y,x]=x, Y[y,x]=y
    # Wait, np.meshgrid(arange, arange, indexing='xy')
    # x_coords = arange, y_coords = arange
    # X[y, x] = x, Y[y, x] = y. Correct.

    # Values for slopes
    V_m1 = X - Y  # -9 to 9
    V_0 = X       # 0 to 9
    V_1 = X + Y   # 0 to 18
    V_2 = X + 2*Y # 0 to 27

    # We will use bincount for efficiency
    # For each slope, we need to sum P over cells with that value.
    # Since P is a 2D array, we can flatten it and use bincount with weights.
    # But the mapping from (y,x) to value is fixed.
    
    # Precompute the "bin index" for each cell for each slope
    # V_m1 values are -9..9. Shift by 9 to get 0..18.
    bin_idx_m1 = (V_m1 + 9).flatten()
    bin_idx_0 = V_0.flatten()
    bin_idx_1 = V_1.flatten()
    bin_idx_2 = V_2.flatten()

    num_bins_m1 = 19
    num_bins_0 = 10
    num_bins_1 = 19
    num_bins_2 = 28

    def compute_ratio_from_logP(logP_flat):
        # logP_flat is 1D array of length 100
        # Convert to probabilities via softmax to ensure sum=1 and P>0
        # Subtract max for numerical stability
        z = logP_flat - np.max(logP_flat)
        P_flat = np.exp(z)
        sum_P = np.sum(P_flat)
        if sum_P == 0:
            return 0.0 # Invalid
        P_flat = P_flat / sum_P
        
        # Compute entropies
        # D = H(x-y)
        counts_m1 = np.bincount(bin_idx_m1, weights=P_flat, minlength=num_bins_m1)
        # M = max(H(x), H(x+y), H(x+2y))
        counts_0 = np.bincount(bin_idx_0, weights=P_flat, minlength=num_bins_0)
        counts_1 = np.bincount(bin_idx_1, weights=P_flat, minlength=num_bins_1)
        counts_2 = np.bincount(bin_idx_2, weights=P_flat, minlength=num_bins_2)
        
        def entropy(counts):
            c = counts[counts > 1e-15]
            if len(c) == 0:
                return 0.0
            return -np.sum(c * np.log2(c))
            
        H_m1 = entropy(counts_m1)
        H_0 = entropy(counts_0)
        H_1 = entropy(counts_1)
        H_2 = entropy(counts_2)
        
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 0.0
            
        return H_m1 / M

    def objective(logP_flat):
        # We want to maximize ratio, so minimize negative ratio
        r = compute_ratio_from_logP(logP_flat)
        return -r

    start_time = time.time()
    time_limit = 590.0 # Safety margin
    
    best_score = 0.0
    best_P = np.ones((10, 10)) / 100.0
    
    # Initial points
    # 1. Uniform
    # 2. Random
    # 3. Structured (peaks)
    
    init_points = []
    # Uniform
    init_points.append(np.zeros(100))
    
    # Random
    rng = np.random.default_rng(42)
    for _ in range(20):
        init_points.append(rng.normal(0, 1, 100))
        
    # Structured: peaks at corners, center, etc.
    # Create a grid of peaks
    for y in range(10):
        for x in range(10):
            if (x+y) % 3 == 0: # Sparse peaks
                p = np.zeros(100)
                p[y*10+x] = 10.0 # High weight in log space
                init_points.append(p)
                
    # Add some random sparse peaks
    for _ in range(20):
        p = np.zeros(100)
        idx = rng.choice(100, size=10, replace=False)
        p[idx] = 10.0
        init_points.append(p)

    for i, init in enumerate(init_points):
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = minimize(objective, init, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-8})
            
            if res.success or res.fun < -best_score:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    # Reconstruct P
                    z = res.x - np.max(res.x)
                    P_flat = np.exp(z)
                    P_flat = P_flat / np.sum(P_flat)
                    best_P = P_flat.reshape(10, 10)
        except Exception:
            pass

    # Ensure validity
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-9:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = best_P / s
        
    return best_P

# EVOLVE-BLOCK-END
