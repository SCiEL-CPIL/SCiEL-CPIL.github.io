import numpy as np
from scipy.optimize import minimize

N = 400

def _score_continuous(x):
    # x is a real-valued vector in [-1, 1]
    # Compute full autocorrelation
    r = np.correlate(x, x, mode='full')
    max_abs_r = np.max(np.abs(r))
    s = np.sum(x)
    s_sq = s * s
    if s_sq < 1e-12:
        return 1e18
    return max_abs_r / s_sq

def _score_binary(x):
    # x is a vector of +/-1
    r = np.correlate(x, x, mode='full')
    max_abs_r = np.max(np.abs(r))
    s = np.sum(x)
    s_sq = s * s
    if s_sq == 0:
        return float('inf')
    return max_abs_r / s_sq

def construct():
    # Start with a random binary sequence
    np.random.seed(12345)
    x_bin = np.random.choice([-1.0, 1.0], size=N)
    
    # Local search on binary domain with incremental updates
    best_x = x_bin.copy()
    best_score = _score_binary(best_x)
    
    # Precompute correlation
    r = np.correlate(best_x, best_x, mode='full')
    s = np.sum(best_x)
    
    def update_correlation(r, x, idx, new_val):
        # Update r when x[idx] changes from old_val to new_val
        # r[k] = sum_i x[i] * x[i+k-N+1] (roughly, depends on indexing)
        # Actually np.correlate(x, x, 'full')[k] = sum_i x[i] * x[i + k - (N-1)]
        # Let's use the property: if x[idx] changes by delta, then r changes by delta * x shifted
        old_val = x[idx]
        delta = new_val - old_val
        x[idx] = new_val
        # r_new[k] = r_old[k] + delta * x[k - idx] + delta * x[idx - k] ... this is getting complex
        # For N=400, recomputing is O(N^2) = 160k ops, which is fine for ~5000 iterations
        return np.correlate(x, x, mode='full')
    
    num_iterations = 20000
    for it in range(num_iterations):
        # Pick a random index to flip
        idx = np.random.randint(N)
        old_val = best_x[idx]
        best_x[idx] = -old_val
        
        new_score = _score_binary(best_x)
        
        if new_score < best_score:
            best_score = new_score
        else:
            best_x[idx] = old_val
            # Occasionally do a multi-flip move
            if np.random.random() < 0.1:
                idx2 = np.random.randint(N)
                if idx2 != idx:
                    best_x[idx] = -best_x[idx]
                    best_x[idx2] = -best_x[idx2]
                    new_score2 = _score_binary(best_x)
                    if new_score2 < best_score:
                        best_score = new_score2
                    else:
                        best_x[idx] = -best_x[idx]
                        best_x[idx2] = -best_x[idx2]
    
    # Now try to refine using a continuous optimization step, then round
    # This might help escape local minima
    x_cont = best_x.astype(float)
    
    # Use L-BFGS-B to minimize the continuous score
    try:
        result = minimize(_score_continuous, x_cont, method='L-BFGS-B', 
                         bounds=[(-1, 1)] * N, 
                         options={'maxiter': 200, 'ftol': 1e-9})
        x_opt = result.x
        # Round to nearest binary value
        x_rounded = np.where(x_opt > 0, 1.0, -1.0)
        score_rounded = _score_binary(x_rounded)
        
        if score_rounded < best_score:
            best_x = x_rounded
            best_score = score_rounded
    except:
        pass
    
    # Another round of local search on the improved binary sequence
    for it in range(5000):
        idx = np.random.randint(N)
        old_val = best_x[idx]
        best_x[idx] = -old_val
        new_score = _score_binary(best_x)
        if new_score < best_score:
            best_score = new_score
        else:
            best_x[idx] = old_val
    
    return best_x.astype(int)
