import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def _compute_overlaps(h):
    """
    Compute C_k = sum_i h_i (1 - h_{i+k}) for k=1..N-1.
    """
    g = 1.0 - h
    # Linear correlation
    # c[k] = sum_i h[i] * g[i+k]
    # np.correlate(h, g, 'full') gives lags from -(N-1) to N-1.
    # Index N is lag 0.
    # Indices N+1 to 2N-1 are positive lags 1 to N-1.
    # Indices 1 to N-1 are negative lags -(N-1) to -1.
    c = np.correlate(h, g, mode='full')
    # We want lags 1 to N-1 (and -1 to -(N-1) which are symmetric if h is real? 
    # C_k = sum h_i (1-h_{i+k}). 
    # C_{-k} = sum h_i (1-h_{i-k}) = sum h_{i+k} (1-h_i) = sum h_i (1-h_{i-k}) ... 
    # Actually C_k and C_{-k} are not necessarily equal unless h is symmetric.
    # But the problem asks for "shifted 1-h". Usually implies all shifts.
    # The max overlap is max over all k != 0.
    
    # c[N+k] corresponds to shift k.
    # c[N-k] corresponds to shift -k.
    # Let's just take all non-zero lags.
    c_nonzero = np.concatenate([c[:N], c[N+1:]])
    return c_nonzero

def _objective_and_grad(h_flat, beta, lambda_sum):
    """
    Objective: Softmax of overlaps + penalty for sum constraint.
    """
    h = h_flat
    c = _compute_overlaps(h)
    
    # Log-sum-exp for numerical stability
    c_max = np.max(c)
    lse = c_max + np.log(np.sum(np.exp(beta * (c - c_max)))) / beta
    
    sum_diff = np.sum(h) - TARGET_SUM
    penalty = lambda_sum * sum_diff**2
    
    obj = lse + penalty
    
    # Gradient
    # Gradient of LSE: exp(beta * (c - c_max)) / sum(exp(...))
    weights = np.exp(beta * (c - c_max))
    weights /= np.sum(weights)
    
    # Gradient of c_k with respect to h
    # c_k = sum_i h_i (1 - h_{i+k})
    # d c_k / d h_j = (1 - h_{j+k}) if j is the first index, 
    #                 - h_{j-k} if j is the second index?
    # Let's derive carefully.
    # c_k = sum_i h_i - sum_i h_i h_{i+k}
    # d c_k / d h_j = 1 - (h_{j+k} + h_{j-k})  (with boundary conditions h_out=0)
    
    # This is O(N^2) to compute full Jacobian.
    # For L-BFGS-B, we can provide numerical gradients if analytical is too slow/complex, 
    # but N=100 is small enough for analytical if optimized.
    
    # Actually, let's just use numerical gradient for simplicity and robustness in this step, 
    # or implement analytical. 
    # Analytical grad:
    # grad_h[j] = sum_k weights[k] * (1 - h_{j+k} - h_{j-k}) + 2 * lambda_sum * sum_diff
    # Note: c array corresponds to specific lags.
    # c[0] is lag 1? 
    # c[:N] are lags -(N-1)..-1. c[N+1:] are lags 1..N-1.
    
    # Let's map c indices to lags.
    # lags_neg = np.arange(-(N-1), 0)
    # lags_pos = np.arange(1, N)
    
    # It is easier to compute grad by looping or using convolution-like operations.
    # Given N=100, O(N^2) per gradient step is 10,000 ops, which is fast.
    
    grad = np.zeros(N)
    # We need to know which c_k corresponds to which lag.
    # c_vec = _compute_overlaps(h)
    # c_vec[0] -> lag -99, c_vec[1] -> lag -98, ..., c_vec[98] -> lag -1
    # c_vec[99] -> lag 1, ..., c_vec[198] -> lag 99
    
    lags = np.concatenate([np.arange(-(N-1), 0), np.arange(1, N)])
    
    for k_idx, lag in enumerate(lags):
        w = weights[k_idx]
        if w < 1e-10: continue
        
        # Contribution to grad from this lag
        # d c_lag / d h_j = 1 - (h_{j+lag} + h_{j-lag})
        # We can vectorize this.
        # Let's do it for all j at once for this lag? No, loop over lags is N, inner is vectorized.
        
        # h_shifted_pos: h[j+lag]
        # h_shifted_neg: h[j-lag]
        
        # Create arrays of indices
        j = np.arange(N)
        
        # Indices for h[j+lag]
        idx_pos = j + lag
        valid_pos = (idx_pos >= 0) & (idx_pos < N)
        h_pos = np.zeros(N)
        h_pos[valid_pos] = h[idx_pos[valid_pos]]
        
        # Indices for h[j-lag]
        idx_neg = j - lag
        valid_neg = (idx_neg >= 0) & (idx_neg < N)
        h_neg = np.zeros(N)
        h_neg[valid_neg] = h[idx_neg[valid_neg]]
        
        grad += w * (1.0 - h_pos - h_neg)
        
    grad += 2.0 * lambda_sum * sum_diff
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Initial candidates
    candidates = []
    
    # 1. Alternating
    h_alt = np.full(N, 0.5)
    candidates.append(h_alt)
    
    # 2. Sine waves
    t = np.arange(N)
    for freq in [1, 2, 3, 5, 10]:
        for amp in [0.3, 0.4, 0.49]:
            h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * t / N)
            candidates.append(np.clip(h_sin, 0, 1))
            
    # 3. Random Binary
    rng = np.random.default_rng(42)
    for _ in range(5):
        h_bin = rng.choice([0, 1], size=N)
        # Adjust sum to be exactly 50
        # If sum is not 50, swap some values
        s = np.sum(h_bin)
        if s > 50:
            idx_ones = np.where(h_bin == 1)[0]
            to_swap = s - 50
            swap_idx = rng.choice(idx_ones, size=to_swap, replace=False)
            h_bin[swap_idx] = 0
        elif s < 50:
            idx_zeros = np.where(h_bin == 0)[0]
            to_swap = 50 - s
            swap_idx = rng.choice(idx_zeros, size=to_swap, replace=False)
            h_bin[swap_idx] = 1
        candidates.append(h_bin.astype(float))
        
    # 4. Random Uniform
    for _ in range(5):
        h_rand = rng.uniform(0, 1, size=N)
        # Project to sum 50 roughly
        h_rand = h_rand - np.mean(h_rand) + 0.5
        h_rand = np.clip(h_rand, 0, 1)
        # Fine tune sum
        diff = 50 - np.sum(h_rand)
        if diff != 0:
            # Distribute diff
            step = diff / N
            h_rand += step
            h_rand = np.clip(h_rand, 0, 1)
        candidates.append(h_rand)
        
    # Optimize each candidate
    for i, h_init in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
            
        h_init = h_init.copy()
        
        # Ensure sum is close to 50 initially
        # Use a simple projection
        for _ in range(10):
            diff = 50 - np.sum(h_init)
            if abs(diff) < 1e-6: break
            mask = h_init < 1.0 if diff > 0 else h_init > 0.0
            if not np.any(mask): break
            step = diff / np.sum(mask)
            h_init[mask] += step
            h_init = np.clip(h_init, 0, 1)
            
        # Run optimization with annealed beta
        # Start with low beta, increase
        current_h = h_init.copy()
        current_obj = float('inf')
        
        # We will run a few iterations of L-BFGS-B with increasing beta
        # To save time, we run one optimization per candidate with a fixed high beta?
        # Or a short sequence?
        # Given time limit, let's try 2-3 beta levels.
        
        betas = [10.0, 50.0, 100.0]
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            # Define objective for this beta
            lambda_sum = 1000.0 # Strong penalty for sum constraint
            
            def obj_func(x):
                val, _ = _objective_and_grad(x, beta, lambda_sum)
                return val
                
            def grad_func(x):
                _, g = _objective_and_grad(x, beta, lambda_sum)
                return g
                
            try:
                res = minimize(obj_func, current_h, method='L-BFGS-B', jac=grad_func,
                               bounds=[(0, 1)] * N,
                               options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
                current_h = res.x
                current_obj = res.fun
            except Exception as e:
                # If optimization fails, keep previous best
                pass
                
        # Evaluate true max overlap
        c_final = _compute_overlaps(current_h)
        true_max = np.max(c_final)
        
        # Check sum constraint
        if abs(np.sum(current_h) - 50) > 1e-5:
            # Final adjustment
            diff = 50 - np.sum(current_h)
            step = diff / N
            current_h += step
            current_h = np.clip(current_h, 0, 1)
            # Re-evaluate
            c_final = _compute_overlaps(current_h)
            true_max = np.max(c_final)
            
        if true_max < best_score:
            best_score = true_max
            best_h = current_h.copy()
            
    # If no valid h found, return a default
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    # Final polish: Local search with small perturbations
    if time.time() < deadline - 2.0:
        rng_local = np.random.default_rng(123)
        h_best = best_h.copy()
        best_val = _max_overlap(h_best)
        
        for _ in range(50):
            if time.time() > deadline - 2.0: break
            # Random small perturbation
            idx = rng_local.choice(N, size=2, replace=False)
            delta = rng_local.uniform(-0.01, 0.01)
            h_try = h_best.copy()
            h_try[idx[0]] += delta
            h_try[idx[1]] -= delta
            
            # Project to [0,1] and sum 50
            h_try = np.clip(h_try, 0, 1)
            diff = 50 - np.sum(h_try)
            if abs(diff) > 1e-9:
                # Simple correction
                mask = h_try < 1.0 if diff > 0 else h_try > 0.0
                if np.any(mask):
                    h_try[mask] += diff / np.sum(mask)
                    h_try = np.clip(h_try, 0, 1)
                    
            val = _max_overlap(h_try)
            if val < best_val:
                best_val = val
                h_best = h_try.copy()
                
        best_h = h_best
        
    return best_h

def _max_overlap(h):
    c = _compute_overlaps(h)
    return np.max(c)
