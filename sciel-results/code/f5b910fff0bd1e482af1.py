import time
import numpy as np

N = 100
SCALE = 2.0 / N
TARGET_SUM = N / 2.0

def _compute_O(h):
    """
    Compute the overlap vector O for all shifts k=0..N-1.
    O_pos[k] = SCALE * sum_{j=k}^{N-1} h[j]*(1-h[j-k])
    O_neg[k] = SCALE * sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
    """
    # Efficient vectorized computation
    # O_pos[k] = SCALE * (sum_{j=k}^{N-1} h[j] - sum_{j=k}^{N-1} h[j]h[j-k])
    
    # Precompute suffix sums of h
    h_rev = h[::-1]
    suffix_sums = np.cumsum(h_rev)[::-1] # suffix_sums[i] = sum h[i:]
    
    # Compute sum of products h[j]*h[j-k] for each k
    # This is a cross-correlation-like operation but only for the valid range
    # For a fixed k, we need sum_{j=k}^{N-1} h[j]*h[j-k]
    # Let's use a loop for this part as it's O(N^2) total but with small constants in NumPy if vectorized per k?
    # Actually, N=100 is small. A simple loop is fine and clear.
    
    S_prod_pos = np.zeros(N)
    S_prod_neg = np.zeros(N)
    
    for k in range(N):
        if k == 0:
            S_prod_pos[0] = np.dot(h, h)
            S_prod_neg[0] = np.dot(h, h)
        else:
            S_prod_pos[k] = np.dot(h[k:], h[:-k])
            S_prod_neg[k] = np.dot(h[:N-k], h[k:])
            
    O_pos = SCALE * (suffix_sums - S_prod_pos)
    
    prefix_sums = np.cumsum(h)
    prefix_sums_shifted = np.zeros(N)
    prefix_sums_shifted[0] = 0.0
    prefix_sums_shifted[1:] = prefix_sums[:-1]
    # O_neg[k] = SCALE * (sum_{j=0}^{N-1-k} h[j] - sum_{j=0}^{N-1-k} h[j]h[j+k])
    # sum_{j=0}^{N-1-k} h[j] is prefix_sums[N-1-k]
    # prefix_sums is 1-indexed effectively in the array, prefix_sums[i] = sum h[0..i]
    # So sum h[0..N-1-k] is prefix_sums[N-1-k]
    
    O_neg = SCALE * (prefix_sums[N-1 - np.arange(N)] - S_prod_neg)
    
    return np.concatenate([O_pos, O_neg])

def _soft_max_grad(h, tau):
    """
    Compute value and gradient of the soft-max of O.
    """
    O = _compute_O(h)
    Omax = O.max()
    z = tau * (O - Omax)
    ez = np.exp(z)
    w = ez / ez.sum()
    
    # Gradient of soft-max: w * (O - Omax) * tau? No.
    # f(h) = (1/tau) log(sum exp(tau * O_k(h)))
    # grad f = sum w_k * grad O_k
    # We need grad O_k w.r.t h.
    
    # Let's compute the Jacobian J where J[k, i] = d O_k / d h_i
    # O_pos[k] = SCALE * (sum_{j=k}^{N-1} h[j] - sum_{j=k}^{N-1} h[j]h[j-k])
    # d O_pos[k] / d h[i]:
    # 1. From sum h[j]: non-zero if i in [k, N-1], value SCALE
    # 2. From -sum h[j]h[j-k]:
    #    Term h[j]h[j-k] involves h[i] if:
    #    a) j = i => i in [k, N-1] => term -SCALE * h[i-k]
    #    b) j-k = i => j = i+k => i+k in [k, N-1] => i in [0, N-1-k] => term -SCALE * h[i+k]
    
    # So d O_pos[k] / d h[i] = SCALE * (1_{i>=k} - 1_{i>=k}h[i-k] - 1_{i<N-k}h[i+k])
    
    # O_neg[k] = SCALE * (sum_{j=0}^{N-1-k} h[j] - sum_{j=0}^{N-1-k} h[j]h[j+k])
    # d O_neg[k] / d h[i]:
    # 1. From sum h[j]: non-zero if i in [0, N-1-k], value SCALE
    # 2. From -sum h[j]h[j+k]:
    #    a) j = i => i in [0, N-1-k] => term -SCALE * h[i+k]
    #    b) j+k = i => j = i-k => i-k in [0, N-1-k] => i in [k, N-1] => term -SCALE * h[i-k]
    
    # So d O_neg[k] / d h[i] = SCALE * (1_{i<N-k} - 1_{i<N-k}h[i+k] - 1_{i>=k}h[i-k])
    
    # Instead of building the full Jacobian (2N x N), we can compute the weighted sum directly.
    # grad = sum_k w_k * grad O_k
    
    # Let's accumulate the gradient.
    grad = np.zeros(N)
    
    # Vectorized accumulation might be tricky due to different index ranges.
    # Given N=100, a loop over k is acceptable if the inner operations are vectorized.
    
    for k in range(N):
        w_k = w[k]
        if w_k < 1e-12:
            continue
            
        # Contribution from O_pos[k]
        # d O_pos[k] / d h[i] = SCALE * (1_{i>=k} - 1_{i>=k}h[i-k] - 1_{i<N-k}h[i+k])
        
        # Part 1: i in [k, N-1]
        idx1 = np.arange(k, N)
        if len(idx1) > 0:
            grad[idx1] += w_k * SCALE * (1.0 - h[idx1 - k])
            
        # Part 2: i in [0, N-1-k]
        idx2 = np.arange(0, N - k)
        if len(idx2) > 0:
            grad[idx2] -= w_k * SCALE * h[idx2 + k]
            
        # Contribution from O_neg[k]
        # d O_neg[k] / d h[i] = SCALE * (1_{i<N-k} - 1_{i<N-k}h[i+k] - 1_{i>=k}h[i-k])
        
        # Part 1: i in [0, N-1-k]
        if len(idx2) > 0:
            grad[idx2] += w_k * SCALE * (1.0 - h[idx2 + k])
            
        # Part 2: i in [k, N-1]
        if len(idx1) > 0:
            grad[idx1] -= w_k * SCALE * h[idx1 - k]
            
    # The gradient of the soft-max objective is:
    # grad_obj = sum_k w_k * grad O_k
    # Note: w_k = exp(tau*(O_k - Omax)) / sum exp(...)
    # This is already computed in the loop above as 'grad'.
    
    # Value of soft-max:
    val = Omax + np.log(ez.sum()) / tau
    
    return val, grad

def _project(z):
    """
    Project z onto {h in [0,1], sum h = TARGET_SUM}
    """
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = TARGET_SUM
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)

def _fista(init, tau, deadline, max_iter=300):
    h = _project(np.asarray(init, dtype=float).copy())
    h_prev = h.copy()
    t = 1.0
    L = 5.0 # Initial Lipschitz estimate
    
    # Adaptive L
    best_h = h.copy()
    best_val = float(_compute_O(h).max())
    
    for k in range(max_iter):
        if time.time() > deadline - 2.0:
            return best_h, best_val
            
        # FISTA momentum
        y = h + (t - 1) / (t + 1) * (h - h_prev)
        
        # Gradient at y
        val_y, grad_y = _soft_max_grad(y, tau)
        
        # Step
        h_new = _project(y - 1.0 / L * grad_y)
        
        # Check convergence / improvement
        val_new = float(_compute_O(h_new).max())
        
        # Simple backtracking if it doesn't improve?
        # For soft-max, if L is too small, the step might be too big.
        # If val_new > val_y + 0.5 * (1/L) * ||grad_y||^2, increase L.
        # But computing norm is extra. Let's just trust L for now or use a fixed larger L.
        # Given N=100, the Hessian is bounded. L=10 should be safe.
        
        if val_new < best_val:
            best_h = h_new.copy()
            best_val = val_new
            
        h_prev = h
        h = h_new
        t = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        
        # Optional: if no improvement for a while, restart or increase L?
        
    return best_h, best_val

def _local_search(h, deadline, max_iters=20000):
    """
    Local search by swapping pairs or adjusting single elements (with compensation).
    """
    h = _project(h.copy())
    best_h = h.copy()
    best_val = float(_compute_O(h).max())
    
    rng = np.random.default_rng(999)
    
    for _ in range(max_iters):
        if time.time() > deadline - 2.0:
            break
            
        # Pick a random move
        # 1. Swap two elements
        # 2. Adjust one element up, another down (keeping sum)
        
        move_type = rng.integers(0, 2)
        
        if move_type == 0:
            i, j = rng.choice(N, 2, replace=False)
            h_new = h.copy()
            h_new[i], h_new[j] = h[j], h[i]
        else:
            # Move delta from i to j
            i, j = rng.choice(N, 2, replace=False)
            # Ensure feasibility
            # h[i] + delta <= 1 => delta <= 1 - h[i]
            # h[j] - delta >= 0 => delta <= h[j]
            # h[i] - delta >= 0 => delta >= -h[i]
            # h[j] + delta <= 1 => delta >= h[j] - 1
            
            delta_max = min(1.0 - h[i], h[j])
            delta_min = max(h[j] - 1.0, -h[i])
            
            if delta_max <= delta_min:
                continue
                
            # Pick a random delta in [delta_min, delta_max]
            # Or try a fixed small delta? Random is better for exploration.
            delta = rng.uniform(delta_min, delta_max)
            
            h_new = h.copy()
            h_new[i] += delta
            h_new[j] -= delta
            
        val_new = float(_compute_O(h_new).max())
        if val_new < best_val - 1e-12:
            best_h = h_new
            best_val = val_new
            h = h_new
        elif rng.random() < 0.05: # Simulated annealing style?
            # Accept worse moves occasionally?
            # For now, stick to hill climbing.
            pass
            
    return best_h, best_val

def construct():
    start = time.time()
    deadline = start + 100.0
    
    best_h = np.full(N, 0.5)
    best_val = float(_compute_O(best_h).max())
    
    idx = np.arange(N)
    inits = []
    
    # 1. Random bang-bang (Reduced count)
    rng = np.random.default_rng(42)
    for _ in range(5):
        bb = np.zeros(N)
        bb[rng.choice(N, N // 2, replace=False)] = 1.0
        inits.append(bb)
        
    # 2. Structured continuous (Reduced count)
    for f in (1, 2, 3):
        inits.append(0.5 + 0.4 * np.cos(2.0 * np.pi * f * idx / N))
        
    # 3. Random continuous (Reduced count)
    for _ in range(5):
        inits.append(rng.random(N))
        
    # 4. Periodic bang-bang (Reduced count)
    for period in [3, 5, 7]:
        bb = np.zeros(N)
        ones = period // 2
        pattern = np.zeros(period)
        pattern[:ones] = 1.0
        bb = np.tile(pattern, N // period + 1)[:N]
        # Adjust sum
        if bb.sum() > TARGET_SUM:
             idxs = np.where(bb==1)[0]
             to_remove = rng.choice(idxs, int(bb.sum() - TARGET_SUM), replace=False)
             bb[to_remove] = 0
        elif bb.sum() < TARGET_SUM:
             idxs = np.where(bb==0)[0]
             to_add = rng.choice(idxs, int(TARGET_SUM - bb.sum()), replace=False)
             bb[to_add] = 1
        inits.append(bb)

    # Optimization loop
    # Reduced number of restarts to fit within time budget
    # We keep the best 10 initializers
    inits = inits[:10]
    
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        try:
            h, val = _fista(init, tau=50.0, deadline=deadline)
            if np.all(np.isfinite(h)) and val < best_val:
                best_h = h
                best_val = val
        except Exception:
            continue

    # Removed local search to ensure we finish in time.
    # FISTA with soft-max surrogate is typically effective enough.

    # Ensure exact sum and bounds
    best_h = _project(best_h)
    for _ in range(10):
        d = TARGET_SUM - best_h.sum()
        if abs(d) < 1e-12:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
        
    return best_h
