import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 50
DEADLINE = 145.0  # 150s limit, 5s margin

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-9):
        return -1.0
    h = np.clip(h, 0.0, None)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -1.0
    l2_sq = np.sum(h * h)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv < 1e-15:
        return -1.0
    return l2_sq / (l1 * max_conv)

def smooth_max(x, beta=200.0):
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def objective_smooth(params, support, beta=200.0):
    """Smooth objective: minimize -log(ratio) for h supported on given indices."""
    w = np.exp(params)
    w = w / np.sum(w)
    h = np.zeros(N)
    h[support] = w
    l1 = np.sum(h)
    l2_sq = np.sum(h * h)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = smooth_max(h_conv, beta)
    if max_conv < 1e-15:
        return 1000.0
    ratio = l2_sq / (l1 * max_conv)
    if ratio <= 0:
        return 1000.0
    return -np.log(ratio)

def objective_exact(params, support):
    """Exact objective: minimize -log(ratio)."""
    w = np.exp(params)
    w = w / np.sum(w)
    h = np.zeros(N)
    h[support] = w
    r = compute_ratio(h)
    if r <= 0:
        return 1000.0
    return -np.log(r)

def optimize_weights(support, x0=None, time_budget=3.0):
    """Optimize weights on a fixed support. Returns (best_h, best_val)."""
    k = len(support)
    if k == 0:
        return None, -1.0
    if x0 is None:
        x0 = np.zeros(k)  # uniform weights in log-space
    
    best_val = -1.0
    best_w = None
    
    # Phase 1: Smooth surrogate
    try:
        res = minimize(objective_smooth, x0, args=(support, 200.0),
                       method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.x is not None:
            w = np.exp(res.x)
            w = w / np.sum(w)
            h = np.zeros(N)
            h[support] = w
            val = compute_ratio(h)
            if val > best_val:
                best_val = val
                best_w = w.copy()
    except Exception:
        pass
    
    if time.time() > DEADLINE - 10:
        if best_w is not None:
            h = np.zeros(N)
            h[support] = best_w
            return h, compute_ratio(h)
        return None, -1.0
    
    # Phase 2: Exact polish
    try:
        x0_exact = np.log(best_w + 1e-12) if best_w is not None else x0
        res2 = minimize(objective_exact, x0_exact, args=(support,),
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
        if res2.x is not None:
            w = np.exp(res2.x)
            w = w / np.sum(w)
            h = np.zeros(N)
            h[support] = w
            val = compute_ratio(h)
            if val > best_val:
                best_val = val
                best_w = w.copy()
    except Exception:
        pass
    
    if best_w is None:
        return None, -1.0
    
    h = np.zeros(N)
    h[support] = best_w
    return h, compute_ratio(h)

def is_sidon(arr):
    """Check if a set of integers is a Sidon set (B2 set)."""
    arr = sorted(set(arr))
    n = len(arr)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                return False
            sums[s] = (i, j)
    return True

def generate_supports():
    """Generate diverse support patterns to try."""
    supports = []
    
    # 1. Uniform (all indices)
    supports.append(list(range(N)))
    
    # 2. Symmetric patterns
    # Try various symmetric subsets
    for half_size in range(1, N // 2 + 1):
        # Symmetric about center
        center = (N - 1) // 2
        idx = set()
        for d in range(half_size):
            idx.add(center - d)
            idx.add(center + d)
            if center - d == center + d:
                pass
        idx = sorted([i for i in idx if 0 <= i < N])
        if len(idx) >= 2:
            supports.append(idx)
    
    # 3. Sidon sets of various sizes
    for target_size in [3, 4, 5, 6, 7, 8, 9, 10]:
        sidon = []
        for i in range(N):
            candidate = sidon + [i]
            if is_sidon(candidate):
                sidon.append(i)
                if len(sidon) == target_size:
                    break
        if len(sidon) >= 2:
            supports.append(sidon)
    
    # 4. Random sparse supports
    np.random.seed(42)
    for k in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40]:
        for trial in range(3):
            idx = sorted(np.random.choice(N, k, replace=False).tolist())
            supports.append(idx)
    
    # 5. Structured: arithmetic progressions
    for start in range(0, N, 5):
        for step in [1, 2, 3, 4, 5]:
            idx = list(range(start, N, step))
            if len(idx) >= 2:
                supports.append(idx)
    
    # 6. Alternating patterns
    supports.append(list(range(0, N, 2)))
    supports.append(list(range(1, N, 2)))
    supports.append(list(range(0, N, 3)))
    supports.append(list(range(1, N, 3)))
    supports.append(list(range(2, N, 3)))
    
    # 7. Block patterns
    for block_size in [2, 3, 4, 5]:
        for offset in range(block_size):
            idx = []
            for i in range(0, N, block_size * 2):
                for j in range(block_size):
                    val = i + j + offset
                    if val < N:
                        idx.append(val)
            if len(idx) >= 2:
                supports.append(idx)
    
    # 8. Symmetric Sidon-like: pick indices that are "far apart"
    # Greedy: maximize minimum distance
    for target_size in [3, 4, 5, 6, 7, 8]:
        idx = [0, N - 1]
        while len(idx) < target_size:
            best_pos = -1
            best_min_dist = -1
            for pos in range(N):
                if pos in idx:
                    continue
                min_dist = min(abs(pos - i) for i in idx)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_pos = pos
            if best_pos >= 0:
                idx.append(best_pos)
            else:
                break
        idx = sorted(idx)
        if len(idx) >= 2:
            supports.append(idx)
    
    # Remove duplicates
    unique_supports = []
    seen = set()
    for s in supports:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_supports.append(s)
    
    return unique_supports

def construct():
    start_time = time.time()
    
    best_h = np.ones(N) / N
    best_val = compute_ratio(best_h)
    
    # Generate candidate supports
    supports = generate_supports()
    
    # Sort by a heuristic: prefer smaller supports (more sparse) and Sidon-like
    # Actually, let's just try them all within time budget
    np.random.seed(123)
    
    # Shuffle to diversify
    np.random.shuffle(supports)
    
    # Always include some key patterns first
    priority = []
    # Symmetric small patterns
    for s in supports:
        if len(s) <= 8:
            priority.append(s)
    # Then the rest
    for s in supports:
        if s not in priority:
            priority.append(s)
    
    # Optimize each support
    for support in priority:
        if time.time() > DEADLINE - 15:
            break
        
        k = len(support)
        if k < 2:
            continue
        
        # Try multiple initializations
        n_inits = 3 if k <= 10 else 2
        for init_i in range(n_inits):
            if time.time() > DEADLINE - 15:
                break
            
            if init_i == 0:
                x0 = np.zeros(k)  # uniform
            elif init_i == 1:
                # Random log-weights
                x0 = np.random.randn(k) * 0.5
            else:
                # Geometric decay
                x0 = np.linspace(0, -1, k)
            
            h_opt, val_opt = optimize_weights(support, x0=x0)
            
            if h_opt is not None and val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
    
    # Final global polish on the best solution
    if time.time() < DEADLINE - 5:
        try:
            # Extract support from best_h
            tol = 1e-10
            nonzero = np.where(best_h > tol)[0]
            if len(nonzero) >= 2:
                support = list(nonzero)
                w = best_h[support]
                w = w / np.sum(w)
                x0 = np.log(w + 1e-12)
                
                # Multiple polish attempts
                for attempt in range(3):
                    if time.time() > DEADLINE - 3:
                        break
                    try:
                        res = minimize(objective_exact, x0, args=(support,),
                                       method='L-BFGS-B',
                                       options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13})
                        if res.x is not None:
                            w_new = np.exp(res.x)
                            w_new = w_new / np.sum(w_new)
                            h_new = np.zeros(N)
                            h_new[support] = w_new
                            val_new = compute_ratio(h_new)
                            if val_new > best_val:
                                best_val = val_new
                                best_h = h_new.copy()
                                x0 = res.x.copy()
                    except Exception:
                        break
        except Exception:
            pass
    
    # Ensure non-negative and valid
    best_h = np.clip(best_h, 0.0, None)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N) / N
    
    return list(best_h)
