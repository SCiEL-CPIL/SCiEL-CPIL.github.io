import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _generate_sidon_set(n, k, rng):
    """
    Greedy generation of a Sidon-like set of size k from {0, ..., n-1}.
    A Sidon set has distinct pairwise sums.
    """
    S = []
    sums = set()
    # Start with 0
    S.append(0)
    # Try to add elements
    for i in range(1, n):
        if len(S) >= k:
            break
        # Check if adding i creates duplicate sums
        valid = True
        new_sums = []
        for s in S:
            val = s + i
            if val in sums:
                valid = False
                break
            new_sums.append(val)
        # Also check sums within the new pair itself? 
        # Sidon condition: a+b = c+d implies {a,b}={c,d}
        # We just need to ensure no new sum equals an existing sum, 
        # and no new sum equals a sum of two existing elements that isn't just itself?
        # Actually, if we build incrementally, we just need to ensure 
        # s_i + s_j (where j < i) are all distinct from each other and from previous sums.
        
        if valid:
            # Check sums between new element and all existing elements
            for s in S:
                val = s + i
                if val in sums:
                    valid = False
                    break
            if valid:
                # Add new sums to the set
                for s in S:
                    sums.add(s + i)
                S.append(i)
                
    return S

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Try single spike
    for i in range(N):
        h_spike = np.zeros(N)
        h_spike[i] = 1.0
        s = _score_raw(h_spike)
        if s > best_score:
            best_score = s
            best_h = h_spike.copy()
            
    # 2. Try pairs of spikes
    for i in range(N):
        for j in range(i+1, N):
            h_pair = np.zeros(N)
            h_pair[i] = 1.0
            h_pair[j] = 1.0
            s = _score_raw(h_pair)
            if s > best_score:
                best_score = s
                best_h = h_pair.copy()

    # 3. Parametric search for good initializations
    # Gaussian
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # Triangular
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()

    # 4. Sidon-like sparse initializations
    # These have low autoconvolution peaks relative to their L2 norm
    for k in range(5, 15):
        for _ in range(5): # Multiple random restarts for Sidon generation
            if time.time() > deadline: break
            rng = np.random.default_rng()
            sidon_idx = _generate_sidon_set(N, k, rng)
            if len(sidon_idx) < 2:
                continue
            h_sidon = np.zeros(N)
            h_sidon[sidon_idx] = 1.0
            # Add small noise to avoid exact symmetries or numerical issues
            h_sidon += np.random.rand(N) * 0.01
            s = _score_raw(h_sidon)
            if s > best_score:
                best_score = s
                best_h = h_sidon.copy()

    # 5. Random sparse initialization
    for _ in range(100):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        h_rand += np.random.rand(N) * 0.1
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # Optimization using SLSQP
    # Objective: Maximize Score(h) = sum(h^2) / (sum(h) * max(conv(h,h)))
    # With constraint sum(h) = 1, this is sum(h^2) / max(conv(h,h))
    
    def neg_obj_h(h):
        l1 = np.sum(h)
        if l1 == 0: return 0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0: return 0
        return -(l2_sq / max_conv)
        
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h} # h >= 0
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    # Start from best_h (normalized)
    h_start = best_h / np.sum(best_h)
    
    # Optimization loop with multiple restarts
    inits = []
    
    # 1. Best found so far
    inits.append(h_start)
    
    # 2. Uniform
    inits.append(np.ones(N) / N)
    
    # 3. Gaussians
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N/2, 5):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    # 4. Triangles
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N, 5):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 5. Sparse (Sidon-like) - Reuse some from above
    for k in range(5, 10):
        sidon_idx = _generate_sidon_set(N, k, np.random.default_rng())
        if len(sidon_idx) >= 2:
            h_s = np.zeros(N)
            h_s[sidon_idx] = 1.0
            h_s /= np.sum(h_s)
            inits.append(h_s)
        
    # 6. Random smooth
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)

    # Optimize from each init
    for h_init in inits:
        if time.time() > deadline: break
        
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        
        try:
            res = minimize(neg_obj_h, h_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                h_candidate = res.x
                # Ensure non-negativity
                h_candidate = np.maximum(h_candidate, 0)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
            
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                h_candidate = res.x
                h_candidate = np.maximum(h_candidate, 0)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)
