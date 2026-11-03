import numpy as np
import time
import sys

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    # Continuous autoconvolution of step function f(t) = sum h_i 1_{[x_i, x_{i+1})}(t)
    # The peak is proportional to the max of the discrete convolution of h with itself.
    # Specifically, max(f*f) = DELTA * max(conv(h,h))
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # 1. Generate diverse initializations
    inits = []
    x_idx = np.arange(N)
    
    # Uniform
    inits.append(np.ones(N) / N)
    
    # Gaussians
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(0.5, N/2, 10):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 0:
                h_g /= np.sum(h_g)
                inits.append(h_g)
                
    # Triangles
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(1, N, 10):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # Sparse random
    for _ in range(50):
        h_s = np.zeros(N)
        k = np.random.randint(3, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # Random smooth
    for _ in range(50):
        h_r = np.random.rand(N)
        kernel = np.ones(3)/3
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)
        
    # Power laws
    for p in np.linspace(0.5, 3.0, 10):
        h_p = (x_idx + 1) ** (-p)
        h_p /= np.sum(h_p)
        inits.append(h_p)
        
    # Alternating
    for f in np.linspace(0.1, 1.0, 10):
        h_a = 0.5 + 0.5 * np.cos(2 * np.pi * f * x_idx)
        h_a /= np.sum(h_a)
        inits.append(h_a)

    # 2. Coordinate Descent with adaptive step size
    # Objective: Maximize S(h) = sum(h^2) / max(conv(h,h)) subject to sum(h)=1, h>=0
    # We work in the simplex.
    
    def get_score(h):
        return _score_raw(h)
    
    def optimize_cd(h_init, max_iters=500):
        h = h_init.copy()
        h = np.maximum(h, 0)
        h /= np.sum(h)
        
        current_score = get_score(h)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            improved = False
            # Try moving mass from j to i
            # To speed up, we can try a few random pairs or all pairs
            # With N=50, all pairs is 2500.
            # Let's try random pairs first to find promising directions, then refine?
            # Or just all pairs since it's fast enough?
            # Convolution is O(N^2) = 2500 ops. 2500 pairs * 2500 ops = 6.25M ops per iter.
            # 500 iters = 3.125B ops. Too slow.
            
            # Strategy:
            # 1. Compute conv(h,h) once.
            # 2. Identify the index k where max occurs.
            # 3. The objective is f = sum(h^2) / conv_max.
            #    Increasing h_i increases numerator by 2h_i + eps^2.
            #    Decreasing h_j decreases numerator by 2h_j - eps^2.
            #    Change in conv_max depends on whether the max shifts.
            
            # Simpler: Coordinate ascent on individual h_i?
            # Fix sum(h) = 1. If I increase h_i by eps, I must decrease some other h_j by eps.
            
            # Let's try a heuristic:
            # For each i, try to increase h_i slightly and decrease the h_j that causes the least damage to the denominator.
            # The denominator is max_k sum_l h_l h_{k-l}.
            # The term involving h_i in the convolution is sum_l h_l h_{k-l} where l=i or k-l=i.
            # i.e., h_i * sum_{l != i} h_{k-l} + h_i^2 (if k=2i).
            # This is complex.
            
            # Alternative: Nelder-Mead is derivative-free and often robust.
            # But CD is better for structured problems.
            
            # Let's stick to a simpler CD:
            # Randomly pick i, j.
            # Try moving mass from j to i.
            
            if np.random.rand() < 0.5:
                i = np.random.randint(N)
                j = np.random.randint(N)
                if i == j: continue
                
                # Try moving small amounts
                base_h = h.copy()
                best_eps = 0
                best_local_score = current_score
                
                # Binary search for best eps in [0, h_j]
                lo, hi = 0.0, h[j]
                for _ in range(10):
                    mid = (lo + hi) / 2
                    h_test = base_h.copy()
                    h_test[i] += mid
                    h_test[j] -= mid
                    s_test = get_score(h_test)
                    if s_test > best_local_score:
                        best_local_score = s_test
                        best_eps = mid
                        lo = mid
                    else:
                        hi = mid
                
                if best_eps > 0:
                    h[i] += best_eps
                    h[j] -= best_eps
                    current_score = best_local_score
                    if current_score > best_score:
                        best_score = current_score
                        best_h = h.copy()
                    improved = True
            
            if not improved and it > 50:
                # Stuck, maybe break or random restart?
                # Let's just break and rely on multiple inits
                break
                
        return h, current_score

    # Run CD from each init
    for h_init in inits:
        if time.time() > deadline:
            break
        
        h_opt, s_opt = optimize_cd(h_init, max_iters=200)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # 3. Local Refinement with Nelder-Mead on the best solution
    if time.time() < deadline - 5:
        def neg_score(h):
            return -get_score(h)
        
        from scipy.optimize import minimize
        
        # NM requires unconstrained or bounds.
        # We can use a penalty method or transform variables.
        # Simple: just use NM on the vector, but project back to simplex?
        # NM doesn't support projection easily in scipy.
        # Let's use COBYLA which supports constraints.
        
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
            {'type': 'ineq', 'fun': lambda h: h}
        ]
        
        try:
            res = minimize(neg_score, best_h, method='COBYLA', constraints=constraints, options={'maxiter': 2000, 'rhobeg': 1e-4})
            if res.fun < -best_score:
                h_cand = res.x
                s_cand = get_score(h_cand)
                if s_cand > best_score:
                    best_score = s_cand
                    best_h = h_cand
        except Exception:
            pass

    return list(best_h)
