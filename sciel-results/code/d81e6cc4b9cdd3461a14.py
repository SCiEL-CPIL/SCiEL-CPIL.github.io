import numpy as np
import time
import random

N = 50

def _ratio(h):
    """Raw objective: L2^2 / (L1 * max_autoconv)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        return -np.inf
    h = np.clip(h, 0.0, None)
    l1 = np.sum(h)
    if l1 <= 0.0:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 0.0:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0.0:
        return 0.0
    return l2_sq / (l1 * m)

def _optimize_epigraph(h_init, max_iter=500, ftol=1e-8):
    """
    Optimize h to minimize the peak of the autoconvolution, 
    subject to sum(h)=1 and h>=0.
    Variables: x = [h_0, ..., h_{N-1}, M]
    Objective: Minimize M
    Constraints:
      1. sum(h) - 1 = 0
      2. M - (h*h)_k >= 0 for all k
    """
    from scipy.optimize import minimize

    def obj(x):
        return x[N]

    def obj_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad

    def constraint_sum(x):
        return np.sum(x[:N]) - 1.0

    def constraint_sum_grad(x):
        grad = np.zeros(N + 1)
        grad[:N] = 1.0
        return grad

    def constraints_autoconv(x):
        h = x[:N]
        M = x[N]
        h_clip = np.clip(h, 0, None)
        c = np.convolve(h_clip, h_clip, mode="full")
        return M - c

    def constraints_autoconv_grad(x):
        h = x[:N]
        grad = np.zeros((2 * N - 1, N + 1))
        for k in range(2 * N - 1):
            derivs = np.zeros(N)
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    derivs[i] = 2.0 * h[j]
            grad[k, :N] = -derivs
            grad[k, N] = 1.0
        return grad

    h_init = np.asarray(h_init, dtype=np.float64)
    s = np.sum(h_init)
    if s <= 0:
        h_init = np.ones(N) / N
        s = 1.0
    h_init = h_init / s
    
    c_init = np.convolve(h_init, h_init, mode="full")
    M_init = np.max(c_init)
    x0 = np.concatenate([h_init, [M_init]])
    
    bounds = [(0.0, None)] * N + [(0.0, None)]
    
    cons = [
        {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
        {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
    ]
    
    try:
        res = minimize(
            obj, 
            x0, 
            method='SLSQP', 
            jac=obj_grad,
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': ftol}
        )
        
        if res.success or res.fun < M_init:
            h_res = res.x[:N]
            h_res = np.clip(h_res, 0, None)
            s = np.sum(h_res)
            if s > 0:
                h_res = h_res / s
            else:
                return None, 0.0
                
            ratio_res = _ratio(h_res)
            return h_res, ratio_res
    except Exception:
        pass
    return None, 0.0

def _find_sidon_set(N, K, attempts=1000):
    """
    Attempts to find a Sidon set of size K in [0, N).
    Returns the set if found, None otherwise.
    """
    for _ in range(attempts):
        candidates = list(range(N))
        random.shuffle(candidates)
        
        S = []
        sums = set()
        
        for cand in candidates:
            if len(S) == K:
                break
            
            conflict = False
            new_sums = set()
            
            if cand + cand in sums:
                conflict = True
            else:
                new_sums.add(cand + cand)
                
            if not conflict:
                for s in S:
                    sm = cand + s
                    if sm in sums:
                        conflict = True
                        break
                    if sm in new_sums:
                        conflict = True
                        break
                    new_sums.add(sm)
            
            if not conflict:
                S.append(cand)
                sums.update(new_sums)
        
        if len(S) == K:
            return S
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N) / N
    best_ratio = _ratio(best_h)
    
    # 1. Search for large Sidon sets to get good structured starts
    # Theoretical max K is around sqrt(N) * 1.5 or so.
    # We try from 12 down to 5.
    found_sidon_supports = []
    for K in range(12, 4, -1):
        if time.time() > deadline - 20.0:
            break
            
        attempts = 500 if K > 8 else 1000
        S = _find_sidon_set(N, K, attempts)
        
        if S is not None:
            h = np.zeros(N)
            h[S] = 1.0 / K
            ratio = _ratio(h)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h.copy()
            found_sidon_supports.append((K, S, ratio))
            # Since larger K gives higher ratio for Sidon sets (ratio=K),
            # and we are iterating downwards, the first one found is likely the best structured start.
            # But we keep all to use as multiple starts for optimization.
            
    # 2. Optimize using SLSQP epigraph formulation
    # We use the Sidon sets found, plus uniform, plus some random starts.
    
    # Start with the best Sidon set found (largest K)
    if found_sidon_supports:
        # Sort by K descending
        found_sidon_supports.sort(key=lambda x: x[0], reverse=True)
        
        for K, S, _ in found_sidon_supports[:3]: # Try top 3 largest
            if time.time() > deadline - 15.0:
                break
            
            h_init = np.zeros(N)
            h_init[S] = 1.0 / K
            
            h_opt, r_opt = _optimize_epigraph(h_init, max_iter=300)
            if h_opt is not None and r_opt > best_ratio:
                best_ratio = r_opt
                best_h = h_opt.copy()
                
    # Also try uniform start
    if time.time() > deadline - 15.0:
        pass
    else:
        h_opt, r_opt = _optimize_epigraph(np.ones(N)/N, max_iter=300)
        if h_opt is not None and r_opt > best_ratio:
            best_ratio = r_opt
            best_h = h_opt.copy()

    # 3. Random restarts with sparse supports
    # This helps escape local optima.
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        K = random.randint(5, 15)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        
        h_opt, r_opt = _optimize_epigraph(h_rand, max_iter=200)
        if h_opt is not None and r_opt > best_ratio:
            best_ratio = r_opt
            best_h = h_opt.copy()

    # 4. Polish the best solution
    if best_h is not None and time.time() > deadline - 20.0:
        h_polish_init = best_h.copy()
        h_polish_init[h_polish_init < 1e-6] = 0.0
        s = np.sum(h_polish_init)
        if s > 0:
            h_polish_init /= s
        
        h_polished, r_polished = _optimize_epigraph(h_polish_init, max_iter=1000, ftol=1e-10)
        if h_polished is not None and r_polished > best_ratio:
            best_ratio = r_polished
            best_h = h_polished.copy()

    return list(best_h)
