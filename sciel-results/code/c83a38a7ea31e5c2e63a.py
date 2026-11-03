import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _autoconv_max(h):
    """Compute max of autoconvolution (without DELTA scaling)."""
    h_conv = np.convolve(h, h, mode='full')
    return np.max(h_conv)

def _epigraph_objective(x):
    """
    Epigraph formulation: maximize sum(h^2) subject to sum(h)=1, h>=0, conv(h) <= t.
    x = [h_0, ..., h_{N-1}, t]
    We minimize -sum(h^2) with constraints.
    """
    h = x[:N]
    t = x[N]
    # Objective: minimize -sum(h^2)
    return -np.sum(h**2)

def _epigraph_constraints(x):
    """Return list of constraint dicts for scipy minimize."""
    h = x[:N]
    t = x[N]
    # Equality: sum(h) = 1
    # Inequality: h_i >= 0 for all i
    # Inequality: max(conv(h)) - t <= 0  =>  t - max(conv(h)) >= 0
    # For scipy, inequality constraints are g(x) >= 0
    # So we need: t - max(conv(h)) >= 0
    # But max(conv(h)) is non-differentiable. We use a smooth approximation or
    # handle it via the constraint function.
    
    # Actually, let's use a different approach: use SLSQP with the constraint
    # that conv(h) <= t, which means all entries of conv(h) <= t.
    # This gives 2N-1 inequality constraints.
    pass

def _block_coordinate_descent(h, t, max_iter=5000, tol=1e-10, time_limit=None, start_time=None):
    """
    Block-coordinate descent for the epigraph problem.
    Variables: h (length N) and t (scalar).
    Constraints: sum(h)=1, h>=0, conv(h) <= t.
    Objective: maximize sum(h^2).
    
    Strategy: alternate between updating pairs of h coordinates and t.
    """
    h = h.copy().astype(np.float64)
    t = float(t)
    
    # Normalize h to sum=1
    s = np.sum(h)
    if s > 0:
        h /= s
    else:
        h = np.ones(N) / N
    
    t = _autoconv_max(h)
    
    best_obj = np.sum(h**2)
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Update t: t = max(conv(h))
        t_new = _autoconv_max(h)
        
        # Block-coordinate updates on h
        # Try updating pairs (i, j) to maintain sum=1
        # For each pair, optimize h_i and h_j given others fixed
        improved = False
        
        # Randomly select a subset of pairs to update
        num_updates = min(50, N * (N-1) // 2)
        pairs = np.random.choice(N * (N-1) // 2, size=num_updates, replace=False)
        # Convert linear index to (i, j)
        # Linear index k corresponds to i = k // (N-1), j = i*(N-1) + k % (N-1)
        # But we need i < j. Let's use a different mapping.
        # Simpler: randomly pick pairs
        for _ in range(num_updates):
            i, j = np.random.choice(N, size=2, replace=False)
            if i == j:
                continue
            
            # Fix all other coordinates. Sum of others = 1 - h[i] - h[j]
            sum_others = 1.0 - h[i] - h[j]
            if sum_others < -1e-15:
                continue
            
            # We want to maximize h[i]^2 + h[j]^2 subject to h[i] + h[j] = 1 - sum_others, h[i], h[j] >= 0
            # And the constraint conv(h) <= t must hold.
            # The optimal unconstrained solution is to put all mass on one coordinate.
            # But we need to check the conv constraint.
            
            # Let's try a few candidate values for h[i]
            total_pair = 1.0 - sum_others
            if total_pair < 0:
                continue
            
            # Try h[i] = 0, h[i] = total_pair, and some intermediate values
            candidates = [0.0, total_pair]
            # Add some intermediate values
            for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                candidates.append(frac * total_pair)
            
            best_local_obj = -np.inf
            best_hi = h[i]
            best_hj = h[j]
            
            for hi in candidates:
                hj = total_pair - hi
                if hi < -1e-15 or hj < -1e-15:
                    continue
                hi = max(0, hi)
                hj = max(0, hj)
                
                # Temporarily set h[i] and h[j]
                old_hi, old_hj = h[i], h[j]
                h[i] = hi
                h[j] = hj
                
                # Check conv constraint
                conv_max = _autoconv_max(h)
                if conv_max <= t + 1e-12:
                    obj = np.sum(h**2)
                    if obj > best_local_obj:
                        best_local_obj = obj
                        best_hi = hi
                        best_hj = hj
                
                # Restore
                h[i] = old_hi
                h[j] = old_hj
            
            if best_local_obj > np.sum(h**2) + 1e-15:
                h[i] = best_hi
                h[j] = best_hj
                improved = True
        
        # After updating h, update t
        t = _autoconv_max(h)
        
        # Check convergence
        current_obj = np.sum(h**2)
        if abs(current_obj - best_obj) < tol:
            if not improved:
                break
        best_obj = max(best_obj, current_obj)
    
    return h, t

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Uniform
    h_init = np.ones(N)
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # 2. Gaussian
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # 3. Triangular
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()
    
    # 4. Sparse peaks
    for _ in range(200):
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

    # Epigraph SLSQP optimization
    # x = [h_0, ..., h_{N-1}, t]
    # Objective: minimize -sum(h^2)
    # Constraints:
    #   sum(h) - 1 = 0
    #   h_i >= 0 for all i
    #   t - conv_k(h) >= 0 for all k (2N-1 constraints)
    
    def _neg_sum_sq(x):
        h = x[:N]
        return -np.sum(h**2)
    
    def _eq_sum_h(x):
        return np.sum(x[:N]) - 1.0
    
    def _ineq_h(x):
        return x[:N]
    
    def _ineq_t_conv(x):
        h = x[:N]
        t = x[N]
        conv = np.convolve(h, h, mode='full')
        return t - conv  # Each element must be >= 0
    
    constraints = [
        {'type': 'eq', 'fun': _eq_sum_h},
        {'type': 'ineq', 'fun': _ineq_h},
        {'type': 'ineq', 'fun': _ineq_t_conv}
    ]
    
    # Initial point
    h0 = best_h / np.sum(best_h)
    t0 = _autoconv_max(h0)
    x0 = np.concatenate([h0, [t0]])
    
    bounds = [(0.0, 1.0)] * N + [(0.0, 10.0)]
    
    # Run SLSQP on epigraph formulation
    try:
        res = minimize(_neg_sum_sq, x0, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
        if res.success:
            h_candidate = res.x[:N]
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate
    except Exception:
        pass
    
    # Block-coordinate descent from best_h
    if time.time() < deadline - 5:
        h_bc, t_bc = _block_coordinate_descent(best_h, _autoconv_max(best_h), 
                                               max_iter=10000, time_limit=30, start_time=time.time())
        s_bc = _score_raw(h_bc)
        if s_bc > best_score:
            best_score = s_bc
            best_h = h_bc
    
    # Multi-start with epigraph SLSQP
    inits = [best_h / np.sum(best_h)]
    inits.append(np.ones(N) / N)
    
    for _ in range(30):
        if time.time() > deadline - 10: break
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    for h_init in inits:
        if time.time() > deadline - 5: break
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        t_init = _autoconv_max(h_init)
        x_init = np.concatenate([h_init, [t_init]])
        try:
            res = minimize(_neg_sum_sq, x_init, method='SLSQP', bounds=bounds, 
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                h_candidate = res.x[:N]
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
    
    # Final block-coordinate polish
    if time.time() < deadline - 5:
        h_bc, t_bc = _block_coordinate_descent(best_h, _autoconv_max(best_h), 
                                               max_iter=10000, time_limit=30, start_time=time.time())
        s_bc = _score_raw(h_bc)
        if s_bc > best_score:
            best_score = s_bc
            best_h = h_bc

    return list(best_h)
