"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines linear overlap definition (FFT-accelerated) with analytical gradient optimization.
Structural change: uses vectorized gradient and epigraph formulation for SLSQP."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    # r_k = sum_{i=k}^{n-1} h[i] * h[i-k] for k=0..n-1
    # This is the autocorrelation. Use FFT for O(n log n) instead of O(n^2).
    # Zero-pad to avoid circular convolution issues
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    # Autocorrelation: ifft(|H|^2) gives circular autocorrelation of padded signal
    # For linear autocorrelation of length-n signal, we need:
    # r[k] = sum_{i=0}^{n-1-k} h[i+k] * h[i] for k=0..n-1
    # Using FFT: correlate h with h
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    # corr[k] for k=0..n-1 gives r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max(h, beta):
    """
    Smooth maximum of the linear overlap values using log-sum-exp.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    # Avoid overflow in exp
    return m + np.log(np.sum(np.exp(beta * (cs - m)))) / beta


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    
    # Vectorized gradient computation
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        
        # Term 1: h_{j-k} for j >= k
        if k > 0:
            grad[k:] += factor * h[:n-k]
        # Term 2: h_{j+k} for j+k < n
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    # Distribute the difference evenly, but clip
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # If clipping changed the sum, do a second pass
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction on one element to ensure exact sum in floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            # Need to add; find element with most room (closest to 1 but not at 1)
            idx = np.argmax(h)
        else:
            # Need to subtract; find element with most room (closest to 0 but not at 0)
            idx = np.argmin(h)
        # Apply correction without re-clipping to preserve exact sum
        h[idx] += diff
        # If somehow out of bounds (shouldn't happen if we chose correctly), clip
        if h[idx] < 0:
            h[idx] = 0.0
        elif h[idx] > 1:
            h[idx] = 1.0
            
    return h


def _coordinate_descent(h, max_sweeps=50, tol=1e-10):
    """
    Coordinate descent to minimize the true max overlap.
    For each coordinate j, try to find the best value of h[j] given all others fixed.
    C_k = 1 - (2/N)*r_k where r_k = sum_{i=k}^{N-1} h[i]*h[i-k]
    Changing h[j] affects r_k for all k where j>=k (term h[j]*h[j-k]) and j+k<N (term h[j+k]*h[j]).
    """
    h = np.array(h, dtype=np.float64, copy=True)
    n = len(h)
    best_score = _score(h)
    
    for sweep in range(max_sweeps):
        improved = False
        for j in range(n):
            # Compute current C_k values
            cs = _linear_overlap_values(h)
            current_max = np.max(cs)
            
            # For coordinate j, r_k changes as:
            # r_k_new = r_k_old + (h_j_new - h_j_old) * (h_{j-k} if j>=k else 0) + (h_{j+k} if j+k<n else 0) * (h_j_new - h_j_old)
            # C_k_new = 1 - (2/N) * r_k_new
            # We want to minimize max_k C_k_new
            # C_k_new = C_k_old - (2/N)*(h_j_new - h_j_old)*(h_{j-k} + h_{j+k})
            # where h_{j-k} is 0 if j<k, h_{j+k} is 0 if j+k>=n
            
            old_val = h[j]
            
            # Coefficient for each k: delta_C_k = -(2/N) * (h_j_new - h_j_old) * (h_{j-k} + h_{j+k})
            # Let a_k = h_{j-k} + h_{j+k} (with boundary handling)
            a = np.zeros(n)
            for k in range(n):
                val = 0.0
                if j >= k:
                    val += h[j - k]
                if j + k < n:
                    val += h[j + k]
                a[k] = val
            
            # C_k_new = C_k_old - (2/N) * (x - old_val) * a[k]
            # We want to minimize max_k [C_k_old - (2/N)*(x - old_val)*a[k]]
            # subject to 0 <= x <= 1 and sum constraint.
            # The sum constraint means we need to adjust another coordinate, which complicates things.
            # For now, just try a few candidate values for x.
            
            # Try: old_val, 0, 1, and values that make the top few C_k equal
            candidates = [old_val, 0.0, 1.0]
            
            # Find the k with max C_k and try to reduce it
            top_k = np.argmax(cs)
            if a[top_k] > 0:
                # Increasing x reduces C_top_k
                candidates.append(min(1.0, old_val + 0.1))
                candidates.append(min(1.0, old_val + 0.01))
            elif a[top_k] < 0:
                # Decreasing x reduces C_top_k
                candidates.append(max(0.0, old_val - 0.1))
                candidates.append(max(0.0, old_val - 0.01))
            
            # Also try to equalize the top 2 C_k values
            top2 = np.argsort(cs)[-2:][::-1]
            if len(top2) == 2:
                k1, k2 = top2[0], top2[1]
                if a[k1] != a[k2]:
                    # C_k1 - C_k2 = (cs[k1]-cs[k2]) - (2/N)*(x-old_val)*(a[k1]-a[k2]) = 0
                    # x = old_val + (N/2)*(cs[k1]-cs[k2])/(a[k1]-a[k2])
                    x_eq = old_val + (n/2.0)*(cs[k1]-cs[k2])/(a[k1]-a[k2])
                    if 0 <= x_eq <= 1:
                        candidates.append(x_eq)
            
            best_x = old_val
            best_local_max = current_max
            
            for x in candidates:
                x = max(0.0, min(1.0, x))
                h[j] = x
                new_cs = _linear_overlap_values(h)
                new_max = np.max(new_cs)
                if new_max < best_local_max - 1e-12:
                    best_local_max = new_max
                    best_x = x
            
            h[j] = best_x
            
            # Fix sum after coordinate change
            diff = N/2.0 - np.sum(h)
            if abs(diff) > 1e-14:
                # Distribute to the coordinate with most room
                mask = (h > 1e-6) & (h < 1.0 - 1e-6)
                if np.any(mask):
                    h[mask] += diff / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
            
            if best_local_max < best_score - 1e-10:
                best_score = best_local_max
                improved = True
        
        if not improved:
            break
    
    return h, best_score


def _epigraph_optimize(h0, maxiter=200, ftol=1e-12):
    """
    Epigraph formulation: minimize t subject to C_k(h) <= t for all k,
    sum(h) = N/2, 0 <= h <= 1.
    """
    n = len(h0)
    x0 = np.concatenate([h0.copy(), [max(_score(h0), 0.5)]])
    
    def obj(x):
        return x[-1]
    
    def obj_grad(x):
        g = np.zeros(n+1)
        g[-1] = 1.0
        return g
    
    # Constraint: C_k(h) - t <= 0 for all k
    # This is expensive; use a smooth approximation or just the max
    def constraint_max(x):
        h = x[:n]
        t = x[-1]
        cs = _linear_overlap_values(h)
        return t - np.max(cs)
    
    def constraint_max_grad(x):
        h = x[:n]
        t = x[-1]
        cs = _linear_overlap_values(h)
        top_k = np.argmax(cs)
        # Gradient of (t - C_top_k) w.r.t. h and t
        g = np.zeros(n+1)
        g[-1] = 1.0
        # dC_k/dh_j = -(2/n)*(h_{j-k} + h_{j+k})
        for j in range(n):
            val = 0.0
            if j >= top_k:
                val += h[j - top_k]
            if j + top_k < n:
                val += h[j + top_k]
            g[j] = (2.0/n) * val
        return g
    
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:n]) - N/2.0,
         'jac': lambda x: np.concatenate([np.ones(n), [0.0]])},
        {'type': 'ineq', 'fun': constraint_max,
         'jac': constraint_max_grad}
    ]
    
    bounds = [(0.0, 1.0)] * n + [(0.0, 2.0)]
    
    try:
        res = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol})
        if res.success:
            return res.x[:n]
    except Exception:
        pass
    return None


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Multiple Analytical Starts
    i = np.arange(N)
    
    # Cosine wave (fundamental)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # Higher frequency cosine
    h_cos2 = 0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)
    h_cos2 = _fix_sum(h_cos2)
    _update_best(h_cos2)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _fix_sum(h_sin)
    _update_best(h_sin)
    
    # Triangular wave
    h_tri = 0.5 + 0.5 * np.abs(2.0 * (i / N - 0.5) * 2.0 - 1.0)
    h_tri = _fix_sum(h_tri)
    _update_best(h_tri)
    
    # 2. Optimization with Analytical Gradient
    def _run_optimization(h0, beta, maxiter, ftol):
        nonlocal best_h, best_score
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        def obj_func(x):
            val, _ = _smooth_max_grad(x, beta)
            return val
            
        def jac_func(x):
            _, grad = _smooth_max_grad(x, beta)
            return grad
            
        try:
            res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.success or res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    def _nesterov_projected(h0, beta, max_iter=2000, lr_init=0.01, deadline=None):
        """
        Nesterov accelerated projected gradient descent on the smooth max objective.
        Projects onto the box [0,1]^N and the hyperplane sum(h)=N/2.
        """
        nonlocal best_h, best_score
        
        def _project_box_sum(x):
            """Project x onto {h : 0<=h_i<=1, sum(h)=N/2}."""
            x = np.clip(x, 0.0, 1.0)
            diff = N / 2.0 - np.sum(x)
            if abs(diff) > 1e-14:
                mask = (x > 0) & (x < 1)
                if np.any(mask):
                    x[mask] += diff / np.sum(mask)
                    x = np.clip(x, 0.0, 1.0)
            return x
        
        x = h0.copy()
        v = np.zeros(N)
        lr = lr_init
        prev_val = _score(x)
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            val, grad = _smooth_max_grad(x, beta)
            
            # Nesterov step: x_{k+1} = x_k - lr * grad, then project
            # Use momentum
            v_new = 0.9 * v - lr * grad
            x_new = x + v_new
            x_new = _project_box_sum(x_new)
            
            new_val = _score(x_new)
            if new_val < prev_val - 1e-12:
                x = x_new
                v = v_new
                prev_val = new_val
                if new_val < best_score:
                    best_score = new_val
                    best_h = x.copy()
            else:
                # Reduce learning rate
                lr *= 0.9
                v = 0.9 * v
                # Try a smaller step
                x_try = x - 0.1 * lr * grad
                x_try = _project_box_sum(x_try)
                val_try = _score(x_try)
                if val_try < prev_val - 1e-12:
                    x = x_try
                    v = 0.9 * v - 0.1 * lr * grad
                    prev_val = val_try
                    if val_try < best_score:
                        best_score = val_try
                        best_h = x.copy()
        
        return x

    # Phase 1: Refine best analytical start with SLSQP
    if time.time() - t0 < _DEADLINE - 60:
        _run_optimization(best_h.copy(), beta=50.0, maxiter=200, ftol=1e-9)
    
    # Phase 2: Nesterov projected gradient for faster convergence
    if best_h is not None and time.time() - t0 < _DEADLINE - 40:
        h_nest = _nesterov_projected(best_h.copy(), beta=50.0, max_iter=3000, lr_init=0.005,
                                     deadline=_DEADLINE - 30)
        if h_nest is not None:
            h_nest = _fix_sum(h_nest)
            s_nest = _score(h_nest)
            if s_nest < best_score:
                best_h = h_nest
                best_score = s_nest
    
    # Phase 3: Try random starts
    if time.time() - t0 < _DEADLINE - 30:
        for _ in range(5):
            if time.time() - t0 > _DEADLINE - 15:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            if time.time() - t0 < _DEADLINE - 10:
                _run_optimization(h_rand, beta=50.0, maxiter=80, ftol=1e-8)
    
    # Phase 4: Coordinate descent on best solution
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        h_cd, score_cd = _coordinate_descent(best_h.copy(), max_sweeps=30, tol=1e-10)
        if score_cd < best_score:
            best_h = h_cd
            best_score = score_cd
    
    # Phase 5: Nesterov with higher beta for final polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 8:
        h_nest2 = _nesterov_projected(best_h.copy(), beta=150.0, max_iter=2000, lr_init=0.003,
                                      deadline=_DEADLINE - 5)
        if h_nest2 is not None:
            h_nest2 = _fix_sum(h_nest2)
            s_nest2 = _score(h_nest2)
            if s_nest2 < best_score:
                best_h = h_nest2
                best_score = s_nest2
    
    # Phase 6: Coordinate descent polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 4:
        h_cd2, score_cd2 = _coordinate_descent(best_h.copy(), max_sweeps=20, tol=1e-11)
        if score_cd2 < best_score:
            best_h = h_cd2
            best_score = score_cd2
    
    # Phase 7: Epigraph final polish if time permits
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        h_epi = _epigraph_optimize(best_h.copy(), maxiter=100, ftol=1e-13)
        if h_epi is not None:
            h_epi = _fix_sum(h_epi)
            s_epi = _score(h_epi)
            if s_epi < best_score:
                best_h = h_epi
                best_score = s_epi
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
