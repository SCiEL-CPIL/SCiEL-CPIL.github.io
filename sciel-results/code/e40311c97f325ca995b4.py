import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
import sys

N = 50
TIME_LIMIT = 115.0  # safety margin before 120s kill

def _autoconv(h):
    """Compute autoconvolution of h with itself (linear convolution, no zero-padding needed for full support)."""
    # h has length N, support on indices 0..N-1
    # autoconv[k] = sum_i h[i] * h[i-k] for k in -(N-1)..(N-1)
    # We only need the non-negative lag part or full? 
    # Supremum norm of autoconvolution: max over all lags.
    # For nonnegative h, the max is likely at lag 0 or near it.
    # Let's compute full autoconvolution.
    h = np.asarray(h)
    ac = np.convolve(h, h, mode='full')
    return ac

def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||autoconv(h)||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(h**2)
    ac = _autoconv(h)
    ac_max = np.max(ac)
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def construct():
    start_time = time.time()
    
    # Initial guess: uniform
    best_h = np.ones(N) / N
    best_score = _score(best_h)
    
    # Try a few random initializations
    np.random.seed(42)
    for _ in range(20):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        h_init = np.random.dirichlet(np.ones(N))
        s = _score(h_init)
        if s > best_score:
            best_score = s
            best_h = h_init.copy()
    
    # Main optimization: maximize score = ||h||_2^2 / (||h||_1 * ||ac||_inf)
    # Since h is nonnegative, let's normalize ||h||_1 = 1 to simplify.
    # Then maximize ||h||_2^2 / ||ac||_inf subject to sum(h)=1, h>=0.
    # Actually, the ratio is scale-invariant, so we can fix ||h||_1 = 1.
    
    def _score_normalized(h_raw):
        """h_raw is unconstrained, we project to simplex."""
        h = np.maximum(h_raw, 0)
        s = np.sum(h)
        if s < 1e-12:
            return 1e12
        h = h / s
        ac = _autoconv(h)
        ac_max = np.max(ac)
        if ac_max < 1e-12:
            return 1e12
        return -(np.sum(h**2) / ac_max)  # minimize negative score
    
    # Use L-BFGS-B or SLSQP
    # Better: use the fact that we can reparameterize.
    # Let's try a direct optimization on the simplex using projected gradient or similar.
    # But scipy's minimize with bounds is simpler.
    
    def _objective(h_raw):
        h = np.maximum(h_raw, 0)
        s = np.sum(h)
        if s < 1e-12:
            return 1e12
        h = h / s
        ac = _autoconv(h)
        ac_max = np.max(ac)
        if ac_max < 1e-12:
            return 1e12
        return -(np.sum(h**2) / ac_max)
    
    # Start from best_h normalized
    h0 = best_h / np.sum(best_h)
    
    # Try SLSQP
    try:
        res = minimize(_objective, h0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success or res.fun < _objective(h0):
            h_opt = np.maximum(res.x, 0)
            s = np.sum(h_opt)
            if s > 1e-12:
                h_opt = h_opt / s
                s_opt = _score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
    except Exception:
        pass
    
    # Try a few more random restarts with optimization
    for i in range(50):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        h_init = np.random.dirichlet(np.ones(N) * 0.5)  # more peaked
        h0 = h_init / np.sum(h_init)
        try:
            res = minimize(_objective, h0, method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-9})
            if res.fun < 1e11:
                h_opt = np.maximum(res.x, 0)
                s = np.sum(h_opt)
                if s > 1e-12:
                    h_opt = h_opt / s
                    s_opt = _score(h_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish: try to increase peakiness
    # The optimal solution for this type of problem often has a small support.
    # Let's try solutions with support of size k for k in 2..10
    for k in range(2, min(15, N)):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        # Try all subsets of size k? Too many. Try random subsets.
        for _ in range(100):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            idx = np.random.choice(N, k, replace=False)
            h_sub = np.random.dirichlet(np.ones(k))
            h_full = np.zeros(N)
            h_full[idx] = h_sub
            s = _score(h_full)
            if s > best_score:
                best_score = s
                best_h = h_full.copy()
            
            # Optimize on this support
            def _obj_sub(x):
                h_full = np.zeros(N)
                h_full[idx] = np.maximum(x, 0)
                return _objective(h_full)
            
            try:
                res = minimize(_obj_sub, h_sub, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-10})
                if res.fun < 1e11:
                    h_sub_opt = np.maximum(res.x, 0)
                    s_sub = np.sum(h_sub_opt)
                    if s_sub > 1e-12:
                        h_sub_opt = h_sub_opt / s_sub
                        h_full = np.zeros(N)
                        h_full[idx] = h_sub_opt
                        s_opt = _score(h_full)
                        if s_opt > best_score:
                            best_score = s_opt
                            best_h = h_full.copy()
            except Exception:
                pass
    
    # Return the best heights
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print("Score:", _score(h))
    print("Heights:", h)
