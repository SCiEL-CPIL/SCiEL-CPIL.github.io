import numpy as np
import time
from scipy.optimize import minimize

N = 100
K = 10 # Number of control points

def _correlation_value(h):
    """Compute the evaluator's value: max(corr) / N * 2"""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h to [0,1] and sum=N/2 using bisection on a global shift"""
    target = N / 2.0
    
    lo = -np.max(h) - 1.0
    hi = 1.0 - np.min(h) + 1.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        val = np.sum(np.clip(h + mid, 0.0, 1.0)) - target
        if val > 0:
            hi = mid
        else:
            lo = mid
            
    s = (lo + hi) / 2.0
    h_proj = np.clip(h + s, 0.0, 1.0)
    
    diff = np.sum(h_proj) - target
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmax(h_proj)
        else:
            idx = np.argmin(h_proj)
        h_proj[idx] -= diff
        h_proj = np.clip(h_proj, 0.0, 1.0)
        
    return h_proj

def _soft_max(corr, lam):
    """Soft max using log-sum-exp for numerical stability"""
    m = np.max(corr)
    if m == 0:
        return 0.0
    shifted = lam * (corr - m)
    s = np.sum(np.exp(shifted))
    if s == 0:
        return m
    return m + np.log(s) / lam

def _interp_controls(c):
    """Interpolate K control points to N heights"""
    # Control points are defined at indices 0, N/(K-1), ..., N-1
    # We want the first and last to be exactly the control points.
    # np.interp expects increasing x.
    x_ctrl = np.linspace(0, N-1, K)
    x_full = np.arange(N)
    h = np.interp(x_full, x_ctrl, c)
    return h

def _make_low_dim_obj(lam):
    def obj(c):
        h = _interp_controls(c)
        h = _project_to_feasible(h)
        corr = np.correlate(h, 1 - h, mode="full")
        return _soft_max(corr, lam)
    return obj

def _make_high_dim_obj(lam):
    def obj(h):
        h = _project_to_feasible(h)
        corr = np.correlate(h, 1 - h, mode="full")
        return _soft_max(corr, lam)
    return obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    bounds_ctrl = [(0.0, 1.0)] * K
    bounds_full = [(0.0, 1.0)] * N
    
    # Phase 1: Low-dimensional search
    # Try multiple random starts for the control points
    rng = np.random.RandomState(42)
    
    # Include some structured starts
    inits_ctrl = [
        np.full(K, 0.5),
        np.linspace(0, 1, K),
        np.linspace(1, 0, K),
        np.sin(np.linspace(0, 2*np.pi, K)) * 0.5 + 0.5,
        np.cos(np.linspace(0, 2*np.pi, K)) * 0.5 + 0.5,
        np.sign(np.sin(np.linspace(0, 2*np.pi, K))) * 0.5 + 0.5,
    ]
    # Add random starts
    for _ in range(10):
        inits_ctrl.append(rng.uniform(0, 1, K))
    
    # Ramp lambda for low dim
    lam_schedule = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    for lam in lam_schedule:
        if time.time() > deadline - 60:
            break
            
        obj_low = _make_low_dim_obj(lam)
        
        for c0 in inits_ctrl:
            if time.time() > deadline - 60:
                break
                
            try:
                res = minimize(
                    obj_low,
                    c0,
                    method='L-BFGS-B',
                    bounds=bounds_ctrl,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6, 'maxls': 20}
                )
                h_opt = _interp_controls(res.x)
                h_opt = _project_to_feasible(h_opt)
                val_opt = _correlation_value(h_opt)
                
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
                    
            except Exception:
                pass
    
    # Phase 2: High-dimensional polish
    # Use the best low-dim result as a start for high-dim optimization
    # Also try the seed and maybe a few other high-dim starts if time permits
    
    lam_high_schedule = [500.0, 1000.0, 2000.0, 5000.0]
    
    starts_high = [
        best_h.copy(),
        np.full(N, 0.5),
        np.linspace(0, 1, N),
        np.sin(np.linspace(0, 2*np.pi, N)) * 0.5 + 0.5,
    ]
    
    for lam in lam_high_schedule:
        if time.time() > deadline - 20:
            break
            
        obj_high = _make_high_dim_obj(lam)
        
        for h0 in starts_high:
            if time.time() > deadline - 20:
                break
                
            # Project start to feasible
            h0_feas = _project_to_feasible(h0)
            
            try:
                res = minimize(
                    obj_high,
                    h0_feas,
                    method='L-BFGS-B',
                    bounds=bounds_full,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
                )
                h_opt = _project_to_feasible(res.x)
                val_opt = _correlation_value(h_opt)
                
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
                    # Update the start for next lambda iteration to be the new best
                    starts_high[0] = best_h.copy()
                    
            except Exception:
                pass
    
    # Final high precision polish on the very best
    if time.time() < deadline - 5:
        lam_final = 10000.0
        obj_final = _make_high_dim_obj(lam_final)
        try:
            res = minimize(
                obj_final,
                best_h,
                method='L-BFGS-B',
                bounds=bounds_full,
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            h_opt = _project_to_feasible(res.x)
            val_opt = _correlation_value(h_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
            
    best_h = _project_to_feasible(best_h)
    
    return best_h
