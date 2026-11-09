import numpy as np
import time

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return np.all(np.isfinite(x)) and x.shape == (2,)

def _norm(x):
    return float(np.linalg.norm(x))

def _safe_call(F, x):
    """Call F safely, returning None on failure."""
    try:
        y = F(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float)
        if y.shape != (2,):
            return None
        if not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None

def _dogleg_trust_region(F, x0, max_iter=40, tol=1e-9, call_budget=150):
    """
    Dogleg trust-region method on ||F(x)||^2.
    Uses a model m(d) = ||F + J d||^2 with J from finite differences.
    The dogleg path connects the Gauss-Newton step (d_gn) and the steepest descent step (d_sd).
    Returns (x, success, calls_used)
    """
    x = np.array(x0, dtype=float)
    calls = 0
    
    Fx = _safe_call(F, x)
    calls += 1
    if Fx is None:
        return x, False, calls
    
    f_curr = _norm(Fx)
    if f_curr <= 1e-8:
        return x, True, calls
    
    # Initial Jacobian estimate
    h = 1e-5
    e1 = np.array([h, 0.0])
    e2 = np.array([0.0, h])
    
    Fx1 = _safe_call(F, x + e1)
    Fx2 = _safe_call(F, x + e2)
    calls += 2
    
    if Fx1 is None or Fx2 is None:
        # Try smaller h
        h = 1e-6
        e1[0] = h
        e2[1] = h
        Fx1 = _safe_call(F, x + e1)
        Fx2 = _safe_call(F, x + e2)
        calls += 2
        if Fx1 is None or Fx2 is None:
            return x, False, calls
        
    J = np.zeros((2, 2))
    J[:, 0] = (Fx1 - Fx) / h
    J[:, 1] = (Fx2 - Fx) / h
    
    # Initial trust region radius
    # Heuristic: based on initial step size
    delta = 1.0
    
    for i in range(max_iter):
        if calls >= call_budget:
            break
        if _elapsed() > 110:
            break
            
        # Update Jacobian every iteration for accuracy (costs 2 calls)
        Fx1 = _safe_call(F, x + e1)
        Fx2 = _safe_call(F, x + e2)
        calls += 2
        
        if Fx1 is None or Fx2 is None:
             # If FD fails, break
             break
             
        J[:, 0] = (Fx1 - Fx) / h
        J[:, 1] = (Fx2 - Fx) / h
        
        # Compute Gauss-Newton step: J^T J d = -J^T F
        try:
            JTJ = J.T @ J
            JTg = J.T @ Fx
            d_gn = np.linalg.solve(JTJ, -JTg)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse or just break
            break
            
        if not np.all(np.isfinite(d_gn)):
            break
            
        # Steepest descent step: d_sd = -grad(f) / ||grad(f)||^2 * delta
        # grad(f) = 2 * J^T F
        grad = 2 * J.T @ Fx
        g_norm_sq = np.dot(grad, grad)
        
        if g_norm_sq < 1e-16:
            # Gradient is zero, but f is not? Unlikely.
            break
            
        d_sd = -grad / g_norm_sq * delta
        
        # Dogleg step:
        # 1. If ||d_gn|| <= delta, take d_gn.
        # 2. Else, take d = d_sd + t * (d_gn - d_sd) where t is chosen such that ||d|| = delta.
        #    d = d_sd + t * w, where w = d_gn - d_sd
        #    ||d_sd + t w||^2 = delta^2
        #    t^2 ||w||^2 + 2 t d_sd . w + ||d_sd||^2 - delta^2 = 0
        #    t = (-d_sd . w + sqrt((d_sd . w)^2 - ||w||^2 (||d_sd||^2 - delta^2))) / ||w||^2
        
        d_gn_norm = np.linalg.norm(d_gn)
        
        if d_gn_norm <= delta:
            d = d_gn
        else:
            w = d_gn - d_sd
            w_norm_sq = np.dot(w, w)
            if w_norm_sq < 1e-16:
                d = d_sd
            else:
                d_sd_dot_w = np.dot(d_sd, w)
                d_sd_norm_sq = np.dot(d_sd, d_sd)
                disc = d_sd_dot_w**2 - w_norm_sq * (d_sd_norm_sq - delta**2)
                if disc < 0:
                    # Numerical issue, fallback to steepest descent
                    d = d_sd
                else:
                    t = (-d_sd_dot_w + np.sqrt(disc)) / w_norm_sq
                    # Clamp t to [0, 1]
                    t = np.clip(t, 0.0, 1.0)
                    d = d_sd + t * w
        
        x_new = x + d
        Fx_new = _safe_call(F, x_new)
        calls += 1
        
        if Fx_new is None:
            delta *= 0.5
            continue
            
        f_new = _norm(Fx_new)
        
        if f_new < f_curr:
            # Accept
            x = x_new
            Fx = Fx_new
            f_curr = f_new
            
            if f_curr <= 1e-8:
                return x, True, calls
                
            # Adjust trust region
            # If step was truncated (dogleg), increase delta. Otherwise, keep or increase slightly.
            # Simple heuristic: if d was d_gn, increase delta. If d was on the dogleg, increase delta.
            # If d was d_sd, decrease delta.
            # We don't track which one exactly, but we can check if d == d_gn or d == d_sd.
            # For simplicity: if f_new < f_curr, increase delta slightly, capped.
            delta = min(delta * 1.5, 10.0)
        else:
            # Reject
            delta *= 0.5
            if delta < 1e-10:
                break
                
    return x, False, calls

def find_root_system(F, x0):
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + 110.0
    
    x0 = np.asarray(x0, dtype=float)
    
    # Strategy: Dogleg Trust Region from x0.
    # If it fails, try a few other starting points.
    
    # 1. Try x0
    x_res, success, calls_used = _dogleg_trust_region(F, x0, max_iter=30, call_budget=100)
    
    if success:
        return x_res
        
    # If failed, check if we found a good point anyway
    F_final = _safe_call(F, x_res)
    if F_final is not None:
        calls_used += 1
        if _norm(F_final) <= 1e-8:
            return x_res
            
    # 2. Try other starting points if budget allows
    # Global budget: 300 calls
    global_call_budget = 300
    
    if calls_used >= global_call_budget:
        return x_res
        
    # Generate some alternative starting points
    scale = max(1.0, np.linalg.norm(x0))
    offsets = [
        [0.1 * scale, 0.0], [-0.1 * scale, 0.0],
        [0.0, 0.1 * scale], [0.0, -0.1 * scale],
        [0.5 * scale, 0.0], [-0.5 * scale, 0.0],
        [0.0, 0.5 * scale], [0.0, -0.5 * scale],
    ]
    
    best_x = x_res
    best_norm = _norm(F_final) if F_final is not None else float('inf')
    
    for offset in offsets:
        if _elapsed() > deadline:
            break
        if calls_used >= global_call_budget:
            break
            
        x_start = x0 + np.array(offset)
        x_res2, success2, calls_in_run = _dogleg_trust_region(F, x_start, max_iter=20, call_budget=50)
        calls_used += calls_in_run
        
        if success2:
            return x_res2
            
        # Check best
        F_final2 = _safe_call(F, x_res2)
        if F_final2 is not None:
            calls_used += 1
            norm_final2 = _norm(F_final2)
            if norm_final2 < best_norm:
                best_norm = norm_final2
                best_x = x_res2
                
    return best_x
