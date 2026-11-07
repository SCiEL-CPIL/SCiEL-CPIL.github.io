"""Root finder for 3 fixed two-dimensional systems using Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x such that ||F(x)|| <= 1e-8 using Broyden's method.
    """
    start_time = time.time()
    # Leave 10s margin for 120s limit, or 40s for 150s limit. 
    # The prompt says "KILLS after 150s". 
    # "Each evaluation must finish within 120 s" in contract.
    # Let's use 100s to be very safe.
    deadline = start_time + 100.0
    
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
        
    try:
        Fx = np.array(F(x), dtype=float)
    except Exception:
        return x

    if not np.all(np.isfinite(Fx)):
        return x

    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x

    # Initial Jacobian estimation using forward differences
    J = np.zeros((2, 2))
    h = 1e-5
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fx_e = np.array(F(x + e), dtype=float)
            if np.all(np.isfinite(Fx_e)):
                J[:, j] = (Fx_e - Fx) / h
            else:
                J[:, j] = np.array([0.0, 0.0])
        except Exception:
            J[:, j] = np.array([0.0, 0.0])
            
    # Check for singular/zero Jacobian
    try:
        cond = np.linalg.cond(J)
    except Exception:
        cond = np.inf
        
    if cond > 1e10 or np.allclose(J, 0):
        J = np.eye(2)

    calls = 3 # 1 for Fx, 2 for Jacobian columns
    best_x = x
    best_norm = norm_Fx
    max_iter = 100

    for k in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute Newton step
        try:
            s = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            try:
                s, _, _, _ = np.linalg.lstsq(J, -Fx, rcond=None)
            except Exception:
                break
        
        if not np.all(np.isfinite(s)):
            break
            
        # Backtracking line search
        step_scale = 1.0
        accepted = False
        Fx_old = Fx
        norm_Fx_old = norm_Fx
        
        for _ in range(10):
            x_new = x + step_scale * s
            if not np.all(np.isfinite(x_new)):
                step_scale *= 0.5
                continue
                
            try:
                Fx_new = np.array(F(x_new), dtype=float)
                calls += 1
            except Exception:
                break
            
            if not np.all(np.isfinite(Fx_new)):
                step_scale *= 0.5
                continue
                
            norm_Fx_new = np.linalg.norm(Fx_new)
            
            # Sufficient decrease condition
            if norm_Fx_new < norm_Fx_old:
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
                accepted = True
                break
            else:
                step_scale *= 0.5
                
        if not accepted:
            break
            
        if norm_Fx <= 1e-8:
            break
            
        # Broyden Rank-1 Update
        y = Fx - Fx_old
        s_actual = step_scale * s
        
        denom = np.dot(s_actual, s_actual)
        if denom > 1e-15:
            J = J + np.outer((y - J @ s_actual), s_actual) / denom
            
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x

    return best_x
# EVOLVE-BLOCK-END
