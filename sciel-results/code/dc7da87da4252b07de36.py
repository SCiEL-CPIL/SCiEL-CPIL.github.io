"""Root finder for 3 fixed 2D systems using Multi-Start Newton with Central Differences."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _safe_eval(F, x):
    """Safely evaluate F at x, returning (Fx, norm_Fx, is_valid)."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,):
            Fx = Fx.reshape(2)
        if not np.all(np.isfinite(Fx)):
            return None, float('inf'), False
        norm_Fx = np.linalg.norm(Fx)
        return Fx, norm_Fx, True
    except Exception:
        return None, float('inf'), False

def _newton_central_diff(F, x0, max_iter=10, tol=1e-10):
    """
    Newton's method with central difference Jacobian.
    Each iteration costs 4 function calls (for Jacobian) + 1 (for F at new x) = 5 calls.
    However, we can reuse the central difference points if we structure it carefully, 
    but standard central diff is 4 calls. Let's stick to standard central diff for robustness.
    Actually, to save calls, we can use forward differences (2 calls) but central is more stable.
    Given the 500 call limit, 10 iterations * 5 calls = 50 calls per start.
    We can afford ~10 starts.
    """
    x = np.array(x0, dtype=float)
    h = 1e-5
    
    Fx, norm_F, valid = _safe_eval(F, x)
    if not valid:
        return None, float('inf')
    if norm_F <= tol:
        return x, norm_F

    for _ in range(max_iter):
        # Central Difference Jacobian
        # J[:, 0] = (F(x + h*e0) - F(x - h*e0)) / (2h)
        # J[:, 1] = (F(x + h*e1) - F(x - h*e1)) / (2h)
        # This costs 4 calls.
        try:
            e0 = np.array([h, 0.0])
            e1 = np.array([0.0, h])
            
            Fp0, _, v0 = _safe_eval(F, x + e0)
            Fm0, _, v1 = _safe_eval(F, x - e0)
            Fp1, _, v2 = _safe_eval(F, x + e1)
            Fm1, _, v3 = _safe_eval(F, x - e1)
            
            if not (v0 and v1 and v2 and v3):
                break
                
            J = np.zeros((2, 2))
            J[:, 0] = (Fp0 - Fm0) / (2 * h)
            J[:, 1] = (Fp1 - Fm1) / (2 * h)
            
            if not np.all(np.isfinite(J)):
                break
                
            # Solve J * s = -Fx
            try:
                s = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If singular, try pseudo-inverse or break
                s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                
            if not np.all(np.isfinite(s)):
                break
                
            x_new = x + s
            F_new, norm_new, valid_new = _safe_eval(F, x_new)
            
            if not valid_new:
                # If new point is invalid, try a smaller step
                x_new = x + 0.5 * s
                F_new, norm_new, valid_new = _safe_eval(F, x_new)
                if not valid_new:
                    break
                    
            x = x_new
            Fx = F_new
            norm_F = norm_new
            
            if norm_F <= tol:
                return x, norm_F
                
        except Exception:
            break
            
    return x, norm_F

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 140.0  # Safety margin from 150s
    
    # Generate a set of diverse starting points
    starts = []
    
    # 1. The provided start
    x0_arr = np.array(x0, dtype=float)
    if np.all(np.isfinite(x0_arr)):
        starts.append(x0_arr)
    else:
        starts.append(np.zeros(2))
        
    # 2. Origin
    starts.append(np.array([0.0, 0.0]))
    
    # 3. Points on unit circle and small radii
    # Use a fixed set of angles to ensure reproducibility and coverage
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    radii = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for r in radii:
        for a in angles:
            starts.append(np.array([r * np.cos(a), r * np.sin(a)]))
            
    # 4. Random points in a larger box (seeded for reproducibility)
    rng = np.random.RandomState(42)
    for _ in range(10):
        starts.append(rng.uniform(-10, 10, 2))
        
    # Sort starts by distance to x0? Or just iterate?
    # Iterating in order: x0, origin, then others.
    
    best_x = starts[0]
    best_norm = float('inf')
    call_count = 0
    max_calls = 500  # Hard limit per system
    
    for x_start in starts:
        if time.time() - start_time > time_limit:
            break
            
        # Check if we've used too many calls
        # We don't have a direct call counter, but we can estimate.
        # _newton_central_diff uses ~5 calls per iter, max 10 iters = 50 calls.
        # We have 500 calls. So we can afford ~10 starts.
        # Let's track approximate calls to be safe.
        if call_count > max_calls - 60: # Reserve 60 calls for safety/next start
            break
            
        x_res, norm_res = _newton_central_diff(F, x_start, max_iter=10, tol=1e-10)
        
        # Estimate calls used: 1 (initial) + 10*5 (max) = 51
        # In reality, it stops early. But for budgeting, assume worst case.
        call_count += 51 
        
        if norm_res < best_norm:
            best_norm = norm_res
            best_x = x_res
            
        if best_norm <= 1e-8:
            break
            
    return best_x
# EVOLVE-BLOCK-END
