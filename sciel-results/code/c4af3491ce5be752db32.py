"""Root finder for 3 fixed 2D systems using Trust Region Method with Broyden Updates."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _safe_eval(F, x, call_counter):
    """Safely evaluate F at x, incrementing call_counter[0]."""
    call_counter[0] += 1
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,):
            if Fx.size == 2:
                Fx = Fx.reshape(2)
            else:
                return None, float('inf'), False
        if not np.all(np.isfinite(Fx)):
            return None, float('inf'), False
        norm_Fx = float(np.linalg.norm(Fx))
        return Fx, norm_Fx, True
    except Exception:
        return None, float('inf'), False

def _trust_region_solve(F, x0, call_counter, max_iter=15, tol=1e-10):
    """
    Solve F(x) = 0 using a Trust Region method.
    Starts at x0.
    Returns (x, norm_F).
    """
    x = x0.copy()
    
    # Initial Evaluation
    Fx, norm_F, valid = _safe_eval(F, x, call_counter)
    if not valid:
        return x, float('inf')
        
    if norm_F <= tol:
        return x, norm_F

    # Estimate Jacobian using Central Differences
    h = 1e-5
    J = np.zeros((2, 2))
    for i in range(2):
        x_plus = x.copy()
        x_plus[i] += h
        Fx_plus, _, valid_plus = _safe_eval(F, x_plus, call_counter)
        
        x_minus = x.copy()
        x_minus[i] -= h
        Fx_minus, _, valid_minus = _safe_eval(F, x_minus, call_counter)
        
        if not valid_plus or not valid_minus:
            # Fallback to Identity if Jacobian estimation fails
            J = np.eye(2)
            break
        J[:, i] = (Fx_plus - Fx_minus) / (2 * h)
    
    # Trust Region Radius
    delta = 1.0
    
    for _ in range(max_iter):
        if call_counter[0] > 480:
            break
            
        # 1. Solve Subproblem
        # min ||Fx + J s||^2 s.t. ||s|| <= delta
        # Gauss-Newton step: s_gn = -J^{-1} Fx
        try:
            s_gn = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse or damped step
            try:
                s_gn = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                s_gn = -0.1 * Fx
                
        norm_s_gn = np.linalg.norm(s_gn)
        
        if norm_s_gn <= delta:
            s = s_gn
        else:
            # Shrink step to fit trust region
            s = delta * s_gn / norm_s_gn
            
        # 2. Take Step
        x_new = x + s
        F_new, norm_new, valid_new = _safe_eval(F, x_new, call_counter)
        
        if not valid_new:
            # Step went to invalid region, shrink trust region
            delta *= 0.5
            continue
            
        # 3. Calculate Reduction Ratio
        # Predicted Reduction: ||Fx||^2 - ||Fx + J s||^2
        pred_F = Fx + J @ s
        pred_red = np.dot(Fx, Fx) - np.dot(pred_F, pred_F)
        
        # Actual Reduction: ||Fx||^2 - ||F_new||^2
        act_red = np.dot(Fx, Fx) - np.dot(F_new, F_new)
        
        if pred_red > 1e-12:
            rho = act_red / pred_red
        else:
            rho = 0.0
            
        # 4. Update Trust Region and State
        if rho > 0.75:
            # Good agreement, expand trust region
            delta *= 2.0
            x = x_new
            Fx = F_new
            norm_F = norm_new
            
            # Update Jacobian (Broyden's Good)
            delta_F = F_new - Fx
            delta_x = x_new - x
            denom = np.dot(delta_x, delta_x)
            if denom > 1e-12:
                J = J + np.outer(delta_F - J @ delta_x, delta_x) / denom
                
        elif rho > 0.25:
            # Acceptable agreement, keep trust region
            x = x_new
            Fx = F_new
            norm_F = norm_new
            
            # Update Jacobian
            delta_F = F_new - Fx
            delta_x = x_new - x
            denom = np.dot(delta_x, delta_x)
            if denom > 1e-12:
                J = J + np.outer(delta_F - J @ delta_x, delta_x) / denom
                
        else:
            # Poor agreement, shrink trust region
            delta *= 0.5
            
        if norm_F <= tol:
            break
            
    return x, norm_F

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 140.0  # Safety margin from 150s
    
    x0_arr = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)
        
    call_counter = [0]
    max_calls = 500
    
    best_x = x0_arr
    best_norm = float('inf')
    
    # Check x0 first
    Fx0, norm0, valid0 = _safe_eval(F, x0_arr, call_counter)
    if valid0:
        best_norm = norm0
        best_x = x0_arr
        if norm0 <= 1e-8:
            return best_x

    # Generate starting points
    # 1. x0
    # 2. Origin
    # 3. Points on a circle/grid
    # 4. Random points
    
    starts = [x0_arr, np.zeros(2)]
    
    # Grid/Circle points
    radii = [0.5, 1.0, 2.0, 5.0]
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    for r in radii:
        for a in angles:
            starts.append(np.array([r * np.cos(a), r * np.sin(a)]))
            
    # Random points
    rng = np.random.RandomState(42)
    for _ in range(10):
        starts.append(rng.uniform(-10, 10, 2))
        
    # Run Trust Region from each start
    for x_start in starts:
        if time.time() - start_time > time_limit:
            break
        if call_counter[0] > max_calls - 20: # Reserve calls for safety
            break
            
        x_res, norm_res = _trust_region_solve(F, x_start, call_counter, max_iter=15)
        
        if norm_res < best_norm:
            best_norm = norm_res
            best_x = x_res
            
        if best_norm <= 1e-8:
            break
            
    return best_x
# EVOLVE-BLOCK-END
