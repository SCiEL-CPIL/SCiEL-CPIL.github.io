import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if dx < t and dy < t:
                return False
    return True

def _objective(params, n=10, tol=1e-7):
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    pen = 0.0
    # Boundary penalties
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            pen += 1e6 * (1.0 - s)
            continue
        if cx - s/2 < -tol:
            pen += 1e6 * (-tol - (cx - s/2))
        if cx + s/2 > 1 + tol:
            pen += 1e6 * ((cx + s/2) - (1 + tol))
        if cy - s/2 < -tol:
            pen += 1e6 * (-tol - (cy - s/2))
        if cy + s/2 > 1 + tol:
            pen += 1e6 * ((cy + s/2) - (1 + tol))
            
    # Overlap penalties using smooth log-sum-exp surrogate
    # For each pair, we want to penalize when both |dx| < t and |dy| < t
    # Use smooth max: soft_max(a,b) = log(exp(a/eps) + exp(b/eps)) * eps
    # where a = t - |dx|, b = t - |dy|
    # Overlap exists when both a > 0 and b > 0
    # We penalize the product of soft-max values
    eps = 0.01
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides[i] + sides[j]) / 2.0 - tol
            vx = t - abs(dx)
            vy = t - abs(dy)
            # Smooth max of 0 and vx
            svx = eps * np.log(1.0 + np.exp(vx / eps))
            svy = eps * np.log(1.0 + np.exp(vy / eps))
            pen += 1e5 * (svx * svy)
            
    return -np.sum(sides) + pen

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()

    # Configuration: 4 large (0.375), 6 small (0.25)
    s_large = 0.375
    s_small = 0.25
    
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 4 Large at corners
    centers[0] = [s_large/2, s_large/2]
    sides[0] = s_large
    centers[1] = [1 - s_large/2, s_large/2]
    sides[1] = s_large
    centers[2] = [s_large/2, 1 - s_large/2]
    sides[2] = s_large
    centers[3] = [1 - s_large/2, 1 - s_large/2]
    sides[3] = s_large
    
    # 4 Small in central strip
    centers[4] = [0.5, 0.125]
    sides[4] = s_small
    centers[5] = [0.5, 0.375]
    sides[5] = s_small
    centers[6] = [0.5, 0.625]
    sides[6] = s_small
    centers[7] = [0.5, 0.875]
    sides[7] = s_small
    
    # 2 Small in side gaps
    centers[8] = [0.1875, 0.5]
    sides[8] = s_small
    centers[9] = [0.8125, 0.5]
    sides[9] = s_small
    
    update_best(centers, sides)
    
    # Optimization to see if we can improve
    params_init = np.concatenate([centers.flatten(), sides])
    
    try:
        res = minimize(_objective, params_init, args=(n,), method='Powell',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        params_opt = res.x
        c_opt = params_opt[:2*n].reshape(n, 2)
        s_opt = np.maximum(params_opt[2*n:], 1e-6)
        update_best(c_opt, s_opt)
    except Exception:
        pass
    
    # Additional optimization with different initial configurations
    # Try a more symmetric starting point
    if time.time() < deadline - 30:
        # Alternative initial configuration: 2 very large + 8 small
        s_large2 = 0.45
        s_small2 = 0.22
        centers_alt = np.zeros((n, 2))
        sides_alt = np.zeros(n)
        
        # 2 large squares
        centers_alt[0] = [s_large2/2, s_large2/2]
        sides_alt[0] = s_large2
        centers_alt[1] = [1 - s_large2/2, 1 - s_large2/2]
        sides_alt[1] = s_large2
        
        # 8 small squares around
        # Top edge
        centers_alt[2] = [0.25, 1 - s_small2/2]
        sides_alt[2] = s_small2
        centers_alt[3] = [0.75, 1 - s_small2/2]
        sides_alt[3] = s_small2
        
        # Bottom edge
        centers_alt[4] = [0.25, s_small2/2]
        sides_alt[4] = s_small2
        centers_alt[5] = [0.75, s_small2/2]
        sides_alt[5] = s_small2
        
        # Left edge
        centers_alt[6] = [s_small2/2, 0.5]
        sides_alt[6] = s_small2
        
        # Right edge
        centers_alt[7] = [1 - s_small2/2, 0.5]
        sides_alt[7] = s_small2
        
        # Center
        centers_alt[8] = [0.5, 0.5]
        sides_alt[8] = s_small2
        centers_alt[9] = [0.5, 0.5]
        sides_alt[9] = s_small2
        
        update_best(centers_alt, sides_alt)
        
        params_alt = np.concatenate([centers_alt.flatten(), sides_alt])
        try:
            res = minimize(_objective, params_alt, args=(n,), method='Powell',
                           options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9})
            params_opt = res.x
            c_opt = params_opt[:2*n].reshape(n, 2)
            s_opt = np.maximum(params_opt[2*n:], 1e-6)
            update_best(c_opt, s_opt)
        except Exception:
            pass
    
    # Multi-start with random perturbations
    if best_centers is not None:
        rng = np.random.default_rng(42)
        for restart in range(40):
            if time.time() > deadline - 15:
                break
            
            # Create a perturbed version of the best solution
            c_pert = best_centers.copy()
            s_pert = best_sides.copy()
            
            # Random perturbation
            c_pert += rng.normal(0, 0.04, size=c_pert.shape)
            s_pert += rng.normal(0, 0.015, size=s_pert.shape)
            s_pert = np.maximum(s_pert, 0.05)
            
            # Clamp to valid range
            for i in range(n):
                s = s_pert[i]
                c_pert[i, 0] = np.clip(c_pert[i, 0], s/2, 1 - s/2)
                c_pert[i, 1] = np.clip(c_pert[i, 1], s/2, 1 - s/2)
            
            params_pert = np.concatenate([c_pert.flatten(), s_pert])
            
            try:
                res = minimize(_objective, params_pert, args=(n,), method='Powell',
                               options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-9})
                params_opt = res.x
                c_opt = params_opt[:2*n].reshape(n, 2)
                s_opt = np.maximum(params_opt[2*n:], 1e-6)
                update_best(c_opt, s_opt)
            except Exception:
                pass
    
    # Final high-precision polish on the best solution found
    if best_centers is not None and time.time() < deadline - 5:
        params_best = np.concatenate([best_centers.flatten(), best_sides])
        try:
            res = minimize(_objective, params_best, args=(n,), method='Powell',
                           options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-11})
            params_opt = res.x
            c_opt = params_opt[:2*n].reshape(n, 2)
            s_opt = np.maximum(params_opt[2*n:], 1e-6)
            update_best(c_opt, s_opt)
        except Exception:
            pass
    
    if best_centers is None:
        # Fallback
        best_centers = centers
        best_sides = sides

    return best_centers, best_sides
