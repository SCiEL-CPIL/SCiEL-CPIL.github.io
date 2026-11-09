# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _smooth_overlap_penalty(centers, sides, tol=1e-7):
    """
    Smooth penalty for overlaps.
    Uses a soft-max approximation for the max(0, overlap) terms.
    """
    n = len(sides)
    pen = 0.0
    eps = 1e-12
    sqrt_eps = np.sqrt(eps)
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t <= 0:
                continue
            ox = t - dx
            oy = t - dy
            # Smooth max(0, x) approximation: g(x) = (x + sqrt(x^2 + eps))/2 - sqrt(eps)/2
            # This is C1 smooth and approximates max(0, x) well.
            soft_A = (ox + np.sqrt(ox*ox + eps)) / 2.0 - sqrt_eps
            soft_B = (oy + np.sqrt(oy*oy + eps)) / 2.0 - sqrt_eps
            if soft_A > 0 and soft_B > 0:
                pen += soft_A * soft_B
    return pen

def _boundary_penalty(centers, sides, tol=1e-7):
    """Penalty for squares going outside [0,1]^2."""
    pen = 0.0
    eps = 1e-12
    sqrt_eps = np.sqrt(eps)
    for i in range(len(sides)):
        cx, cy = centers[i]
        s = sides[i]
        v_l = -tol - (cx - s/2)
        v_r = (cx + s/2) - (1 + tol)
        v_b = -tol - (cy - s/2)
        v_t = (cy + s/2) - (1 + tol)
        
        def smooth_max_0(x):
            return (x + np.sqrt(x*x + eps)) / 2.0 - sqrt_eps
        
        pen += smooth_max_0(v_l)**2
        pen += smooth_max_0(v_r)**2
        pen += smooth_max_0(v_b)**2
        pen += smooth_max_0(v_t)**2
    return pen

def _objective(params, n=10, penalty_weight=100.0):
    """Objective: minimize -sum(sides) + penalties."""
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    # Ensure sides are positive
    sides_clipped = np.maximum(sides, 1e-6)
    
    obj = -np.sum(sides_clipped)
    obj += penalty_weight * _smooth_overlap_penalty(centers, sides_clipped)
    obj += penalty_weight * _boundary_penalty(centers, sides_clipped)
    return obj

def _is_valid(centers, sides, tol=1e-7):
    """Check if configuration is valid."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Boundary check
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Overlap check
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if dx < t and dy < t:
                return False
    return True

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
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

    # --- Initial Configurations ---
    
    # Config 1: 2 large (0.5), 8 small (0.25)
    # Layout:
    # Top Left: 0.5x0.5 at (0.25, 0.75)
    # Top Right: 0.5x0.5 at (0.75, 0.75)
    # Bottom Left: 4 squares of 0.25 in 0.5x0.5 area
    # Bottom Right: 4 squares of 0.25 in 0.5x0.5 area
    
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    c1[0] = [0.25, 0.75]; s1[0] = 0.5
    c1[1] = [0.75, 0.75]; s1[1] = 0.5
    
    # Bottom Left Quadrant (0 to 0.5, 0 to 0.5)
    c1[2] = [0.125, 0.375]; s1[2] = 0.25
    c1[3] = [0.375, 0.375]; s1[3] = 0.25
    c1[4] = [0.125, 0.125]; s1[4] = 0.25
    c1[5] = [0.375, 0.125]; s1[5] = 0.25
    
    # Bottom Right Quadrant (0.5 to 1.0, 0 to 0.5)
    c1[6] = [0.625, 0.375]; s1[6] = 0.25
    c1[7] = [0.875, 0.375]; s1[7] = 0.25
    c1[8] = [0.625, 0.125]; s1[8] = 0.25
    c1[9] = [0.875, 0.125]; s1[9] = 0.25
    
    update_best(c1, s1)
    
    # Config 2: 3x3 grid of 0.3 + 1 center 0.1
    c3 = np.zeros((n, 2))
    s3 = np.zeros(n)
    idx = 0
    for r in range(3):
        for c in range(3):
            c3[idx] = [0.05 + (c + 0.5)*0.3, 0.05 + (r + 0.5)*0.3]
            s3[idx] = 0.3
            idx += 1
    c3[9] = [0.5, 0.5]
    s3[9] = 0.1
    update_best(c3, s3)

    # Config 3: 4x3 grid (12 cells, use 10)
    # 4 cols, 3 rows. Side = 0.25.
    c4 = np.zeros((n, 2))
    s4 = np.zeros(n)
    for i in range(n):
        r = i // 4
        c = i % 4
        c4[i] = [(c + 0.5) * 0.25, (r + 0.5) * 0.25]
        s4[i] = 0.25
    update_best(c4, s4)

    # Config 4: 5x2 grid
    c5 = np.zeros((n, 2))
    s5 = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        c5[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s5[i] = 0.2
    update_best(c5, s5)

    # Random Restarts
    np.random.seed(42)
    
    starts = []
    starts.append((c1.copy(), s1.copy()))
    starts.append((c3.copy(), s3.copy()))
    starts.append((c4.copy(), s4.copy()))
    starts.append((c5.copy(), c5.copy())) # Wait, sides should be s5
    
    # Correcting starts list
    starts = [(c1.copy(), s1.copy()), (c3.copy(), s3.copy()), (c4.copy(), s4.copy()), (c5.copy(), s5.copy())]
    
    # Add random perturbations of Config 1
    for _ in range(5):
        c_r = c1.copy() + np.random.normal(0, 0.02, (n, 2))
        s_r = s1.copy() + np.random.normal(0, 0.01, n)
        s_r = np.clip(s_r, 0.1, 0.6)
        starts.append((c_r, s_r))
        
    # Add random perturbations of Config 3
    for _ in range(5):
        c_r = c3.copy() + np.random.normal(0, 0.02, (n, 2))
        s_r = s3.copy() + np.random.normal(0, 0.01, n)
        s_r = np.clip(s_r, 0.05, 0.5)
        starts.append((c_r, s_r))
        
    # Add random perturbations of Config 4
    for _ in range(5):
        c_r = c4.copy() + np.random.normal(0, 0.02, (n, 2))
        s_r = s4.copy() + np.random.normal(0, 0.01, n)
        s_r = np.clip(s_r, 0.1, 0.4)
        starts.append((c_r, s_r))

    # Pure random starts
    for _ in range(10):
        c_r = np.random.uniform(0.1, 0.9, (n, 2))
        s_r = np.random.uniform(0.1, 0.4, n)
        starts.append((c_r, s_r))

    # Optimization Loop
    # We will use SLSQP with smooth penalties.
    # To improve performance, we can try multiple penalty weights if stuck, 
    # but for simplicity and speed, we stick to one weight per start.
    
    for c_init, s_init in starts:
        if time.time() > deadline:
            break
            
        params_init = np.concatenate([c_init.flatten(), s_init])
        
        try:
            # Use SLSQP
            res = minimize(_objective, params_init, args=(n, 100.0), method='SLSQP',
                           options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
            
            if res.success or res.fun < _objective(params_init, n, 100.0):
                params_opt = res.x
                c_opt = params_opt[:2*n].reshape(n, 2)
                s_opt = np.maximum(params_opt[2*n:], 1e-6)
                
                update_best(c_opt, s_opt)
                
                # Polish step: If we found a valid solution, try to polish it with tighter tolerance
                if _is_valid(c_opt, s_opt):
                    # Try a second pass with lower penalty weight to allow more growth?
                    # Or just tighter ftol.
                    res_polish = minimize(_objective, params_opt, args=(n, 100.0), method='SLSQP',
                                          options={'maxiter': 100, 'ftol': 1e-12, 'disp': False})
                    if res_polish.success:
                        params_pol = res_polish.x
                        c_pol = params_pol[:2*n].reshape(n, 2)
                        s_pol = np.maximum(params_pol[2*n:], 1e-6)
                        update_best(c_pol, s_pol)
                        
        except Exception:
            pass

    if best_centers is None:
        # Fallback to Config 1
        best_centers = c1
        best_sides = s1

    return best_centers, best_sides

# EVOLVE-BLOCK-END
