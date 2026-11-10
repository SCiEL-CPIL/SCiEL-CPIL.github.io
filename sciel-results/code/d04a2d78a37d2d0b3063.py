import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    """Check if the packing is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
            
    # Check no overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_dx = (sides[i] + sides[j])/2 - tol
            min_dy = (sides[i] + sides[j])/2 - tol
            if dx < min_dx and dy < min_dy:
                return False
    return True

def _vectorized_penalty(centers, sides, beta=50.0):
    """
    Vectorized smooth penalty for overlaps and boundaries.
    Returns a scalar penalty.
    """
    n = len(sides)
    penalty = 0.0
    
    # Overlap penalty
    # Create pairwise difference matrices
    # centers shape (n, 2)
    # dx[i, j] = |cx_i - cx_j|
    cx = centers[:, 0]
    cy = centers[:, 1]
    
    # Efficiently compute pairwise distances
    # We only need i < j
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(cx[i] - cx[j])
            dy = abs(cy[i] - cy[j])
            t = (sides[i] + sides[j]) / 2.0
            
            gap_x = dx - t
            gap_y = dy - t
            
            # Soft min of gap_x and gap_y
            # We want penalty if min(gap_x, gap_y) < 0
            # Let m = min(gap_x, gap_y)
            # If m >= 0, penalty is 0 (or close to 0)
            # If m < 0, penalty is large
            # Use soft min: -1/beta * log(exp(-beta*gap_x) + exp(-beta*gap_y))
            # This is approx min(gap_x, gap_y) for large beta
            # We want penalty = max(0, -min(gap_x, gap_y))^2 ?
            # Or just soft_min itself if negative.
            
            # Let's use a smooth approximation of max(0, -min(gap_x, gap_y))
            # Let v = -min(gap_x, gap_y)
            # v = max(-gap_x, -gap_y)
            # Soft max: 1/beta * log(exp(beta*(-gap_x)) + exp(beta*(-gap_y)))
            
            # To avoid overflow, subtract max
            a = -gap_x
            b = -gap_y
            m = np.maximum(a, b)
            if m > 500.0 / beta:
                continue
            
            # Soft max of a and b
            val = m + np.log(np.exp(a - m) + np.exp(b - m)) / beta
            # val is approx max(a, b) = -min(gap_x, gap_y)
            # If val > 0, there is a violation.
            # Penalty is val^2 (quadratic) or val (linear)
            # Quadratic is smoother at 0
            penalty += val * val
            
    # Boundary penalties
    for i in range(n):
        s = sides[i]
        # Gaps to boundaries
        g_left = centers[i, 0] - s/2
        g_right = 1 - centers[i, 0] - s/2
        g_bottom = centers[i, 1] - s/2
        g_top = 1 - centers[i, 1] - s/2
        
        # We want min(g_left, g_right, g_bottom, g_top) >= 0
        # Let v = -min(gaps) = max(-g_left, -g_right, -g_bottom, -g_top)
        a = -g_left
        b = -g_right
        c = -g_bottom
        d = -g_top
        
        m = np.maximum(np.maximum(a, b), np.maximum(c, d))
        if m > 500.0 / beta:
            continue
        
        val = m + np.log(np.exp(a - m) + np.exp(b - m) + np.exp(c - m) + np.exp(d - m)) / beta
        penalty += val * val
        
    return penalty

def _objective(centers, sides):
    """Return sum of sides if valid, else a large penalty."""
    n = len(sides)
    tol = 1e-7
    
    # Check bounds
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return -1e6
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return -1e6
        if s <= 0:
            return -1e6
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_dx = (sides[i] + sides[j])/2 - tol
            min_dy = (sides[i] + sides[j])/2 - tol
            if dx < min_dx and dy < min_dy:
                return -1e6
    
    return np.sum(sides)

def _generate_initial_config_grid():
    """Generate a grid-based initial config."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 3x3 grid of 1/3 squares, plus 1 extra
    # Actually, let's just do a 4x3 grid? No, 10 squares.
    # Let's do a 2x5 grid of 0.2 x 0.5? No, squares.
    # 2 rows of 5 squares of side 0.2.
    # Sum = 10 * 0.2 = 2.0.
    # Better: 1 row of 5 squares of side 0.2 (height 0.2), 1 row of 5 squares of side 0.2?
    # Let's use the 3.0 packing as a baseline.
    sides[0] = 0.5
    sides[1] = 0.5
    centers[0, 0] = 0.25
    centers[0, 1] = 0.25
    centers[1, 0] = 0.75
    centers[1, 1] = 0.25
    
    sides[2] = 0.25
    centers[2, 0] = 0.125
    centers[2, 1] = 0.625
    
    sides[3] = 0.25
    centers[3, 0] = 0.375
    centers[3, 1] = 0.625
    
    sides[4] = 0.25
    centers[4, 0] = 0.125
    centers[4, 1] = 0.875
    
    sides[5] = 0.25
    centers[5, 0] = 0.375
    centers[5, 1] = 0.875
    
    sides[6] = 0.25
    centers[6, 0] = 0.625
    centers[6, 1] = 0.625
    
    sides[7] = 0.25
    centers[7, 0] = 0.875
    centers[7, 1] = 0.625
    
    sides[8] = 0.25
    centers[8, 0] = 0.625
    centers[8, 1] = 0.875
    
    sides[9] = 0.25
    centers[9, 0] = 0.875
    centers[9, 1] = 0.875
    
    return centers, sides

def _generate_random_config(rng):
    """Generate a random valid-ish config."""
    n = 10
    centers = np.random.uniform(0.1, 0.9, (n, 2))
    sides = np.random.uniform(0.1, 0.3, n)
    
    # Try to make it valid by shrinking
    for _ in range(100):
        if _is_valid(centers, sides):
            break
        sides *= 0.9
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], sides[i]/2, 1 - sides[i]/2)
            centers[i, 1] = np.clip(centers[i, 1], sides[i]/2, 1 - sides[i]/2)
            
    return centers, sides

def _local_search(centers, sides, time_limit=10.0, beta_start=10.0, beta_end=100.0):
    """
    Uses scipy.optimize.minimize with a smooth penalty function.
    """
    start_time = time.time()
    n = len(sides)
    
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = sides[i]
    
    best_x = x0.copy()
    best_score = _objective(centers, sides)
    
    if best_score < 0:
        return centers, sides, best_score
    
    def unpack(x):
        c = np.zeros((n, 2))
        s = np.zeros(n)
        for i in range(n):
            c[i, 0] = x[3*i]
            c[i, 1] = x[3*i+1]
            s[i] = x[3*i+2]
        return c, s
    
    betas = np.linspace(beta_start, beta_end, 5)
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        def obj(x, b=beta):
            if time.time() - start_time > time_limit:
                return 1e6
            c, s = unpack(x)
            if np.any(s < 0.01) or np.any(s > 1.0):
                return 1e6
            pen = _vectorized_penalty(c, s, beta=b)
            return -np.sum(s) + pen
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 200, 'disp': False})
            if res.success or res.fun < obj(x0):
                x0 = res.x
        except Exception:
            pass
        
        c, s = unpack(x0)
        s = np.clip(s, 0.01, 1.0)
        for i in range(n):
            c[i, 0] = np.clip(c[i, 0], s[i]/2, 1 - s[i]/2)
            c[i, 1] = np.clip(c[i, 1], s[i]/2, 1 - s[i]/2)
        
        score = _objective(c, s)
        if score > best_score:
            best_score = score
            best_x = x0.copy()
            
    c, s = unpack(best_x)
    return c, s, best_score

def squares_in_square10() -> tuple:
    """
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    """
    deadline = time.time() + 600
    rng = np.random.RandomState(42)
    
    # Start with known 3.0 packing
    centers, sides = _generate_initial_config_grid()
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _objective(best_centers, best_sides)
    
    # Multi-start local search
    for attempt in range(10):
        if time.time() > deadline - 60:
            break
        
        if attempt == 0:
            c_init, s_init = centers.copy(), sides.copy()
        elif attempt == 1:
            c_init, s_init = _generate_random_config(rng)
        else:
            # Perturb best solution
            c_init = best_centers + rng.normal(0, 0.05, (10, 2))
            s_init = best_sides * (1 + rng.normal(0, 0.1, 10))
            s_init = np.clip(s_init, 0.05, 0.6)
            for i in range(10):
                c_init[i, 0] = np.clip(c_init[i, 0], s_init[i]/2, 1 - s_init[i]/2)
                c_init[i, 1] = np.clip(c_init[i, 1], s_init[i]/2, 1 - s_init[i]/2)
        
        time_left = deadline - time.time() - 60
        if time_left < 5:
            break
            
        c_new, s_new, score_new = _local_search(
            c_init, s_init, 
            time_limit=min(30.0, time_left), 
            beta_start=5.0, 
            beta_end=100.0
        )
        
        if score_new > best_score:
            best_score = score_new
            best_centers = c_new.copy()
            best_sides = s_new.copy()
            
    # Final polish
    if time.time() < deadline - 10:
        c_new, s_new, score_new = _local_search(
            best_centers, best_sides,
            time_limit=min(30.0, deadline - time.time() - 10),
            beta_start=50.0,
            beta_end=200.0
        )
        if score_new > best_score:
            best_score = score_new
            best_centers = c_new.copy()
            best_sides = s_new.copy()
    
    if not _is_valid(best_centers, best_sides):
        centers, sides = _generate_initial_config_grid()
        return centers, sides
    
    return best_centers, best_sides
