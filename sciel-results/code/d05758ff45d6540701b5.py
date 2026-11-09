import numpy as np
import time
from scipy.optimize import minimize

def _check_validity(centers, sides, tol=1e-7):
    """
    Check if the configuration is valid.
    Returns True if valid, False otherwise.
    """
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    # Check bounds
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
            
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            cx_i, cy_i = centers[i]
            cx_j, cy_j = centers[j]
            s_i = sides[i]
            s_j = sides[j]
            
            dx = abs(cx_i - cx_j)
            dy = abs(cy_i - cy_j)
            
            t = (s_i + s_j) / 2.0 - tol
            if dx < t and dy < t:
                return False
    return True

def _penalty_and_objective(x, lambda_overlap=1000.0, lambda_bounds=1000.0):
    """
    Objective function for minimization.
    x is a flat array of [cx1, cy1, s1, cx2, cy2, s2, ...]
    We want to maximize sum(s), so minimize -sum(s) + penalties.
    """
    n = 10
    centers = x[0::3].reshape(-1, 2) # Wait, x structure: cx, cy, s
    # Let's define x as [cx1, cy1, s1, cx2, cy2, s2, ...]
    cx = x[0::3]
    cy = x[1::3]
    s = x[2::3]
    
    if np.any(s <= 0):
        return 1e6
    
    # Objective: -sum(s)
    obj = -np.sum(s)
    
    # Penalties
    penalty = 0.0
    
    # Bounds penalty
    for i in range(n):
        if cx[i] - s[i]/2 < 0:
            penalty += lambda_bounds * (cx[i] - s[i]/2)**2
        if cx[i] + s[i]/2 > 1:
            penalty += lambda_bounds * (cx[i] + s[i]/2 - 1)**2
        if cy[i] - s[i]/2 < 0:
            penalty += lambda_bounds * (cy[i] - s[i]/2)**2
        if cy[i] + s[i]/2 > 1:
            penalty += lambda_bounds * (cy[i] + s[i]/2 - 1)**2
            
    # Overlap penalty
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(cx[i] - cx[j])
            dy = abs(cy[i] - cy[j])
            t = (s[i] + s[j]) / 2.0
            
            # If they overlap in both dimensions
            if dx < t and dy < t:
                # Calculate overlap depth
                overlap_x = t - dx
                overlap_y = t - dy
                # The minimum distance to separate is min(overlap_x, overlap_y)
                # We penalize the square of this minimum distance
                min_overlap = min(overlap_x, overlap_y)
                penalty += lambda_overlap * (min_overlap**2)
                
    return obj + penalty

def _generate_initial_guesses():
    guesses = []
    
    # Guess 1: 4x4 Grid
    k = 4
    side = 1.0 / k
    idx = np.arange(10)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    x = np.zeros(30)
    x[0::3] = cx
    x[1::3] = cy
    x[2::3] = side
    guesses.append(x)
    
    # Guess 2: 3x3 Grid (9 squares) + 1 smaller in center? 
    # Actually, just start with 3x3 grid side 0.33 and one small square 0.1 in center
    k = 3
    side = 1.0 / k
    idx = np.arange(9)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    # Add 10th square in center, but it will overlap. 
    # Let's shrink the 9 squares to 0.3 and put 10th in center 0.2
    side_9 = 0.3
    cx = (cols + 0.5) * side_9
    cy = (rows + 0.5) * side_9
    x = np.zeros(30)
    x[0::3] = cx
    x[1::3] = cy
    x[2::3] = side_9
    # 10th square
    x[27] = 0.5
    x[28] = 0.5
    x[29] = 0.2
    guesses.append(x)
    
    # Guess 3: Random
    np.random.seed(42)
    for _ in range(5):
        cx = np.random.uniform(0.1, 0.9, 10)
        cy = np.random.uniform(0.1, 0.9, 10)
        s = np.random.uniform(0.1, 0.3, 10)
        x = np.zeros(30)
        x[0::3] = cx
        x[1::3] = cy
        x[2::3] = s
        guesses.append(x)
        
    return guesses

def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place 10 squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths.
    """
    start_time = time.time()
    deadline = start_time + 500  # 500 seconds limit to be safe
    
    best_score = -np.inf
    best_x = None
    best_centers = None
    best_sides = None
    
    guesses = _generate_initial_guesses()
    
    for guess in guesses:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_penalty_and_objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            x_opt = res.x
            cx = x_opt[0::3]
            cy = x_opt[1::3]
            s = x_opt[2::3]
            
            # Ensure sides are positive
            s = np.maximum(s, 1e-6)
            
            centers = np.stack([cx, cy], axis=1)
            
            if _check_validity(centers, s):
                score = np.sum(s)
                if score > best_score:
                    best_score = score
                    best_x = x_opt
                    best_centers = centers
                    best_sides = s
                    
        except Exception:
            continue
            
    if best_centers is None:
        # Fallback to baseline
        k = 4
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(10, side, dtype=np.float64)
        return centers, sides
        
    return best_centers, best_sides

# EVOLVE-BLOCK-END
