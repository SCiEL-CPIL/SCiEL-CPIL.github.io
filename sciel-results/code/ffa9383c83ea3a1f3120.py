# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 axis-aligned squares inside a unit square.
    Uses multi-start local optimization with a penalty-based objective function.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s safety margin
    
    best_score = -np.inf
    best_x = None
    
    def unpack(x):
        # x is shape (30,): 10 centers (20 dims) + 10 sides
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        return centers, sides

    def objective(x):
        centers, sides = unpack(x)
        
        # Boundary penalty
        # Centers must be in [s/2, 1-s/2]
        # If s is too large, it's impossible. We penalize violation.
        bound_pen = 0.0
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            # Lower bounds
            if cx < s/2: bound_pen += (s/2 - cx)**2
            if cy < s/2: bound_pen += (s/2 - cy)**2
            # Upper bounds
            if cx > 1 - s/2: bound_pen += (cx - (1 - s/2))**2
            if cy > 1 - s/2: bound_pen += (cy - (1 - s/2))**2
        
        # Overlap penalty
        # Penalty if squares i and j overlap in both x and y
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                sx = (sides[i] + sides[j]) / 2.0
                sy = (sides[i] + sides[j]) / 2.0
                # Violation in x
                vx = max(0.0, sx - dx)
                # Violation in y
                vy = max(0.0, sy - dy)
                # Product is zero if separated in either dimension
                overlap_pen += vx * vy
        
        # We want to maximize sum of sides, so minimize -sum(sides) + penalties
        # Use a large weight for penalties to ensure feasibility
        weight = 100.0
        return -np.sum(sides) + weight * (bound_pen + overlap_pen)

    def is_valid(centers, sides, tol=1e-7):
        if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(sides)):
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
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if dx < t and dy < t:
                    return False
        return True

    def score(centers, sides):
        return np.sum(sides)

    # Initial guesses
    inits = []
    
    # 1. Grid initialization (Baseline)
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers_init = np.stack([cx, cy], axis=1)
    sides_init = np.full(n, side)
    inits.append(np.concatenate([centers_init.flatten(), sides_init]))
    
    # 2. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(20):
        c = rng.uniform(0.2, 0.8, size=(n, 2))
        s = rng.uniform(0.1, 0.3, size=n)
        inits.append(np.concatenate([c.flatten(), s]))
        
    # 3. Specific heuristic: One large square, others small
    # Large square in corner
    c_large = np.array([0.25, 0.25])
    s_large = 0.5
    # Fill rest
    c_rest = rng.uniform(0.2, 0.8, size=(n-1, 2))
    s_rest = rng.uniform(0.1, 0.2, size=n-1)
    c = np.vstack([c_large, c_rest])
    s = np.concatenate([[s_large], s_rest])
    inits.append(np.concatenate([c.flatten(), s]))

    # Optimization loop
    for i, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                objective, 
                x0, 
                method='Powell', 
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            x_best = res.x
            centers, sides = unpack(x_best)
            
            # Clip sides to be positive
            sides = np.maximum(sides, 1e-6)
            
            # Check validity
            if is_valid(centers, sides):
                s_val = score(centers, sides)
                if s_val > best_score:
                    best_score = s_val
                    best_x = x_best
        except Exception as e:
            continue

    # If no valid solution found, return baseline
    if best_x is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    centers, sides = unpack(best_x)
    return centers, sides

# EVOLVE-BLOCK-END
