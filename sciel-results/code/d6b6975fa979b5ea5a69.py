# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _is_valid(centers, sides, tol=1e-7):
    """Check if the configuration is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
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
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                return False
    return True


def _objective(params, n=10):
    """
    Objective function to minimize (negative of sum of sides).
    params: [s_0, ..., s_9, cx_0, cy_0, ..., cx_9, cy_9]
    """
    sides = params[:n]
    centers = params[n:].reshape(n, 2)
    
    # Penalty for violations
    penalty = 0.0
    tol = 1e-7
    
    # Bounds penalty
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            penalty += 1000.0 * (1.0 - s)
        if cx - s/2 < -tol:
            penalty += 1000.0 * (-tol - (cx - s/2))
        if cx + s/2 > 1 + tol:
            penalty += 1000.0 * ((cx + s/2) - (1 + tol))
        if cy - s/2 < -tol:
            penalty += 1000.0 * (-tol - (cy - s/2))
        if cy + s/2 > 1 + tol:
            penalty += 1000.0 * ((cy + s/2) - (1 + tol))
    
    # Overlap penalty
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            dx = abs(centers[i,0] - centers[j,0])
            dy = abs(centers[i,1] - centers[j,1])
            if dx < t and dy < t:
                # Penalty proportional to how much they overlap
                penalty += 1000.0 * (t - dx) * (t - dy)
    
    # We want to maximize sum of sides, so minimize negative sum
    return -np.sum(sides) + penalty


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Uses a multi-start local optimization approach to find a good packing.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Start with a reasonable initial configuration
    # Use a 3x3 grid plus one extra square
    # Initial guess: mix of sizes
    best_params = None
    best_score = -np.inf
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Config 1: 3x3 grid with 9 squares of side 1/3, plus one smaller
    side1 = 1.0 / 3.0
    params1 = np.zeros(3 * n)
    for i in range(9):
        col = i % 3
        row = i // 3
        params1[i] = side1
        params1[n + i] = (col + 0.5) * side1
        params1[n + n + i] = (row + 0.5) * side1
    # 10th square: small square in a corner
    params1[9] = 0.1
    params1[n + 9] = 0.05
    params1[n + n + 9] = 0.05
    initial_configs.append(params1)
    
    # Config 2: 2x5 grid
    side2 = 1.0 / 5.0
    params2 = np.zeros(3 * n)
    for i in range(10):
        col = i % 5
        row = i // 5
        params2[i] = side2
        params2[n + i] = (col + 0.5) * side2
        params2[n + n + i] = (row + 0.5) * side2
    initial_configs.append(params2)
    
    # Config 3: Randomized perturbation of config 1
    rng = np.random.default_rng(42)
    params3 = params1.copy()
    params3[9] = rng.uniform(0.05, 0.2)
    params3[n + 9] = rng.uniform(0, 0.5)
    params3[n + n + 9] = rng.uniform(0, 0.5)
    initial_configs.append(params3)
    
    # Config 4: Another random perturbation
    rng2 = np.random.default_rng(123)
    params4 = params1.copy()
    params4[9] = rng2.uniform(0.1, 0.3)
    params4[n + 9] = rng2.uniform(0, 1)
    params4[n + n + 9] = rng2.uniform(0, 1)
    initial_configs.append(params4)
    
    for init_params in initial_configs:
        if time.time() > deadline:
            break
        
        # Local optimization using Nelder-Mead
        try:
            result = minimize(
                _objective,
                init_params,
                method='Nelder-Mead',
                args=(n,),
                options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if result.success or result.fun < best_score:
                best_params = result.x
                best_score = result.fun
        except Exception:
            pass
    
    # If optimization didn't find anything good, fall back to grid
    if best_params is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides
    
    # Extract centers and sides from best_params
    sides = best_params[:n].astype(np.float64)
    centers = best_params[n:].reshape(n, 2).astype(np.float64)
    
    # Validate and fix if needed
    if not _is_valid(centers, sides):
        # Try to fix by shrinking sides slightly
        sides_fixed = sides * 0.999
        centers_fixed = centers.copy()
        if _is_valid(centers_fixed, sides_fixed):
            centers, sides = centers_fixed, sides_fixed
        else:
            # Fall back to grid
            k = int(np.ceil(np.sqrt(n)))
            side = 1.0 / k
            idx = np.arange(n)
            cols = idx % k
            rows = idx // k
            cx = (cols + 0.5) * side
            cy = (rows + 0.5) * side
            centers = np.stack([cx, cy], axis=1).astype(np.float64)
            sides = np.full(n, side, dtype=np.float64)
    
    return centers, sides


# EVOLVE-BLOCK-END
