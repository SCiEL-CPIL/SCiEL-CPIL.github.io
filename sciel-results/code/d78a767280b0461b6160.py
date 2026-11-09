# EVOLVE-BLOCK-START
import numpy as np
import time


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


def _local_optimize(params, n=10, max_iter=200, tol=1e-7):
    """
    Simple coordinate descent with line search for local optimization.
    """
    best_params = params.copy()
    best_score = _objective(best_params, n)
    
    for iteration in range(max_iter):
        improved = False
        for dim in range(len(best_params)):
            # Try to improve one coordinate at a time
            for delta in [0.01, 0.005, 0.001, 0.0005, -0.01, -0.005, -0.001, -0.0005]:
                new_params = best_params.copy()
                new_params[dim] += delta
                new_score = _objective(new_params, n)
                if new_score < best_score:
                    best_params = new_params
                    best_score = new_score
                    improved = True
        if not improved:
            break
    
    return best_params, best_score


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.
    
    Uses a multi-start local optimization approach with diverse initial configurations.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    best_params = None
    best_score = -np.inf
    
    # Generate diverse initial configurations
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
    
    # Config 3: Mixed sizes - 4 large, 6 small
    params3 = np.zeros(3 * n)
    # 4 large squares of side 0.25 in a 2x2 grid
    for i in range(4):
        col = i % 2
        row = i // 2
        params3[i] = 0.25
        params3[n + i] = (col + 0.5) * 0.25
        params3[n + n + i] = (row + 0.5) * 0.25
    # 6 small squares of side 0.125 in remaining space
    small_side = 0.125
    positions = [
        (0.5, 0.75), (0.75, 0.75), (1.0, 0.75),  # top row
        (0.5, 1.0), (0.75, 1.0), (1.0, 1.0)      # right column
    ]
    for i in range(6):
        params3[4 + i] = small_side
        params3[n + 4 + i] = positions[i][0]
        params3[n + n + 4 + i] = positions[i][1]
    initial_configs.append(params3)
    
    # Config 4: Randomized perturbation of config 1
    rng = np.random.default_rng(42)
    params4 = params1.copy()
    params4[9] = rng.uniform(0.05, 0.2)
    params4[n + 9] = rng.uniform(0, 0.5)
    params4[n + n + 9] = rng.uniform(0, 0.5)
    initial_configs.append(params4)
    
    # Config 5: Another random perturbation
    rng2 = np.random.default_rng(123)
    params5 = params1.copy()
    params5[9] = rng2.uniform(0.1, 0.3)
    params5[n + 9] = rng2.uniform(0, 1)
    params5[n + n + 9] = rng2.uniform(0, 1)
    initial_configs.append(params5)
    
    # Config 6: 5x2 grid with varying sizes
    params6 = np.zeros(3 * n)
    for i in range(10):
        col = i % 5
        row = i // 5
        # Vary sizes slightly
        s = 0.18 + 0.02 * np.sin(i * 0.7)
        params6[i] = s
        params6[n + i] = (col + 0.5) * 0.2
        params6[n + n + i] = (row + 0.5) * 0.5
    initial_configs.append(params6)
    
    # Config 7: Concentric-like arrangement
    params7 = np.zeros(3 * n)
    # Center square
    params7[0] = 0.2
    params7[n + 0] = 0.5
    params7[n + n + 0] = 0.5
    # 4 squares around center
    positions = [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]
    for i in range(4):
        params7[1 + i] = 0.15
        params7[n + 1 + i] = positions[i][0]
        params7[n + n + 1 + i] = positions[i][1]
    # 5 more squares in corners
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9), (0.5, 0.1)]
    for i in range(5):
        params7[5 + i] = 0.12
        params7[n + 5 + i] = corners[i][0]
        params7[n + n + 5 + i] = corners[i][1]
    initial_configs.append(params7)
    
    # Optimize each initial configuration
    for idx, init_params in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        try:
            # Use coordinate descent for faster local optimization
            opt_params, opt_score = _local_optimize(init_params, n, max_iter=100)
            
            if opt_score < best_score:
                best_params = opt_params
                best_score = opt_score
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
