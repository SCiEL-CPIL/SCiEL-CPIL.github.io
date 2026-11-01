"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_initialization(n):
    """
    Generate a hexagonal close-packing initialization for n circles in [0,1]^2.
    Returns an array of shape (n, 3) with columns (x, y, radius).
    """
    # Hexagonal lattice: rows offset by half spacing
    # We'll create a grid of points and pick the best n
    
    # Start with a reasonable number of points, then select n
    # For n=32, a 6x6 grid gives 36 points, we pick 32
    # Or we can use a smarter approach: generate many candidate positions
    # and pick the n that maximize minimum distance (or just use a regular hex grid)
    
    # Let's use a hexagonal arrangement
    # Row spacing: dy, column spacing: dx, with dy = dx * sqrt(3)/2
    # For a hex grid, odd rows are offset by dx/2
    
    # We want to fit about n circles. Let's try different grid sizes
    # and pick the one that gives a good packing
    
    best_positions = None
    best_min_dist = 0
    
    for cols in range(5, 10):
        for rows in range(5, 10):
            if rows * cols < n:
                continue
            
            # Hexagonal spacing
            dx = 1.0 / (cols - 0.5)  # approximate
            dy = dx * np.sqrt(3) / 2
            
            # Check if it fits vertically
            total_height = (rows - 1) * dy + dy  # plus margins
            if total_height > 1.0 + dy:  # too tall
                continue
            
            positions = []
            for i in range(rows):
                for j in range(cols):
                    x = (j + 0.25 + (0.5 if i % 2 == 1 else 0)) * dx
                    y = (i + 0.5) * dy + (1.0 - (rows - 1) * dy) / 2
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        positions.append((x, y))
            
            if len(positions) < n:
                continue
            
            # Compute minimum distance between all pairs
            positions = np.array(positions[:n])
            min_dist = np.inf
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    d = np.linalg.norm(positions[i] - positions[j])
                    min_dist = min(min_dist, d)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_positions = positions
            
            if len(positions) >= n:
                break
    
    if best_positions is None or len(best_positions) < n:
        # Fallback to square grid
        side = int(np.ceil(np.sqrt(n)))
        positions = np.array([
            ((i + 0.5) / side, (j + 0.5) / side)
            for i in range(side) for j in range(side)
        ][:n])
    else:
        positions = best_positions[:n]
    
    # Initial radius: half the minimum distance to any other circle or boundary
    min_dist_to_boundary = np.min(np.stack([
        positions[:, 0],
        positions[:, 1],
        1 - positions[:, 0],
        1 - positions[:, 1]
    ], axis=1), axis=1)
    
    min_pairwise = np.array([
        np.min(np.linalg.norm(positions - positions[i], axis=1))
        for i in range(len(positions))
    ])
    # Exclude self-distance
    for i in range(len(positions)):
        min_pairwise[i] = np.min(np.delete(min_pairwise, i))
    
    radii = 0.5 * np.minimum(min_dist_to_boundary, min_pairwise)
    radii = np.maximum(radii, 0.01)  # minimum radius
    
    return np.column_stack([positions, radii])


def _softmin(x, beta):
    """Soft minimum using log-sum-exp trick."""
    # -log-sum-exp(-x * beta) / beta approximates min(x) as beta -> inf
    x_max = np.max(x)
    return x_max - np.log(np.sum(np.exp(beta * (x - x_max)))) / beta


def _objective(params, n):
    """
    Objective function to minimize.
    params: array of shape (3*n,) -> [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    Returns: negative sum of radii + penalty for violations
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Constraints:
    # 1. Circles inside [0,1]^2: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    # 2. Non-overlapping: dist(i,j) >= r_i + r_j for all i != j
    # 3. r > 0
    
    penalty = 0.0
    beta = 100.0  # soft-min parameter
    
    # Boundary constraints (soft)
    # x - r >= 0 -> x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    
    boundary_violations = np.concatenate([
        x - r,      # left
        1 - x - r,  # right
        y - r,      # bottom
        1 - y - r   # top
    ])
    
    # Penalty for boundary violations using soft-min
    # We want all boundary_violations >= 0
    # Penalty = softmin(boundary_violations, beta) if min < 0
    min_boundary = np.min(boundary_violations)
    if min_boundary < 0:
        penalty += -min_boundary * 1000
    
    # Pairwise non-overlap constraints
    # dist(i,j) - r_i - r_j >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            violation = dist - r[i] - r[j]
            if violation < 0:
                penalty += -violation * 1000
    
    # Maximize sum of radii -> minimize -sum(r)
    return -np.sum(r) + penalty


def _objective_vectorized(params, n):
    """
    Vectorized objective function for better performance.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary constraints
    boundary_violations = np.concatenate([
        x - r,
        1 - x - r,
        y - r,
        1 - y - r
    ])
    min_boundary = np.min(boundary_violations)
    if min_boundary < 0:
        penalty += (-min_boundary) ** 2 * 1000 + (-min_boundary) * 100
    
    # Pairwise non-overlap: vectorized
    # Compute all pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # r_i + r_j matrix
    r_sum = r[:, None] + r[None, :]
    
    # violations = dist - r_sum, only for i < j
    # Set diagonal to 0 to ignore self
    np.fill_diagonal(dist, 0)
    np.fill_diagonal(r_sum, 0)
    
    # Only consider upper triangle
    iu = np.triu_indices(n, 1)
    violations = dist[iu] - r_sum[iu]
    
    min_violation = np.min(violations)
    if min_violation < 0:
        penalty += (-min_violation) ** 2 * 1000 + (-min_violation) * 100
    
    # Additional quadratic penalty for all violations
    neg_violations = np.maximum(0, -violations)
    penalty += np.sum(neg_violations ** 2) * 100
    
    neg_boundary = np.maximum(0, -boundary_violations)
    penalty += np.sum(neg_boundary ** 2) * 100
    
    return -np.sum(r) + penalty


def _is_valid(result, n):
    """Check if the result is valid (circles inside unit square, non-overlapping)."""
    if result.shape != (n, 3):
        return False
    
    x = result[:, 0]
    y = result[:, 1]
    r = result[:, 2]
    
    # Check inside unit square
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    if np.any(r < -1e-9):
        return False
    
    # Check non-overlapping
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return False
    
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Step 1: Hexagonal initialization
    result = _hex_initialization(N)
    best_score = np.sum(result[:, 2])
    best_result = result.copy()
    
    # Step 2: Optimize using L-BFGS-B
    # Parameters: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    x0 = result[:, 0].copy()
    y0 = result[:, 1].copy()
    r0 = result[:, 2].copy()
    
    params0 = np.zeros(3 * N)
    params0[0::3] = x0
    params0[1::3] = y0
    params0[2::3] = r0
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((1e-6, 1 - 1e-6))  # x
        bounds.append((1e-6, 1 - 1e-6))  # y
        bounds.append((1e-4, 0.5))        # r
    
    # Use L-BFGS-B with the vectorized objective
    try:
        res = minimize(
            _objective_vectorized,
            params0,
            args=(N,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
        )
        
        if time.time() < deadline:
            params_opt = res.x
            x_opt = params_opt[0::3]
            y_opt = params_opt[1::3]
            r_opt = params_opt[2::3]
            
            # Clip to valid range
            x_opt = np.clip(x_opt, 1e-6, 1 - 1e-6)
            y_opt = np.clip(y_opt, 1e-6, 1 - 1e-6)
            r_opt = np.clip(r_opt, 1e-4, 0.5)
            
            result_opt = np.column_stack([x_opt, y_opt, r_opt])
            
            # Validate and fix if needed
            if _is_valid(result_opt, N):
                score_opt = np.sum(r_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_result = result_opt.copy()
            else:
                # Try to fix by reducing radii
                for scale in [0.9, 0.8, 0.7, 0.6, 0.5]:
                    r_fixed = r_opt * scale
                    result_fixed = np.column_stack([x_opt, y_opt, r_fixed])
                    if _is_valid(result_fixed, N):
                        score_fixed = np.sum(r_fixed)
                        if score_fixed > best_score:
                            best_score = score_fixed
                            best_result = result_fixed.copy()
                        break
    except Exception as e:
        pass
    
    # Step 3: Polish pass with tighter tolerance if time remains
    if time.time() < deadline - 20:
        try:
            params0 = np.zeros(3 * N)
            params0[0::3] = best_result[:, 0]
            params0[1::3] = best_result[:, 1]
            params0[2::3] = best_result[:, 2]
            
            res2 = minimize(
                _objective_vectorized,
                params0,
                args=(N,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 20000}
            )
            
            if time.time() < deadline:
                params_opt = res2.x
                x_opt = params_opt[0::3]
                y_opt = params_opt[1::3]
                r_opt = params_opt[2::3]
                
                x_opt = np.clip(x_opt, 1e-6, 1 - 1e-6)
                y_opt = np.clip(y_opt, 1e-6, 1 - 1e-6)
                r_opt = np.clip(r_opt, 1e-4, 0.5)
                
                result_opt = np.column_stack([x_opt, y_opt, r_opt])
                
                if _is_valid(result_opt, N):
                    score_opt = np.sum(r_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_result = result_opt.copy()
        except Exception as e:
            pass
    
    # Step 4: Final validation and safety
    if not _is_valid(best_result, N):
        # Fall back to a safe grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
