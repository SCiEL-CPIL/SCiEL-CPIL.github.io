"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """
    Compute the score (largest common radius) for a set of centers in [-0.5, 0.5]^2.
    Score = min(
        min_{i<j} ||c_i - c_j|| / 2,
        min_i min_k (0.5 - |c_ik|)
    )
    Returns 0.0 if any center is outside or if there are invalid points.
    """
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds strictly
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    
    # Clip to bounds to be safe for calculation, though they should be inside
    c = np.clip(centers, -0.5, 0.5)
    
    # Distance to walls
    wall_dists = np.minimum(0.5 - np.abs(c), 0.5)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # For N=200, 200x200 matrix is 40,000 elements. Very fast.
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def _soft_min(x, alpha):
    """
    Numerically stable soft-min approximation.
    min(x) ~ -1/alpha * log(sum(exp(-alpha * x)))
    """
    x = np.asarray(x)
    m = np.min(x)
    # If alpha is very large, exp(-alpha * (x - m)) can underflow for large differences,
    # but since we subtract m, the max exponent is 0.
    # The sum will be at least 1 (from the min term).
    # We use the shift to avoid overflow in exp.
    # log(sum(exp(-alpha * (x - m)))) = log(sum(exp(-alpha * x + alpha * m)))
    # = alpha * m + log(sum(exp(-alpha * x)))
    # So -1/alpha * (alpha * m + log(...)) = -m - 1/alpha * log(sum(exp(-alpha * x)))
    # Wait, the formula is:
    # S = -1/alpha * log(sum(exp(-alpha * x_i)))
    # Let x_min = min(x_i).
    # sum(exp(-alpha * x_i)) = exp(-alpha * x_min) * sum(exp(-alpha * (x_i - x_min)))
    # log(sum(...)) = -alpha * x_min + log(sum(exp(-alpha * (x_i - x_min))))
    # S = -1/alpha * (-alpha * x_min + log(sum(exp(-alpha * (x_i - x_min)))))
    # S = x_min - 1/alpha * log(sum(exp(-alpha * (x_i - x_min))))
    
    # Calculate sum(exp(-alpha * (x - m)))
    # x - m >= 0, so -alpha * (x - m) <= 0.
    # exp(val) <= 1. No overflow.
    # Underflow is fine (becomes 0).
    shifted = x - m
    exp_terms = np.exp(-alpha * shifted)
    sum_exp = np.sum(exp_terms)
    # log(sum_exp) is safe.
    return m - (1.0 / alpha) * np.log(sum_exp)

def _get_constraints(centers):
    """
    Returns a flat array of all constraint values:
    - d_ij / 2 for all i < j
    - 0.5 - |c_ik| for all i, k
    """
    c = centers.reshape(N, 2)
    
    # Wall constraints
    walls = 0.5 - np.abs(c)
    wall_constraints = walls.flatten()
    
    # Pair constraints
    # Use broadcasting
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # We only need i < j
    # Create a mask for upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_dists = dists[mask]
    pair_constraints = pair_dists / 2.0
    
    return np.concatenate([wall_constraints, pair_constraints])

def _soft_min_objective(x, alpha):
    """
    Objective function for minimization.
    We want to MAXIMIZE the min constraint.
    So we MINIMIZE -min(constraint).
    Using soft-min: minimize -soft_min(constraints, alpha).
    """
    centers = x.reshape(N, 2)
    # Enforce bounds softly or hard? L-BFGS-B handles bounds.
    # But for safety in calculation, clip? No, L-BFGS-B respects bounds.
    # However, if it steps slightly out due to numerical error, score becomes 0.
    # We will rely on bounds in minimize.
    
    constraints = _get_constraints(centers)
    # We want to maximize min(constraints)
    # So minimize -min(constraints)
    # Approx: -soft_min(constraints, alpha)
    return -_soft_min(constraints, alpha)

def _generate_hex_init():
    """
    Generate an initial hexagonal packing.
    """
    # Target N=200
    # Hex grid spacing s
    # Area per point in hex grid is s^2 * sqrt(3)/2
    # Total Area = 1.0
    # N * s^2 * sqrt(3)/2 = 1.0  => s = sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate points in a grid larger than the square
    # Square is [-0.5, 0.5]
    # We need to cover this area.
    
    points = []
    # y range
    dy = s * np.sqrt(3.0) / 2.0
    # Start from -0.6 to 0.6 to ensure coverage
    y = -0.6
    row = 0
    while y <= 0.6:
        # x offset for alternating rows
        offset = 0.0 if row % 2 == 0 else s / 2.0
        x = -0.6 + offset
        while x <= 0.6:
            points.append((x, y))
            x += s
        y += dy
        row += 1
    
    points = np.array(points)
    
    # Filter points inside the square [-0.5, 0.5]
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # If we have more than N points, remove the ones with the worst local packing
    if len(points) > N:
        # Calculate min distance to neighbor for each point
        # This is O(M^2) where M is number of points. M is approx 200-300.
        # 300^2 = 90,000. Fast.
        min_dists = np.full(len(points), np.inf)
        for i in range(len(points)):
            dists = np.linalg.norm(points[i] - points, axis=1)
            dists[i] = np.inf
            min_dists[i] = np.min(dists)
        
        # Also consider wall distance?
        # Let's just use min neighbor distance for now.
        # Sort indices by min_dist, remove the smallest ones
        idx = np.argsort(min_dists)
        # Keep the largest N
        keep_idx = idx[N:]
        points = points[keep_idx]
        
    # If we have fewer than N points (unlikely with this s), we might need to add
    # But for N=200, s is small enough that we usually have >200 points in the square.
    # If len(points) < N, we just pad with random points? Or just return what we have?
    # The task requires exactly N.
    if len(points) < N:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]
        
    # Jitter slightly to break symmetry
    points += np.random.normal(0, 0.001, points.shape)
    points = np.clip(points, -0.5, 0.5)
    
    return points

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit (evaluator kills at 150s, but keep margin)
    
    # 1. Initialization
    centers = _generate_hex_init()
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # 2. Optimization with Soft-Min Annealing
    # Start with low alpha (smooth) and increase (sharp)
    alphas = [5, 10, 20, 40, 80, 160, 320, 640]
    
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for alpha in alphas:
        if time.time() - start_time > deadline - 10:
            break
            
        def objective(x):
            return _soft_min_objective(x, alpha)
            
        try:
            # Use L-BFGS-B for fast local optimization
            # maxiter can be high, but we stop if time runs out
            result = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            new_centers = result.x.reshape(N, 2)
            new_score = _compute_score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
                x0 = new_centers.flatten()
            else:
                # If it didn't improve, keep the previous x0 for next alpha?
                # Or maybe the step was too big?
                # L-BFGS-B should be stable.
                pass
                
        except Exception as e:
            # If optimization fails, continue to next alpha
            pass
            
    # 3. Final Polish with Nelder-Mead on Exact Objective
    if time.time() - start_time < deadline - 5:
        x0 = best_centers.flatten()
        
        def exact_objective(x):
            c = x.reshape(N, 2)
            # Penalize out of bounds
            if np.any(c < -0.5) or np.any(c > 0.5):
                return 100.0
            return -_compute_score(c)
            
        try:
            # Nelder-Mead is good for final refinement of non-smooth functions
            result = minimize(
                exact_objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            new_centers = result.x.reshape(N, 2)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            new_score = _compute_score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers
                
        except Exception:
            pass
            
    return best_centers

# EVOLVE-BLOCK-END
