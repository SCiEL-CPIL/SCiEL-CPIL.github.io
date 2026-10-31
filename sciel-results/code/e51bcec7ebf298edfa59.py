"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 26

# EVOLVE-BLOCK-START

def _pack_circle_grid(n):
    """
    Generate a reasonable initial packing for n circles in [0,1]^2.
    Uses a grid layout with some jitter to help escape symmetries.
    Returns array of shape (n, 3) with (x, y, r).
    """
    # Estimate grid dimensions
    # For n=26, a 5x6 grid works well
    # Try to find good grid dimensions
    best_side1, best_side2 = 1, 1
    best_min_dim = 0
    for s1 in range(1, int(np.ceil(np.sqrt(n))) + 2):
        s2 = int(np.ceil(n / s1))
        if s1 * s2 >= n:
            min_dim = min(s1, s2)
            if min_dim > best_min_dim:
                best_min_dim = min_dim
                best_side1, best_side2 = s1, s2
    
    side1, side2 = best_side1, best_side2
    
    positions = []
    count = 0
    for i in range(side1):
        for j in range(side2):
            if count >= n:
                break
            x = (i + 0.5) / side1
            y = (j + 0.5) / side2
            # Add small random jitter
            x += random.uniform(-0.01, 0.01)
            y += random.uniform(-0.01, 0.01)
            # Clamp to [0,1]
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            positions.append((x, y))
            count += 1
    
    positions = np.array(positions[:n])
    
    # Compute initial radii based on nearest neighbor distances
    radii = []
    for i in range(n):
        min_dist = 1.0  # Max possible distance
        for j in range(n):
            if i == j:
                continue
            d = np.sqrt((positions[i, 0] - positions[j, 0])**2 + (positions[i, 1] - positions[j, 1])**2)
            min_dist = min(min_dist, d)
        # Also consider distance to walls
        wall_dist = min(positions[i, 0], 1.0 - positions[i, 0], positions[i, 1], 1.0 - positions[i, 1])
        min_dist = min(min_dist, 2.0 * wall_dist)  # Radius limited by wall distance
        r = 0.5 * min_dist * 0.95  # Slightly smaller to ensure no overlap
        radii.append(r)
    
    radii = np.array(radii)
    
    return np.column_stack([positions, radii])


def _objective_and_constraints(x, n):
    """
    x: array of shape (2n,) containing [x1, y1, x2, y2, ..., xn, yn]
    Returns: objective value (negative sum of radii, since we minimize)
    and constraint violations.
    """
    pos = x.reshape(n, 2)
    
    # Compute radii for each circle
    # r_i = min over j != i of 0.5 * dist(i,j), and also min(pos_i, 1-pos_i) for walls
    radii = np.zeros(n)
    for i in range(n):
        min_dist = np.inf
        for j in range(n):
            if i == j:
                continue
            d = np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2)
            min_dist = min(min_dist, d)
        # Wall constraints
        wall_dist = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        min_dist = min(min_dist, 2.0 * wall_dist)
        radii[i] = 0.5 * min_dist
    
    # Objective: maximize sum of radii -> minimize negative sum
    obj = -np.sum(radii)
    
    return obj, radii, pos


def _smooth_objective(x, n, beta=50.0):
    """
    Smooth surrogate objective using log-sum-exp for min distance.
    """
    pos = x.reshape(n, 2)
    
    # For each circle, compute smooth min distance to other circles and walls
    smooth_radii = np.zeros(n)
    for i in range(n):
        # Distances to other circles
        dists = []
        for j in range(n):
            if i == j:
                continue
            d = np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2)
            dists.append(d)
        
        # Wall distances (distance to wall, radius = distance to wall)
        # Wall constraint: x >= r, x <= 1-r, y >= r, y <= 1-r
        # So r <= min(x, 1-x, y, 1-y)
        wall_dists = [pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1]]
        dists.extend(wall_dists)
        
        dists = np.array(dists)
        
        # Smooth min using log-sum-exp
        # min(d) ≈ -(1/beta) * log(sum(exp(-beta * d)))
        # Use numerical stability
        max_d = np.max(dists)
        exp_terms = np.exp(-beta * (dists - max_d))
        smooth_min = max_d - (1.0 / beta) * np.log(np.sum(exp_terms))
        
        smooth_radii[i] = 0.5 * smooth_min
    
    obj = -np.sum(smooth_radii)
    return obj


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_result = None
    best_score = -np.inf
    
    # Generate multiple initial solutions and optimize each
    n_random_starts = 10
    attempts = 0
    
    while attempts < n_random_starts:
        if time.time() > deadline - 10.0:
            break
        
        attempts += 1
        random.seed(attempts * 12345 + 67890)
        np.random.seed(attempts * 12345 + 67890)
        
        # Generate initial solution
        x0_arr = _pack_circle_grid(N)
        x0 = x0_arr[:, :2].flatten()
        
        # Optimize using smooth objective
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(N, 50.0),
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
        except Exception:
            continue
        
        if time.time() > deadline - 5.0:
            break
        
        # Evaluate the result
        pos = result.x.reshape(N, 2)
        
        # Compute exact radii
        radii = np.zeros(N)
        valid = True
        for i in range(N):
            min_dist = np.inf
            for j in range(N):
                if i == j:
                    continue
                d = np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2)
                min_dist = min(min_dist, d)
            wall_dist = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            min_dist = min(min_dist, 2.0 * wall_dist)
            radii[i] = 0.5 * min_dist
        
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, radii])
        
        # Polish with higher precision if time allows
        if time.time() < deadline - 15.0:
            try:
                # Do a second pass with tighter tolerance
                result2 = minimize(
                    _smooth_objective,
                    result.x,
                    args=(N, 200.0),
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * (2 * N),
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                pos2 = result2.x.reshape(N, 2)
                radii2 = np.zeros(N)
                for i in range(N):
                    min_dist = np.inf
                    for j in range(N):
                        if i == j:
                            continue
                        d = np.sqrt((pos2[i, 0] - pos2[j, 0])**2 + (pos2[i, 1] - pos2[j, 1])**2)
                        min_dist = min(min_dist, d)
                    wall_dist = min(pos2[i, 0], 1.0 - pos2[i, 0], pos2[i, 1], 1.0 - pos2[i, 1])
                    min_dist = min(min_dist, 2.0 * wall_dist)
                    radii2[i] = 0.5 * min_dist
                
                score2 = np.sum(radii2)
                if score2 > best_score:
                    best_score = score2
                    best_result = np.column_stack([pos2, radii2])
            except Exception:
                pass
    
    # Fallback to grid if no valid result
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    # Final validation and clamping
    result = best_result.copy()
    result[:, 0] = np.clip(result[:, 0], 0.0, 1.0)
    result[:, 1] = np.clip(result[:, 1], 0.0, 1.0)
    
    # Ensure radii are non-negative and circles are inside
    for i in range(N):
        x, y, r = result[i]
        # Clamp radius to fit in square
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        result[i, 2] = min(r, max_r)
    
    # Verify disjointness and adjust if needed
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((result[i, 0] - result[j, 0])**2 + (result[i, 1] - result[j, 1])**2)
            max_overlap_r = 0.5 * d
            if result[i, 2] + result[j, 2] > d + 1e-9:
                # Scale down both radii
                scale = d / (result[i, 2] + result[j, 2])
                result[i, 2] *= scale
                result[j, 2] *= scale
    
    # Final wall check
    for i in range(N):
        x, y, r = result[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if r > max_r:
            result[i, 2] = max_r
    
    return result

# EVOLVE-BLOCK-END
