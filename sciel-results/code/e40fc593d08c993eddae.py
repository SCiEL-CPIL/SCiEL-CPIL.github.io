"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import sys

N = 26

# EVOLVE-BLOCK-START

def _penalty_function(x, penalty_weight=1000.0):
    """
    Objective function: minimize negative sum of radii + penalty for violations.
    x is a flattened array of [x_1, y_1, ..., x_N, y_N, r_1, ..., r_N]
    
    We use a smooth penalty for overlaps and boundaries.
    Penalty = sum( (violation)^2 ) for all constraints.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    # Ensure radii are positive for calculation stability
    radii = np.maximum(radii, 1e-8)
    
    penalty = 0.0
    
    # 1. Boundary Constraints:
    # Circle must be inside [0, 1] x [0, 1]
    # x_i - r_i >= 0  => violation = max(0, r_i - x_i)
    # x_i + r_i <= 1  => violation = max(0, x_i + r_i - 1)
    # y_i - r_i >= 0  => violation = max(0, r_i - y_i)
    # y_i + r_i <= 1  => violation = max(0, y_i + r_i - 1)
    
    viol_left = np.maximum(0.0, radii - pos[:, 0])
    viol_right = np.maximum(0.0, pos[:, 0] + radii - 1.0)
    viol_bottom = np.maximum(0.0, radii - pos[:, 1])
    viol_top = np.maximum(0.0, pos[:, 1] + radii - 1.0)
    
    penalty += np.sum(viol_left**2) + np.sum(viol_right**2) + np.sum(viol_bottom**2) + np.sum(viol_top**2)
    
    # 2. Overlap Constraints:
    # dist(i, j) >= r_i + r_j
    # violation = max(0, (r_i + r_j)^2 - dist(i, j)^2)
    # Using squared distances avoids sqrt in the main loop, but we need to be careful.
    # Let d_sq = (x_i - x_j)^2 + (y_i - y_j)^2
    # Let r_sum = r_i + r_j
    # violation = max(0, r_sum^2 - d_sq)
    
    # Vectorized pairwise distances
    # dx[i, j] = x_i - x_j
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    d_sq = dx * dx + dy * dy
    
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    r_sum_sq = r_sum * r_sum
    
    # Only consider i < j to avoid double counting
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Violation is positive if circles overlap
    viol_overlap = np.maximum(0.0, r_sum_sq - d_sq)
    
    # Sum penalties for all overlapping pairs
    penalty += np.sum(viol_overlap[mask])
    
    # Objective: maximize sum of radii => minimize -sum(radii) + penalty
    return -np.sum(radii) + penalty_weight * penalty

def _is_valid(pos, radii, tol=1e-8):
    """Check if the configuration is strictly valid."""
    # Boundary check
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    # Overlap check
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    d_sq = dx * dx + dy * dy
    
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Check if any pair is overlapping (d < r_sum)
    # d_sq < r_sum^2
    if np.any(d_sq[mask] < (r_sum[mask] - tol)**2):
        return False
        
    return True

def _post_process(pos, radii):
    """
    Enforce validity by slightly shrinking radii or adjusting positions.
    This ensures the final output is always valid, even if the optimizer
    left small numerical violations.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Iteratively fix violations
    for _ in range(100):
        valid = True
        
        # 1. Fix boundaries
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            # Max radius allowed by walls
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r_wall:
                radii[i] = max_r_wall * 0.9999
                valid = False
        
        # 2. Fix overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale down radii to make them touch
                    # We want dist >= r_i' + r_j'
                    # Let r_i' = r_i * k, r_j' = r_j * k
                    # k = dist / (r_i + r_j)
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] *= k
                        radii[j] *= k
                        valid = False
                        
        if valid:
            break
            
    # Final safety clamp
    for i in range(N):
        radii[i] = max(radii[i], 1e-6)
        pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
        pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])
        
    return pos, radii

def _initial_lattice():
    """Generate a hexagonal lattice initial state."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Hexagonal packing
    # Try to fit as many circles as possible in a hex grid
    # Row 0: 5 circles
    # Row 1: 5 circles
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 5 circles
    # Row 5: 1 circle (center)
    # Total 26
    
    # Estimate radius
    # Width: 4 * r * sqrt(3) + 2r <= 1 => r(4*1.732 + 2) <= 1 => r <= 1/8.928 ~ 0.112
    # Height: 4 * r * 1.5 + 2r <= 1 => r(6 + 2) <= 1 => r <= 0.125
    r = 0.11
    
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        if row == 5:
            # Center the last circle
            if k < N:
                pos[k] = [0.5, 0.5]
                radii[k] = r
                k += 1
            continue
        
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            
            # Check if it fits
            if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
                
        if k >= N:
            break
            
    # Fill any remaining spots with small circles if needed (should not happen with 26)
    while k < N:
        pos[k] = [0.5, 0.5]
        radii[k] = 0.01
        k += 1
        
    return pos, radii

def _initial_random(seed):
    """Generate a random initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.05, 0.15, N)
    return pos, radii

def _initial_greedy(seed):
    """Greedy placement: place circles one by one in the best available spot."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        # Sample random points
        for _ in range(2000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            
            # Max radius allowed by walls
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            
            # Max radius allowed by existing circles
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - radii[j])
                
            if max_r > best_r:
                best_r = max_r
                best_x = x
                best_y = y
                
        if best_r > 0.01:
            pos[i] = [best_x, best_y]
            radii[i] = best_r * 0.95 # Slight shrink to ensure gap
        else:
            # Fallback
            pos[i] = [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)]
            radii[i] = 0.01
            
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    # Define bounds for optimization
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))
        
    # Initializations
    inits = []
    pos_lat, rad_lat = _initial_lattice()
    inits.append((pos_lat, rad_lat))
    
    pos_gr, rad_gr = _initial_greedy(seed=42)
    inits.append((pos_gr, rad_gr))
    
    pos_rand, rad_rand = _initial_random(seed=123)
    inits.append((pos_rand, rad_rand))
    
    # Optimization Strategy:
    # 1. Differential Evolution (Global)
    # 2. L-BFGS-B (Local Polish)
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        # Stage 1: Differential Evolution
        # Use a subset of time for DE
        if time.time() < deadline - 40:
            try:
                # DE is slow, limit iterations
                result_de = differential_evolution(
                    _penalty_function, 
                    bounds, 
                    args=(1000.0,),
                    maxiter=50, 
                    popsize=15,
                    tol=1e-8,
                    seed=idx,
                    polish=False # We will polish manually
                )
                x_de = result_de.x
            except Exception as e:
                x_de = x0
        else:
            x_de = x0
            
        # Stage 2: L-BFGS-B Polish
        # Start from DE result or initial if DE failed
        x_start = x_de.copy()
        
        # Increase penalty weight for polishing
        for p_w in [1000.0, 5000.0, 10000.0]:
            if time.time() > deadline - 10:
                break
            try:
                result_lbfgs = minimize(
                    _penalty_function, 
                    x_start, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    args=(p_w,),
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result_lbfgs.fun < _penalty_function(x_start, p_w) - 1e-10:
                    x_start = result_lbfgs.x
            except Exception:
                pass
        
        # Extract results
        pos_opt = x_start[:2*N].reshape(N, 2)
        radii_opt = x_start[2*N:]
        
        # Post-process to ensure validity
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                
    # Fallback if no valid result found
    if best_pos is None:
        pos_f, rad_f = _initial_lattice()
        pos_f, rad_f = _post_process(pos_f, rad_f)
        best_pos, best_radii = pos_f, rad_f
        
    # Final validation
    if not _is_valid(best_pos, best_radii):
        # Force valid fallback
        pos_f, rad_f = _initial_lattice()
        pos_f, rad_f = _post_process(pos_f, rad_f)
        best_pos, best_radii = pos_f, rad_f
        
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
