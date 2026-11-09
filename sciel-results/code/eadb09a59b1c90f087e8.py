"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys

def _score(positions):
    """Compute the minimum radius given positions in [-0.5, 0.5]^2."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Distance to boundary
    min_boundary = np.min(np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    ))
    
    if min_boundary <= 0:
        return 0.0

    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    return min(min_boundary, min_pair / 2.0)

def _energy_objective(pos_flat, p=10, eps=1e-6):
    """
    Smooth inverse-power-sum energy that penalizes close pairs heavily.
    Minimizing this drives the system to maximize the minimum distance.
    """
    pos = pos_flat.reshape(-1, 2)
    n = len(pos)
    
    # Compute all pairwise distances squared
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    
    # Set diagonal to large value to ignore self-distances
    np.fill_diagonal(dist_sq, 1e10)
    
    # Inverse power sum: sum of 1/(d^p) for all pairs
    # This is smooth and has well-defined gradients
    energy = np.sum(1.0 / (dist_sq + eps) ** (p/2.0))
    
    # Add boundary penalty: penalize points close to or outside boundaries
    # Use smooth penalty: (0.5 - |x|)^p for x, similar for y
    boundary_penalty = 0.0
    x = pos[:, 0]
    y = pos[:, 1]
    # Penalize if x > 0.5 - margin or x < -0.5 + margin
    # Use smooth max-like function
    x_margin = 0.5 - np.abs(x)
    y_margin = 0.5 - np.abs(y)
    
    # Smooth penalty for being close to boundary
    boundary_penalty += np.sum(np.maximum(0, 0.1 - x_margin) ** p)
    boundary_penalty += np.sum(np.maximum(0, 0.1 - y_margin) ** p)
    
    return energy + 1000.0 * boundary_penalty

def _generate_hexagonal_grid(n_circles, square_size=1.0):
    """Generate a hexagonal grid of n_circles within a square."""
    # Hexagonal packing: rows with alternating offsets
    # For a square of side 1, we want to fit as many circles as possible
    # then select n_circles
    
    # Start with a dense hexagonal grid and select the best n_circles
    # Hexagonal spacing: horizontal = d, vertical = d * sqrt(3)/2
    # For 200 circles in unit square, approximate spacing
    
    # Try different spacings and offsets to find good configuration
    best_pos = None
    best_score = 0.0
    
    # Approximate optimal spacing for 200 circles
    # Area per circle ~ 1/200, so diameter ~ 1/sqrt(200) ~ 0.07
    # Hexagonal packing density ~ pi/(2*sqrt(3)) ~ 0.907
    
    for trial in range(5):
        # Vary the base spacing slightly
        base_spacing = 1.0 / np.sqrt(n_circles / 0.9) * (1.0 + 0.05 * trial)
        
        # Generate hexagonal grid
        row_spacing = base_spacing * np.sqrt(3) / 2
        cols = int(np.ceil(1.0 / base_spacing)) + 1
        rows = int(np.ceil(1.0 / row_spacing)) + 1
        
        points = []
        for row in range(rows):
            for col in range(cols):
                x = col * base_spacing
                if row % 2 == 1:
                    x += base_spacing / 2
                y = row * row_spacing
                # Center in unit square [-0.5, 0.5]
                x -= 0.5
                y -= 0.5
                if -0.45 <= x <= 0.45 and -0.45 <= y <= 0.45:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < n_circles:
            continue
        
        # Select n_circles from the grid - take the most central ones
        # Sort by distance from center and take closest n_circles
        center = np.array([0.0, 0.0])
        dists_to_center = np.sum((points - center) ** 2, axis=1)
        sorted_idx = np.argsort(dists_to_center)
        selected = points[sorted_idx[:n_circles]]
        
        score = _score(selected)
        if score > best_score:
            best_score = score
            best_pos = selected.copy()
    
    if best_pos is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:n_circles].copy()
    
    return best_pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin from 150s limit

    best_score = 0.0
    best_positions = None

    # Define bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Generate hexagonal grid initialization
    hex_pos = _generate_hexagonal_grid(N)
    curr_score = _score(hex_pos)
    if curr_score > best_score:
        best_score = curr_score
        best_positions = hex_pos.copy()
    
    # Also try square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    square_pos = grid[:N].copy()
    curr_score_sq = _score(square_pos)
    if curr_score_sq > best_score:
        best_score = curr_score_sq
        best_positions = square_pos.copy()

    # Optimization using smooth energy objective
    # Try multiple passes with different power parameters
    for p_val in [8, 10, 12]:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                _energy_objective, 
                best_positions.ravel(), 
                args=(p_val,),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            new_pos = res.x.reshape(-1, 2)
            new_score = _score(new_pos)
            if new_score > best_score:
                best_score = new_score
                best_positions = new_pos.copy()
                
        except Exception:
            pass
    
    # Final polish: use the actual score function with tighter tolerances
    if time.time() < deadline - 5 and best_positions is not None:
        def _polish_objective(pos_flat):
            pos = pos_flat.reshape(-1, 2)
            out_of_bounds = np.any(pos > 0.5) or np.any(pos < -0.5)
            if out_of_bounds:
                return 1e6
            return -_score(pos)
        
        try:
            res_polish = minimize(
                _polish_objective, 
                best_positions.ravel(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 5000, 'maxfun': 50000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            new_pos_polish = res_polish.x.reshape(-1, 2)
            new_score_polish = _score(new_pos_polish)
            if new_score_polish > best_score:
                best_score = new_score_polish
                best_positions = new_pos_polish.copy()
        except Exception:
            pass

    return best_positions
# EVOLVE-BLOCK-END
