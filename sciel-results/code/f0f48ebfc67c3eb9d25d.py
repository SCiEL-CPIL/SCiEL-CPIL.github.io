# EVOLVE-BLOCK-START
import numpy as np
import time

def squares_in_square10() -> tuple:
    """
    Maximize the total side length of 10 axis-aligned squares in the unit square.
    Uses Simulated Annealing with a penalty function for constraints.
    """
    n = 10
    # Initialize with 3x3 grid packing (baseline)
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    
    # Precompute indices for overlap checks
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Track best valid solution
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = np.sum(sides)
    
    # Current state
    current_centers = centers.copy()
    current_sides = sides.copy()
    current_score, _ = evaluate(current_centers, current_sides, i_idx, j_idx)
    
    # SA Parameters
    T = 0.1
    alpha = 0.9995
    step_size = 0.01
    T_min = 1e-4
    
    # Time management
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Generate a neighbor
        idx = rng.integers(n)
        if rng.random() < 0.5:
            # Move a square
            dim = rng.integers(2)
            delta = rng.normal(0, step_size)
            new_centers = current_centers.copy()
            new_centers[idx, dim] += delta
            new_sides = current_sides
        else:
            # Resize a square
            delta = rng.normal(0, step_size * 0.5)
            new_sides = current_sides.copy()
            new_sides[idx] += delta
            if new_sides[idx] < 0.01:
                new_sides[idx] = 0.01
            new_centers = current_centers
            
        # Evaluate new state
        new_score, is_valid = evaluate(new_centers, new_sides, i_idx, j_idx)
        
        # Acceptance criterion
        if is_valid:
            if new_score > current_score:
                # Always accept improvement
                current_centers = new_centers
                current_sides = new_sides
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
                    best_sides = new_sides.copy()
            else:
                # Metropolis criterion for valid but worse states
                delta_e = new_score - current_score
                if delta_e > 0 or rng.random() < np.exp(delta_e / T):
                    current_centers = new_centers
                    current_sides = new_sides
                    current_score = new_score
        else:
            # Invalid state: Accept with penalty-based probability?
            # For simplicity and to keep the search in the valid basin mostly,
            # we reject invalid states unless the penalty is very small (near valid).
            # However, to escape local minima, we should allow some invalid moves.
            # Let's use the score (which includes penalty) for the decision even if invalid.
            delta_e = new_score - current_score
            if delta_e > 0 or rng.random() < np.exp(delta_e / T):
                current_centers = new_centers
                current_sides = new_sides
                current_score = new_score
                # Note: We do NOT update best_score here as it's invalid.
                
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: Restart if stuck?
            # If T is min and no improvement for a while, we might be stuck.
            # Given the time budget, we just keep going with small steps.
            
    return best_centers, best_sides

def evaluate(centers, sides, i_idx, j_idx):
    """
    Evaluates the state. Returns (score, is_valid).
    Score = sum(sides) - penalty.
    Penalty is high if invalid.
    """
    # 1. Check bounds
    min_x = centers[:, 0] - sides / 2.0
    max_x = centers[:, 0] + sides / 2.0
    min_y = centers[:, 1] - sides / 2.0
    max_y = centers[:, 1] + sides / 2.0
    
    bound_pen = 0.0
    # Vectorized penalty calculation
    bound_pen += np.sum(np.clip(-min_x, 0, None))
    bound_pen += np.sum(np.clip(max_x - 1, 0, None))
    bound_pen += np.sum(np.clip(-min_y, 0, None))
    bound_pen += np.sum(np.clip(max_y - 1, 0, None))
    
    # 2. Check overlaps
    cx = centers[:, 0]
    cy = centers[:, 1]
    
    # Distance in x and y
    dx = (sides[i_idx] + sides[j_idx]) / 2.0 - np.abs(cx[i_idx] - cx[j_idx])
    dy = (sides[i_idx] + sides[j_idx]) / 2.0 - np.abs(cy[i_idx] - cy[j_idx])
    
    # Overlap if dx > 0 and dy > 0
    # Penalty is the area of overlap: max(0, dx) * max(0, dy)
    # This is smooth and differentiable-ish
    overlap_area = np.sum(np.clip(dx, 0, None) * np.clip(dy, 0, None))
    
    # Total penalty
    # Weights: Strong penalty for violations
    penalty = 10.0 * bound_pen + 10.0 * overlap_area
    
    score = np.sum(sides) - penalty
    
    # Check strict validity
    # Valid if no bound violation and no overlap (with tolerance 1e-7)
    strict_valid = True
    if bound_pen > 1e-7:
        strict_valid = False
    else:
        # Check if any pair has dx > 1e-7 AND dy > 1e-7
        # If so, they overlap forbiddenly
        if np.any((dx > 1e-7) & (dy > 1e-7)):
            strict_valid = False
            
    return score, strict_valid

# EVOLVE-BLOCK-END
