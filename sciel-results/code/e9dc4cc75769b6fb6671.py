# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _is_valid(centers, sides, tol=1e-7):
    """Check if the packing is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _penalty(centers, sides, tol=1e-7):
    """Compute total constraint violation (penalty)."""
    n = len(sides)
    pen = 0.0
    # Boundary violations
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        lo_x = cx - s/2
        hi_x = cx + s/2
        lo_y = cy - s/2
        hi_y = cy + s/2
        if lo_x < 0:
            pen += (0 - lo_x) ** 2
        if hi_x > 1:
            pen += (hi_x - 1) ** 2
        if lo_y < 0:
            pen += (0 - lo_y) ** 2
        if hi_y > 1:
            pen += (hi_y - 1) ** 2
    # Overlap violations
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                # Both axes overlap; penalty is the overlap amount
                pen += (t - dx) ** 2 + (t - dy) ** 2
    return pen


def _score(centers, sides, tol=1e-7):
    """Score = sum of sides - penalty. Higher is better."""
    return np.sum(sides) - 100.0 * _penalty(centers, sides, tol)


def _clamp_to_bounds(centers, sides):
    """Ensure all squares are within [0,1]^2 by adjusting centers/sides."""
    n = len(sides)
    for i in range(n):
        s = sides[i]
        cx = centers[i, 0]
        cy = centers[i, 1]
        # Clamp side to positive
        if s < 1e-6:
            s = 1e-6
        # Clamp center so square stays in [0,1]
        cx = max(s/2, min(1 - s/2, cx))
        cy = max(s/2, min(1 - s/2, cy))
        centers[i, 0] = cx
        centers[i, 1] = cy
        sides[i] = s
    return centers, sides


def _generate_initial_config(n=10, seed=42):
    """Generate a diverse initial configuration."""
    rng = random.Random(seed)
    
    # Strategy: try several structural templates and pick random ones
    strategy = rng.randint(0, 4)
    
    if strategy == 0:
        # Grid 3x3 with one removed
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        side = 1.0 / 3
        idx = 0
        for r in range(3):
            for c in range(3):
                if idx >= n:
                    break
                centers[idx, 0] = (c + 0.5) * side
                centers[idx, 1] = (r + 0.5) * side
                sides[idx] = side
                idx += 1
        # Add small noise
        for i in range(n):
            centers[i, 0] += rng.uniform(-0.02, 0.02)
            centers[i, 1] += rng.uniform(-0.02, 0.02)
    
    elif strategy == 1:
        # 2 large + 8 small
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        # Two large squares
        centers[0, 0] = 0.25
        centers[0, 1] = 0.25
        sides[0] = 0.5
        centers[1, 0] = 0.75
        centers[1, 1] = 0.25
        sides[1] = 0.5
        # 8 small squares in remaining space
        small_side = 0.25
        positions = [
            (0.25, 0.75), (0.75, 0.75),
            (0.125, 0.5), (0.375, 0.5),
            (0.625, 0.5), (0.875, 0.5),
            (0.125, 0.75), (0.375, 0.75)
        ]
        for k in range(8):
            centers[2+k, 0] = positions[k][0]
            centers[2+k, 1] = positions[k][1]
            sides[2+k] = small_side
        for i in range(n):
            centers[i, 0] += rng.uniform(-0.01, 0.01)
            centers[i, 1] += rng.uniform(-0.01, 0.01)
    
    elif strategy == 2:
        # Random positions with varied sizes
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        for i in range(n):
            s = rng.uniform(0.1, 0.4)
            sides[i] = s
            centers[i, 0] = rng.uniform(s/2, 1 - s/2)
            centers[i, 1] = rng.uniform(s/2, 1 - s/2)
    
    elif strategy == 3:
        # One large + 9 medium
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        centers[0, 0] = 0.5
        centers[0, 1] = 0.5
        sides[0] = 0.5
        med_side = 0.2
        positions = [
            (0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85),
            (0.5, 0.15), (0.5, 0.85), (0.15, 0.5), (0.85, 0.5), (0.5, 0.5)
        ]
        for k in range(9):
            centers[1+k, 0] = positions[k][0]
            centers[1+k, 1] = positions[k][1]
            sides[1+k] = med_side
        for i in range(n):
            centers[i, 0] += rng.uniform(-0.01, 0.01)
            centers[i, 1] += rng.uniform(-0.01, 0.01)
    
    else:
        # Pure random
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        for i in range(n):
            s = rng.uniform(0.05, 0.35)
            sides[i] = s
            centers[i, 0] = rng.uniform(s/2, 1 - s/2)
            centers[i, 1] = rng.uniform(s/2, 1 - s/2)
    
    centers, sides = _clamp_to_bounds(centers, sides)
    return centers, sides


def _local_search(centers, sides, max_iters=50000, tol=1e-7, seed=42, time_limit=30):
    """Simulated annealing local search."""
    rng = random.Random(seed)
    start_time = time.time()
    n = len(sides)
    
    cur_centers = centers.copy()
    cur_sides = sides.copy()
    cur_score = _score(cur_centers, cur_sides, tol)
    
    best_centers = cur_centers.copy()
    best_sides = cur_sides.copy()
    best_score = cur_score
    
    T = 0.05  # Initial temperature
    T_min = 1e-8
    cooling = 0.9995
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if T < T_min:
            break
        
        # Generate neighbor
        new_centers = cur_centers.copy()
        new_sides = cur_sides.copy()
        
        # Random perturbation
        num_moves = rng.randint(1, 3)
        for _ in range(num_moves):
            move_type = rng.randint(0, 3)
            idx = rng.randint(0, n - 1)
            
            if move_type == 0:
                # Move center x
                step = rng.uniform(-0.05, 0.05)
                new_centers[idx, 0] += step
            elif move_type == 1:
                # Move center y
                step = rng.uniform(-0.05, 0.05)
                new_centers[idx, 1] += step
            elif move_type == 2:
                # Change side length
                step = rng.uniform(-0.03, 0.03)
                new_sides[idx] += step
            else:
                # Swap two squares' positions (structural change)
                idx2 = rng.randint(0, n - 1)
                if idx2 != idx:
                    new_centers[idx], new_centers[idx2] = new_centers[idx2].copy(), new_centers[idx].copy()
                    new_sides[idx], new_sides[idx2] = new_sides[idx2], new_sides[idx]
        
        # Clamp
        new_centers, new_sides = _clamp_to_bounds(new_centers, new_sides)
        
        new_score = _score(new_centers, new_sides, tol)
        delta = new_score - cur_score
        
        # Acceptance criterion
        if delta > 0 or rng.random() < np.exp(delta / T):
            cur_centers = new_centers
            cur_sides = new_sides
            cur_score = new_score
            
            if cur_score > best_score:
                best_score = cur_score
                best_centers = cur_centers.copy()
                best_sides = cur_sides.copy()
        
        T *= cooling
    
    return best_centers, best_sides, best_score


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 630  # seconds, with safety margin
    
    best_centers = None
    best_sides = None
    best_value = -np.inf
    
    # Multiple restarts with different seeds and strategies
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() - start_time > deadline - 30:
            break
        
        seed = 42 + restart * 7
        centers, sides = _generate_initial_config(n, seed)
        
        # Run local search
        time_per_search = min(15, max(3, (deadline - 30 - (time.time() - start_time)) / max(1, num_restarts - restart)))
        c_opt, s_opt, sc = _local_search(centers, sides, max_iters=30000, tol=tol, seed=seed, time_limit=time_per_search)
        
        # Check validity
        if _is_valid(c_opt, s_opt, tol):
            val = np.sum(s_opt)
            if val > best_value:
                best_value = val
                best_centers = c_opt.copy()
                best_sides = s_opt.copy()
    
    # Fallback: grid if no valid solution found
    if best_centers is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
