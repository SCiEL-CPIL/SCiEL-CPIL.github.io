import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 600.0  # Safety margin
    
    def objective(params):
        # params is shape (10, 2)
        c = params.reshape(n, 2)
        # Compute sides for each square
        sides = np.zeros(n)
        for i in range(n):
            ci = c[i]
            # Boundary constraints
            s_bound = np.min([
                2.0 * ci[0],
                2.0 * (1.0 - ci[0]),
                2.0 * ci[1],
                2.0 * (1.0 - ci[1])
            ])
            
            # Neighbor constraints
            s_min_neighbor = np.inf
            for j in range(n):
                if i == j:
                    continue
                cj = c[j]
                dx = ci[0] - cj[0]
                dy = ci[1] - cj[1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 1e-12:
                    # Centers coincide, side is 0
                    s_min_neighbor = 0.0
                    break
                abs_dx = abs(dx)
                abs_dy = abs(dy)
                denom = abs_dx + abs_dy
                if denom < 1e-12:
                    s_min_neighbor = 0.0
                    break
                s_val = dist_sq / denom
                if s_val < s_min_neighbor:
                    s_min_neighbor = s_val
            
            sides[i] = min(s_bound, s_min_neighbor)
            
        return -np.sum(sides)

    def get_sides(centers):
        c = centers.reshape(n, 2)
        sides = np.zeros(n)
        for i in range(n):
            ci = c[i]
            s_bound = np.min([
                2.0 * ci[0],
                2.0 * (1.0 - ci[0]),
                2.0 * ci[1],
                2.0 * (1.0 - ci[1])
            ])
            s_min_neighbor = np.inf
            for j in range(n):
                if i == j:
                    continue
                cj = c[j]
                dx = ci[0] - cj[0]
                dy = ci[1] - cj[1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 1e-12:
                    s_min_neighbor = 0.0
                    break
                abs_dx = abs(dx)
                abs_dy = abs(dy)
                denom = abs_dx + abs_dy
                if denom < 1e-12:
                    s_min_neighbor = 0.0
                    break
                s_val = dist_sq / denom
                if s_val < s_min_neighbor:
                    s_min_neighbor = s_val
            sides[i] = min(s_bound, s_min_neighbor)
        return sides

    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    # Generate diverse initial points
    np.random.seed(42)
    num_starts = 50
    
    for k in range(num_starts):
        if time.time() - start_time > time_limit - 30:
            break
            
        # Random initialization
        centers_init = np.random.rand(n, 2)
        
        # Optimize
        res = minimize(
            objective,
            centers_init.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1)] * (2 * n),
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_score:
            centers_opt = res.x.reshape(n, 2)
            score = -res.fun
            if score > best_score:
                best_score = score
                best_centers = centers_opt
                best_sides = get_sides(centers_opt)

    # If no optimization succeeded or time ran out, use a fallback
    if best_centers is None:
        # Fallback to grid
        k_grid = 4
        side = 1.0 / k_grid
        idx = np.arange(n)
        cols = idx % k_grid
        rows = idx // k_grid
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
        # Note: Grid might place 10th square outside if k=4 and n=10?
        # idx 9 -> row 2, col 1. Inside.
        # idx 15 would be row 3.
        # So grid is valid.

    return best_centers, best_sides
