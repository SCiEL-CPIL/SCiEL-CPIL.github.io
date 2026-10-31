import numpy as np
from scipy.optimize import minimize
import itertools

N = 26

def construct():
    """
    Packs 26 circles in a unit square maximizing the sum of radii.
    Uses randomized penalty-based gradient descent (L-BFGS-B).
    """
    best_sum = -1
    best_result = None

    # Run multiple trials to escape local minima
    for trial in range(10):
        # Random initialization
        # Positions in [0.1, 0.9] to start away from boundaries
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.05) # Start with small radii
        
        # Flatten variables: [x0, y0, r0, x1, y1, r1, ...]
        def unpack(v):
            x = v[0::3]
            y = v[1::3]
            r = v[2::3]
            return x, y, r

        def cost(v):
            x, y, r = unpack(v)
            
            # Objective: Maximize sum of radii -> Minimize negative sum
            obj = -np.sum(r)
            
            # Penalty for overlaps
            penalty = 0.0
            # Vectorized overlap check is tricky with simple loops in Python, 
            # but N=26 is small enough for O(N^2) loop or vectorized numpy ops.
            # Let's use vectorized numpy for speed.
            
            # Compute all pairwise distances
            # dx[i,j] = x[i] - x[j]
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            dist = np.sqrt(dx**2 + dy**2)
            
            # Sum of radii for each pair
            r_sum = r[:, np.newaxis] + r[np.newaxis, :]
            
            # Overlap violation: dist < r_sum
            # We want penalty for (r_sum - dist) when positive
            violations = np.maximum(0, r_sum - dist)
            # Zero out diagonal (self-overlap is 0)
            np.fill_diagonal(violations, 0)
            
            # Square the violations for smooth gradient
            penalty_overlap = np.sum(violations**2)
            
            # Penalty for boundaries
            # x < r  -> r - x > 0
            # x > 1-r -> x - (1-r) > 0
            # Same for y
            
            pen_x_low = np.maximum(0, r - x)
            pen_x_high = np.maximum(0, x - (1 - r))
            pen_y_low = np.maximum(0, r - y)
            pen_y_high = np.maximum(0, y - (1 - r))
            
            penalty_boundary = np.sum(pen_x_low**2) + np.sum(pen_x_high**2) + \
                              np.sum(pen_y_low**2) + np.sum(pen_y_high**2)
            
            total_penalty = penalty_overlap + penalty_boundary
            
            return obj + 100.0 * total_penalty

        # Initial vector
        v0 = np.zeros(N * 3)
        v0[0::3] = x
        v0[1::3] = y
        v0[2::3] = r
        
        # Bounds:
        # x, y in [0, 1]
        # r in [0, 0.5] (max possible radius in unit square)
        bounds = []
        for _ in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        try:
            res = minimize(cost, v0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
            
            # Extract solution
            x_sol, y_sol, r_sol = unpack(res.x)
            
            # Verify validity strictly
            valid = True
            # Check boundaries
            if np.any(x_sol < r_sol - 1e-6) or np.any(x_sol > 1 - r_sol + 1e-6):
                valid = False
            if np.any(y_sol < r_sol - 1e-6) or np.any(y_sol > 1 - r_sol + 1e-6):
                valid = False
            
            if valid:
                # Check overlaps
                dx = x_sol[:, np.newaxis] - x_sol[np.newaxis, :]
                dy = y_sol[:, np.newaxis] - y_sol[np.newaxis, :]
                dist = np.sqrt(dx**2 + dy**2)
                r_sum = r_sol[:, np.newaxis] + r_sol[np.newaxis, :]
                np.fill_diagonal(r_sum, 0)
                # If any dist < r_sum - tolerance, invalid
                if np.any(dist < r_sum - 1e-6):
                    valid = False
            
            if valid:
                current_sum = np.sum(r_sol)
                if current_sum > best_sum:
                    best_sum = current_sum
                    # Ensure exact format
                    best_result = np.column_stack((x_sol, y_sol, r_sol))
                    
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to seed-like grid if optimization fails completely
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
