"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import itertools
import math


def construct():
    """
    Pack 26 circles in the unit square maximizing sum of radii.
    
    Strategy: Use a structured approach with differential evolution on a reduced
    set of parameters, then local refinement.
    
    We use a grid-like initial structure but allow positions and radii to be
    optimized. To handle the non-convexity, we use differential evolution with
    constraints handled via penalty functions.
    """
    
    def objective(params):
        """
        params: array of length 2*N (x, y for each circle) + N radii = 3*N
        Actually, let's use 3*N parameters: x_i, y_i, r_i for i in range(N)
        """
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Penalty for going outside the square
        penalty = 0.0
        for i in range(N):
            if x[i] < r[i]:
                penalty += (r[i] - x[i]) ** 2
            if x[i] > 1 - r[i]:
                penalty += (x[i] - (1 - r[i])) ** 2
            if y[i] < r[i]:
                penalty += (r[i] - y[i]) ** 2
            if y[i] > 1 - r[i]:
                penalty += (y[i] - (1 - r[i])) ** 2
        
        # Penalty for overlapping circles
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        # We want to maximize sum of radii, so minimize negative sum + penalty
        return -np.sum(r) + 100.0 * penalty
    
    def constraints_check(x, y, r):
        """Check if the configuration is valid."""
        for i in range(N):
            if x[i] < r[i] - 1e-9 or x[i] > 1 - r[i] + 1e-9:
                return False
            if y[i] < r[i] - 1e-9 or y[i] > 1 - r[i] + 1e-9:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True
    
    # Initial guess: grid layout
    side = int(math.ceil(math.sqrt(N)))
    init_x = np.zeros(N)
    init_y = np.zeros(N)
    init_r = np.zeros(N)
    
    for i in range(N):
        row = i // side
        col = i % side
        init_x[i] = (col + 0.5) / side
        init_y[i] = (row + 0.5) / side
        init_r[i] = 0.5 / side - 0.01
    
    init_params = np.concatenate([init_x, init_y, init_r])
    
    # Bounds for differential evolution
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
    for i in range(N):
        bounds.append((0.0, 1.0))  # y
    for i in range(N):
        bounds.append((0.0, 0.5))  # r
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=200,
        tol=1e-8,
        polish=True,
        popsize=15,
        init='sobol',
        mutation=(0.5, 1.5),
        recombination=0.7
    )
    
    best_params = result.x
    x = best_params[0:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    # If the DE result is invalid, fall back to a simpler approach
    if not constraints_check(x, y, r):
        # Try a simpler local optimization from the grid start
        def obj_simple(params):
            x = params[0:N]
            y = params[N:2*N]
            r = params[2*N:3*N]
            
            penalty = 0.0
            for i in range(N):
                if x[i] < r[i]:
                    penalty += (r[i] - x[i]) ** 2
                if x[i] > 1 - r[i]:
                    penalty += (x[i] - (1 - r[i])) ** 2
                if y[i] < r[i]:
                    penalty += (r[i] - y[i]) ** 2
                if y[i] > 1 - r[i]:
                    penalty += (y[i] - (1 - r[i])) ** 2
            
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            
            return -np.sum(r) + 100.0 * penalty
        
        # Use L-BFGS-B with bounds
        res = minimize(
            obj_simple,
            init_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        x = res.x[0:N]
        y = res.x[N:2*N]
        r = res.x[2*N:3*N]
    
    # Final validation and cleanup
    # Ensure all circles are within bounds and non-overlapping
    for i in range(N):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    
    # Check for overlaps and adjust if needed
    for _ in range(100):
        adjusted = False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    # Push apart
                    if dist > 1e-10:
                        push = (min_dist - dist) / 2
                        nx = dx / dist
                        ny = dy / dist
                        x[i] += push * nx
                        y[i] += push * ny
                        x[j] -= push * nx
                        y[j] -= push * ny
                    else:
                        x[i] += 0.01
                        x[j] -= 0.01
                    adjusted = True
            # Re-clip
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        if not adjusted:
            break
    
    # Final check
    if not constraints_check(x, y, r):
        # Fall back to the simple grid
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            cx = (col + 0.5) / side
            cy = (row + 0.5) / side
            cr = 0.5 / side - 1e-3
            cells.append([cx, cy, cr])
        return np.array(cells)
    
    return np.column_stack([x, y, r])

# EVOLVE-BLOCK-END
