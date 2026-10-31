"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 21
DEADLINE_SECONDS = 110.0  # safety margin before 120s kill


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_value = -np.inf
    best_result = None

    # Helper: evaluate a packing configuration
    def evaluate(circles, width, height):
        """circles: (n, 3) array of [x, y, r]. Returns (valid, sum_r)."""
        if circles.shape[0] != N:
            return False, 0.0
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Check boundaries
        if np.any(xs < -1e-9) or np.any(ys < -1e-9):
            return False, 0.0
        if np.any(xs + rs > width + 1e-9) or np.any(ys + rs > height + 1e-9):
            return False, 0.0
        if np.any(rs < 1e-9):
            return False, 0.0
        
        # Check pairwise distances
        # Compute distance matrix
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dists = np.sqrt(dx**2 + dy**2)
        min_dists = rs[:, None] + rs[None, :]
        
        # For all pairs i != j, dist >= min_dists
        mask = ~np.eye(N, dtype=bool)
        if np.any(dists[mask] < min_dists[mask] - 1e-9):
            return False, 0.0
        
        return True, np.sum(rs)

    # Objective function for optimization
    def make_objective(width, height):
        """Returns objective function that minimizes negative sum of radii."""
        def objective(params):
            # params: [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20]
            if time.time() - start_time > DEADLINE_SECONDS:
                return 1e10  # signal timeout
            circles = params.reshape(N, 3)
            xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
            
            # Boundary penalties
            pen = 0.0
            pen += np.sum(np.maximum(0, -xs)**2)
            pen += np.sum(np.maximum(0, -ys)**2)
            pen += np.sum(np.maximum(0, xs + rs - width)**2)
            pen += np.sum(np.maximum(0, ys + rs - height)**2)
            
            # Pairwise distance penalties
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dists = np.sqrt(dx**2 + dy**2)
            min_dists = rs[:, None] + rs[None, :]
            mask = ~np.eye(N, dtype=bool)
            violations = np.maximum(0, min_dists[mask] - dists[mask])
            pen += np.sum(violations**2)
            
            # Small penalty for very small radii to encourage larger circles
            pen += 1e-4 * np.sum(np.maximum(0, 0.01 - rs)**2)
            
            return -np.sum(rs) + 100.0 * pen
        
        return objective

    def make_bounds(width, height):
        """Bounds for each circle: x in [0, width], y in [0, height], r in [0.001, min(width,height)/2]"""
        max_r = min(width, height) / 2.0
        bounds = []
        for _ in range(N):
            bounds.append((0.0, width))
            bounds.append((0.0, height))
            bounds.append((0.001, max_r))
        return bounds

    def random_initial(width, height):
        """Generate a random feasible-ish initial configuration."""
        # Start with circles placed at grid positions with small radii
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) * width / side
                cy = (j + 0.5) * height / side
                r = 0.5 * min(width, height) / side * 0.9
                cells.append([cx, cy, r])
        
        # Add randomness
        for k in range(N):
            cells[k][0] += np.random.uniform(-0.1, 0.1) * width / side
            cells[k][1] += np.random.uniform(-0.1, 0.1) * height / side
            cells[k][0] = np.clip(cells[k][0], 0.01, width - 0.01)
            cells[k][1] = np.clip(cells[k][1], 0.01, height - 0.01)
            cells[k][2] = np.clip(cells[k][2], 0.01, min(width, height) / 2 - 0.01)
        
        return np.array(cells).flatten()

    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.66, 1.5, 0.5, 2.0, 0.75, 1.33]
    
    for aspect in aspect_ratios:
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        
        # Perimeter = 2*(width + height) = 4, so width + height = 2
        # aspect = width / height
        # width = aspect * height
        # aspect * height + height = 2 => height = 2/(1+aspect), width = 2*aspect/(1+aspect)
        height = 2.0 / (1.0 + aspect)
        width = 2.0 * aspect / (1.0 + aspect)
        
        # Run multiple random restarts
        for restart in range(3):
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
            
            x0 = random_initial(width, height)
            obj = make_objective(width, height)
            bnds = make_bounds(width, height)
            
            try:
                result = minimize(
                    obj, x0, method='L-BFGS-B',
                    bounds=bnds,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
            except Exception:
                continue
            
            if time.time() - start_time > DEADLINE_SECONDS - 3:
                break
            
            circles = result.x.reshape(N, 3)
            valid, val = evaluate(circles, width, height)
            
            if valid and val > best_value:
                best_value = val
                best_result = circles.copy()
    
    # If no valid result found, fall back to simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
