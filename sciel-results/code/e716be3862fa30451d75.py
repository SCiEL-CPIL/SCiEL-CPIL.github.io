import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    best_score = -np.inf
    best_result = None

    # Helper to evaluate validity and score
    def _evaluate(centers, radii):
        """
        Check if the configuration is valid and return the sum of radii.
        Returns -inf if invalid.
        """
        # Check boundary constraints
        for i in range(len(centers)):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9:
                return -np.inf
            if y - r < -1e-9 or y + r > 1 + 1e-9:
                return -np.inf
        
        # Check pairwise distances
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - 1e-9:
                    return -np.inf
        
        return np.sum(radii)

    def _objective(vars_):
        """
        Objective function for optimization.
        Minimizes -sum(radii) + penalties.
        """
        c = vars_[:2*N].reshape(N, 2)
        r = vars_[2*N:]
        
        # Ensure radii are positive
        r = np.maximum(r, 1e-6)
        
        # Boundary penalties (squared for smoothness)
        pen_bound = 0.0
        for i in range(N):
            x, y = c[i]
            ri = r[i]
            # Distance to left, right, bottom, top
            d_left = x - ri
            d_right = 1 - x - ri
            d_bottom = y - ri
            d_top = 1 - y - ri
            
            if d_left < 0: pen_bound += d_left**2
            if d_right < 0: pen_bound += d_right**2
            if d_bottom < 0: pen_bound += d_bottom**2
            if d_top < 0: pen_bound += d_top**2

        # Overlap penalties (squared for smoothness)
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(c[i] - c[j])
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    pen_overlap += overlap**2

        # Weight for penalties needs to be high enough to enforce constraints
        # but not so high that it dominates the objective gradient
        weight = 1000.0
        return -np.sum(r) + weight * (pen_bound + pen_overlap)

    # Generate a hexagonal-like initialization
    def _make_hex_init():
        """
        Create a hexagonal lattice initialization specifically tuned for N=32.
        Layout: 6, 5, 6, 5, 6, 4 circles per row.
        """
        centers = []
        radii = []
        
        # Approximate spacing for hex packing in unit square
        # Height of hexagon is sqrt(3)/2 * side_length
        # We have 6 rows. Total height ~ 5 * sqrt(3)/2 * s = 1 -> s ~ 0.23
        # Width of row with 6 circles: 6*s = 1.38 (too wide, will be compressed)
        # Let's just place them in a grid first and let optimizer fix it.
        # Better: Place them in a hex pattern scaled to fit [0,1]^2.
        
        rows = [6, 5, 6, 5, 6, 4]
        total_rows = len(rows)
        
        # Vertical spacing
        y_pos = []
        for i in range(total_rows):
            y_pos.append((i + 0.5) / total_rows)
        
        for row_idx, count in enumerate(rows):
            y = y_pos[row_idx]
            # Horizontal spacing
            # Shift odd rows (1, 3, 5) by half a unit to create hex pattern
            offset = 0.0
            if row_idx % 2 == 1:
                offset = 0.5 / count
            
            for col_idx in range(count):
                x = (col_idx + 0.5 + offset) / count
                # Ensure x is within [0,1]
                x = np.clip(x, 0.0, 1.0)
                
                # Estimate initial radius
                # Distance to nearest neighbor in hex is roughly s
                # Let's use a conservative radius
                r = 0.45 / count if count > 1 else 0.2
                
                centers.append([x, y])
                radii.append(r)
        
        return np.array(centers[:N]), np.array(radii[:N])

    # Generate a grid initialization
    def _make_grid_init():
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        radii = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([0.5/side + i/side, 0.5/side + j/side])
                    radii.append(0.45/side)
        return np.array(centers[:N]), np.array(radii[:N])

    # Generate random initialization
    def _make_random_init():
        centers = np.random.rand(N, 2) * 0.8 + 0.1
        radii = np.random.uniform(0.01, 0.08, N)
        return centers, radii

    # Optimization bounds
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
    for i in range(N):
        bounds.append((0, 0.5))  # r

    # List of initialization strategies
    inits = [
        _make_hex_init,
        _make_grid_init,
        _make_random_init,
        _make_random_init,
        _make_random_init
    ]

    np.random.seed(42)

    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Select initialization strategy
        if restart < len(inits):
            c_init, r_init = inits[restart]()
        else:
            # Perturb the hex init
            c_init, r_init = _make_hex_init()
            c_init += np.random.normal(0, 0.01, c_init.shape)
            c_init = np.clip(c_init, 0.01, 0.99)
            r_init = np.clip(r_init + np.random.normal(0, 0.005, N), 0.01, 0.2)

        x0 = np.concatenate([c_init.flatten(), r_init])
        
        try:
            # Use L-BFGS-B for speed
            res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20})
            
            c_opt = res.x[:2*N].reshape(N, 2)
            r_opt = res.x[2*N:]
            
            # Clip small negative radii if any
            r_opt = np.maximum(r_opt, 0)
            
            score = _evaluate(c_opt, r_opt)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([c_opt, r_opt])
                
        except Exception:
            continue

    # If no valid result found, return grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
