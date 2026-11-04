"""Circle packing in unit square: 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import time


def _smooth_max(v, k=50.0):
    """Log-sum-exp style smooth maximum of v."""
    m = np.max(v)
    return m + np.log(np.sum(np.exp(k * (v - m)))) / k


def _objective_and_grad(x, k=50.0):
    """
    x: (2N, 2) positions
    Returns (objective, gradient) where objective = -sum(radii) + penalty.
    
    We maximize sum(r_i) subject to:
      - circles inside [0,1]^2: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
      - circles disjoint: ||x_i - x_j|| >= r_i + r_j
    
    We use a penalty-based approach:
    - For each circle, r_i = min(x_i, 1-x_i, y_i, 1-y_i, min_j (||x_i-x_j|| - r_j) for j!=i)
    But this is non-smooth. Instead, we optimize positions and use a smooth penalty.
    
    Alternative: optimize (x, y, r) directly with constraints.
    """
    pass


def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Strategy: multi-start with gradient-based optimization
    # Variables: x[0:N], y[0:N], r[0:N] -> total 3N = 78 variables
    # Objective: maximize sum(r)
    # Constraints:
    #   r_i >= 0
    #   r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i  (inside square)
    #   (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2  (disjoint)
    
    # We'll use a penalty method with smooth approximation for the disjointness constraint.
    # Or better: use scipy.optimize.minimize with SLSQP or trust-constr for constrained optimization.
    
    # Let's try a different approach: 
    # 1. Start from a reasonable initial configuration
    # 2. Use gradient descent with penalty terms
    
    best_score = 0.0
    best_x = None
    
    # Initial configuration: grid-based
    side = 5  # 5x5 grid = 25 cells, we need 26
    # Use a 5x5 grid plus one extra circle in the center of a cell (split it)
    
    def make_initial(seed=0):
        rng = np.random.RandomState(seed)
        # Create a grid of 5x5 = 25 positions
        positions = []
        for i in range(5):
            for j in range(5):
                positions.append([(i + 0.5) / 5.0, (j + 0.5) / 5.0])
        # Add 26th circle: place it in a random position
        positions.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)])
        
        # Initial radii: small
        base_r = 0.1
        radii = np.full(N, base_r * 0.5)
        
        x = np.array([p[0] for p in positions])
        y = np.array([p[1] for p in positions])
        
        return x, y, radii
    
    def compute_radii(x, y, positions):
        """Compute max feasible radii given positions."""
        n = len(x)
        r = np.zeros(n)
        # Max radius from walls
        wall_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_r)
        r = np.maximum(r, 0)
        # For each pair, r_i + r_j <= dist_ij
        # This is a system of linear inequalities in r.
        # Use a simple iterative approach:
        # r_i <= min(wall_r_i, min_j (dist_ij - r_j))
        # Iterate a few times
        for _ in range(20):
            new_r = wall_r.copy()
            for i in range(n):
                for j in range(n):
                    if i != j:
                        d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                        new_r[i] = min(new_r[i], max(0, d - r[j]))
            if np.allclose(new_r, r, atol=1e-10):
                break
            r = new_r
        return np.maximum(r, 0)
    
    def score(x, y):
        r = compute_radii(x, y, None)
        return np.sum(r), r
    
    # Multi-start optimization
    n_starts = 8
    for start in range(n_starts):
        if time.time() > deadline:
            break
        
        x, y, r0 = make_initial(seed=start * 42 + 7)
        
        # Optimize positions to maximize sum of radii
        # Use a simple gradient ascent with penalty
        
        def objective(params):
            """params: 2N vector [x0,...,xN-1, y0,...,yN-1]"""
            x = params[:N]
            y = params[N:]
            # Keep in bounds
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            r = compute_radii(x, y, None)
            s = np.sum(r)
            # Penalty for being out of bounds (shouldn't happen due to clip)
            return -s
        
        # Use Nelder-Mead for robustness
        x0 = np.concatenate([x, y])
        
        result = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        x_opt = result.x[:N]
        y_opt = result.x[N:]
        x_opt = np.clip(x_opt, 0.01, 0.99)
        y_opt = np.clip(y_opt, 0.01, 0.99)
        
        r_opt = compute_radii(x_opt, y_opt, None)
        s = np.sum(r_opt)
        
        if s > best_score:
            best_score = s
            best_x = (x_opt, y_opt, r_opt)
        
        # Polish with local refinement
        if time.time() > deadline:
            break
        
        # Try small perturbations
        rng = np.random.RandomState(start * 100 + 1)
        for _ in range(100):
            if time.time() > deadline:
                break
            x_p = x_opt + rng.normal(0, 0.01, N)
            y_p = y_opt + rng.normal(0, 0.01, N)
            x_p = np.clip(x_p, 0.01, 0.99)
            y_p = np.clip(y_p, 0.01, 0.99)
            r_p = compute_radii(x_p, y_p, None)
            s_p = np.sum(r_p)
            if s_p > best_score:
                best_score = s_p
                best_x = (x_p, y_p, r_p)
                x_opt, y_opt = x_p, y_p
    
    if best_x is None:
        # Fallback to grid
        x, y, r0 = make_initial(seed=0)
        r0 = compute_radii(x, y, None)
        best_x = (x, y, r0)
    
    x, y, r = best_x
    
    # Final validation and cleanup
    # Ensure all circles are valid
    valid_mask = np.ones(N, dtype=bool)
    for i in range(N):
        if r[i] < 1e-12 or x[i] < 0 or x[i] > 1 or y[i] < 0 or y[i] > 1:
            valid_mask[i] = False
        if x[i] - r[i] < -1e-12 or x[i] + r[i] > 1 + 1e-12:
            valid_mask[i] = False
        if y[i] - r[i] < -1e-12 or y[i] + r[i] > 1 + 1e-12:
            valid_mask[i] = False
    
    # Check disjointness
    for i in range(N):
        if not valid_mask[i]:
            continue
        for j in range(i + 1, N):
            if not valid_mask[j]:
                continue
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - 1e-12:
                valid_mask[i] = False
                valid_mask[j] = False
    
    # If we lost too many circles, fall back to a safe grid
    if np.sum(valid_mask) < N:
        # Safe grid: 5x5 grid with 25 circles + 1 smaller
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
        # Add 26th: find a spot
        # Place it in the middle of the largest empty space
        # For simplicity, place it at (0.5, 0.5) with small radius if not occupied
        # Actually, just use 25 circles from grid and one very small one
        # Better: use a 6x5 grid = 30 cells, take 26
        side_h, side_w = 6, 5
        cells = []
        for i in range(side_w):
            for j in range(side_h):
                cx = (i + 0.5) / side_w
                cy = (j + 0.5) / side_h
                r_max = min(cx, 1 - cx, cy, 1 - cy)
                # Also check against neighbors
                r_val = r_max * 0.95
                cells.append([cx, cy, r_val])
        # Take first 26
        cells = cells[:N]
        # Verify and adjust
        x = np.array([c[0] for c in cells])
        y = np.array([c[1] for c in cells])
        r = np.array([c[2] for c in cells])
        # Recompute max radii
        r = compute_radii(x, y, None)
    
    result = np.column_stack([x, y, r])
    return result
# EVOLVE-BLOCK-END
