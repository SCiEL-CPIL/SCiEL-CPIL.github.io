# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 with pairwise
    disjoint interiors to maximize the sum of their side lengths.

    Strategy:
      1. Build diverse initial configurations.
      2. For each, optimize center positions using a smooth Voronoi-based
         objective where each square's side is limited by its nearest neighbors.
      3. Polish the best result with high-precision bisection.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 590.0  # safety margin before 630s kill

    def check_valid(centers, sides, tol=1e-7):
        """Check if configuration is valid."""
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if not np.all(sides > 0):
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
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    def voronoi_limited_side(centers, i):
        """
        Compute the maximum side length for square i based on its Voronoi cell.
        The square is limited by:
          - Distance to boundaries (left, right, bottom, top)
          - Distance to each other square (in L_infinity metric for axis-aligned)
        """
        cx, cy = centers[i]
        # Boundary limits
        s_left = 2 * cx
        s_right = 2 * (1 - cx)
        s_bottom = 2 * cy
        s_top = 2 * (1 - cy)
        s_max = min(s_left, s_right, s_bottom, s_top)
        
        # Neighbor limits: for each other square j, the separation in x or y
        # limits how large i can be before overlapping
        for j in range(n):
            if j == i:
                continue
            dx = abs(centers[i,0] - centers[j,0])
            dy = abs(centers[i,1] - centers[j,1])
            # If separated in x by dx, then s_i + s_j <= 2*dx
            # We want to maximize s_i, so assume s_j is also at its limit
            # Conservative: s_i <= 2*dx - s_j_limit
            # But s_j depends on i too... Use a simple bound:
            # The gap in x is dx - (s_i + s_j)/2 >= 0
            # So s_i <= 2*dx - s_j
            # To be safe, use s_j <= min(2*cx_j, 2*(1-cx_j), 2*cy_j, 2*(1-cy_j))
            # This is conservative but smooth
            cx_j, cy_j = centers[j]
            s_j_max = min(2*cx_j, 2*(1-cx_j), 2*cy_j, 2*(1-cy_j))
            # Limit from x-separation
            s_i_from_x = max(0, 2*dx - s_j_max)
            # Limit from y-separation
            s_i_from_y = max(0, 2*dy - s_j_max)
            # The square can be limited by either x or y separation
            s_from_j = max(s_i_from_x, s_i_from_y)
            s_max = min(s_max, s_from_j)
        
        return max(s_max, 1e-6)

    def objective_and_grad(x):
        """
        x = [cx_0, cy_0, cx_1, cy_1, ..., cx_9, cy_9] (2n variables)
        Maximize sum of Voronoi-limited sides.
        """
        centers = x.reshape(n, 2)
        sides = np.array([voronoi_limited_side(centers, i) for i in range(n)])
        obj = -np.sum(sides)
        
        # Numerical gradient
        grad = np.zeros(2*n)
        eps = 1e-7
        for k in range(2*n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[k] += eps
            x_minus[k] -= eps
            f_plus = -np.sum(np.array([voronoi_limited_side(x_plus.reshape(n,2), i) for i in range(n)]))
            f_minus = -np.sum(np.array([voronoi_limited_side(x_minus.reshape(n,2), i) for i in range(n)]))
            grad[k] = (f_plus - f_minus) / (2*eps)
        
        return obj, grad

    def generate_initial_configs():
        """Generate diverse initial configurations as center arrays."""
        configs = []
        
        # 1. Grid configuration
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1)
        configs.append(centers.flatten())
        
        # 2. Hand-crafted: one large in corner, others around
        cx = np.array([0.25, 0.75, 0.75, 0.25, 0.75, 0.5, 0.75, 0.625, 0.875, 0.5])
        cy = np.array([0.25, 0.25, 0.75, 0.75, 0.5, 0.75, 0.625, 0.875, 0.5, 0.375])
        centers = np.stack([cx, cy], axis=1)
        configs.append(centers.flatten())
        
        # 3. Random configurations
        np.random.seed(42)
        for trial in range(8):
            cx = np.random.uniform(0.1, 0.9, n)
            cy = np.random.uniform(0.1, 0.9, n)
            centers = np.stack([cx, cy], axis=1)
            configs.append(centers.flatten())
        
        # 4. Two-row layout
        cx = np.array([0.15, 0.35, 0.55, 0.75, 0.15, 0.35, 0.55, 0.75, 0.3, 0.7])
        cy = np.array([0.2, 0.2, 0.2, 0.2, 0.75, 0.75, 0.75, 0.75, 0.5, 0.5])
        centers = np.stack([cx, cy], axis=1)
        configs.append(centers.flatten())
        
        # 5. Three-row layout
        cx = np.array([0.15, 0.5, 0.85, 0.15, 0.5, 0.85, 0.15, 0.5, 0.85, 0.5])
        cy = np.array([0.15, 0.15, 0.15, 0.5, 0.5, 0.5, 0.85, 0.85, 0.85, 0.5])
        centers = np.stack([cx, cy], axis=1)
        configs.append(centers.flatten())
        
        return configs

    def polish(centers, sides, tol=1e-9, max_iter=60):
        """Polish a valid configuration by trying to increase side lengths."""
        centers = centers.copy()
        sides = sides.copy()
        
        for _ in range(max_iter):
            improved = False
            for i in range(n):
                best_delta = 0
                lo, hi = 0.0, 0.02
                for _ in range(max_iter):
                    mid = (lo + hi) / 2
                    new_s = sides[i] + mid
                    if new_s <= 0:
                        hi = mid
                        continue
                    test_sides = sides.copy()
                    test_sides[i] = new_s
                    if check_valid(centers, test_sides, tol=1e-10):
                        lo = mid
                        best_delta = mid
                        improved = True
                    else:
                        hi = mid
                if best_delta > 0:
                    sides[i] += best_delta
            if not improved:
                break
        return centers, sides

    # Main optimization loop
    best_score = -np.inf
    best_result = None

    configs = generate_initial_configs()

    for x0 in configs:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)] * (2*n),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_score:
                centers = result.x.reshape(n, 2)
                # Clamp to [0,1]
                centers = np.clip(centers, 0, 1)
                sides = np.array([voronoi_limited_side(centers, i) for i in range(n)])
                
                if check_valid(centers, sides, tol=1e-7):
                    score = np.sum(sides)
                    if score > best_score:
                        best_score = score
                        best_result = (centers, sides)
        except Exception:
            continue
    
    # Polish the best result
    if best_result is not None:
        centers, sides = best_result
        if time.time() < deadline - 15:
            centers, sides = polish(centers, sides)
            if check_valid(centers, sides, tol=1e-7):
                best_result = (centers, sides)
    
    # Fallback to grid if optimization failed
    if best_result is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        best_result = (centers, sides)
    
    return best_result

# EVOLVE-BLOCK-END
