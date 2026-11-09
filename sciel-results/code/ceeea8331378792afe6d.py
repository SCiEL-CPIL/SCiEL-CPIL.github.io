import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 630.0 - 30.0  # 30s safety margin
    
    def compute_overlap_penalty(centers, sides):
        """Compute total overlap penalty for a configuration."""
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0
                # Overlap if both dx < t and dy < t
                ox = max(0.0, t - dx)
                oy = max(0.0, t - dy)
                if ox > 0 and oy > 0:
                    penalty += ox * oy
        # Boundary penalty
        for i in range(n):
            half = sides[i] / 2.0
            bx = max(0.0, -tol - (centers[i, 0] - half)) + max(0.0, (centers[i, 0] + half) - (1.0 + tol))
            by = max(0.0, -tol - (centers[i, 1] - half)) + max(0.0, (centers[i, 1] + half) - (1.0 + tol))
            penalty += bx + by
        return penalty
    
    def objective(x):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        # Penalize negative sides
        sides_pen = np.sum(np.maximum(0, -sides)) ** 2
        overlap_pen = compute_overlap_penalty(centers, sides)
        # We want to maximize sum(sides), so minimize -sum(sides) + penalties
        return -np.sum(sides) + 100.0 * overlap_pen + 10.0 * sides_pen
    
    def is_valid(centers, sides):
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if not np.all(sides > 0):
            return False
        for i in range(n):
            half = sides[i] / 2.0
            if centers[i, 0] - half < -tol or centers[i, 0] + half > 1.0 + tol:
                return False
            if centers[i, 1] - half < -tol or centers[i, 1] + half > 1.0 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if dx < t and dy < t:
                    return False
        return True
    
    # Generate initial configurations
    def make_initial_configs():
        configs = []
        
        # Config 1: 3x3 grid (baseline)
        k = 3
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1)
        sides = np.full(n, side)
        x = np.concatenate([centers.flatten(), sides])
        configs.append(x)
        
        # Config 2: 4x3 arrangement (4 cols, 3 rows, but only 10 squares)
        # Try 4 columns x 3 rows, fill 10 of 12 cells
        k_cols, k_rows = 4, 3
        side = 1.0 / k_cols
        idx = np.arange(n)
        cols = idx % k_cols
        rows = idx // k_cols
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * (1.0 / k_rows)
        centers = np.stack([cx, cy], axis=1)
        sides = np.full(n, side)
        x = np.concatenate([centers.flatten(), sides])
        configs.append(x)
        
        # Config 3: Mixed - 4 large in corners, 6 smaller
        # Large squares of side 0.35 in corners
        large_side = 0.35
        small_side = 0.2
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        # 4 large in corners
        centers[0] = [large_side/2, large_side/2]
        sides[0] = large_side
        centers[1] = [1 - large_side/2, large_side/2]
        sides[1] = large_side
        centers[2] = [large_side/2, 1 - large_side/2]
        sides[2] = large_side
        centers[3] = [1 - large_side/2, 1 - large_side/2]
        sides[3] = large_side
        # 6 smaller in the middle area
        for i in range(4, 10):
            row = (i - 4) // 3
            col = (i - 4) % 3
            centers[i, 0] = 0.5 + (col - 1) * 0.25
            centers[i, 1] = 0.5 + (row - 0.5) * 0.3
            sides[i] = small_side
        x = np.concatenate([centers.flatten(), sides])
        configs.append(x)
        
        # Config 4: Random
        np.random.seed(42)
        for seed in range(20):
            np.random.seed(seed)
            centers = np.random.uniform(0.1, 0.9, size=(n, 2))
            sides = np.random.uniform(0.1, 0.3, size=n)
            x = np.concatenate([centers.flatten(), sides])
            configs.append(x)
        
        # Config 5: 2x5 grid
        k_cols, k_rows = 5, 2
        side = 1.0 / k_cols
        idx = np.arange(n)
        cols = idx % k_cols
        rows = idx // k_cols
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * (1.0 / k_rows)
        centers = np.stack([cx, cy], axis=1)
        sides = np.full(n, side)
        x = np.concatenate([centers.flatten(), sides])
        configs.append(x)
        
        return configs
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    # Baseline score
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    base_centers = np.stack([cx, cy], axis=1)
    base_sides = np.full(n, side)
    best_centers = base_centers.copy()
    best_sides = base_sides.copy()
    best_score = np.sum(base_sides)
    
    configs = make_initial_configs()
    
    for init_idx, x0 in enumerate(configs):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            x_opt = result.x
            centers = x_opt[:2*n].reshape(n, 2)
            sides = x_opt[2*n:]
            
            # Project sides to be positive
            sides = np.maximum(sides, 1e-6)
            
            if is_valid(centers, sides):
                score = np.sum(sides)
                if score > best_score:
                    best_score = score
                    best_centers = centers.copy()
                    best_sides = sides.copy()
        except Exception:
            pass
    
    # Final validation
    if not is_valid(best_centers, best_sides):
        # Fall back to baseline
        k = 3
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
    
    return best_centers.astype(np.float64), best_sides.astype(np.float64)
