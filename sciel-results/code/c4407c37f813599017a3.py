# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


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

    def compute_penalty_and_obj(x, penalty_weight=1e5):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        
        # Ensure sides are positive
        sides = np.maximum(sides, 1e-10)
        
        obj = np.sum(sides)
        
        penalty = 0.0
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            # Boundary violations
            d_left = cx - s/2 + tol
            if d_left < 0: penalty += d_left**2
            d_right = 1 + tol - (cx + s/2)
            if d_right < 0: penalty += d_right**2
            d_bottom = cy - s/2 + tol
            if d_bottom < 0: penalty += d_bottom**2
            d_top = 1 + tol - (cy + s/2)
            if d_top < 0: penalty += d_top**2
        
        # Overlap violations
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i][0] - centers[j][0])
                dy = abs(centers[i][1] - centers[j][1])
                t = (sides[i] + sides[j])/2 - tol
                if dx < t and dy < t:
                    px = t - dx
                    py = t - dy
                    penalty += px * py
        
        return -(obj) + penalty_weight * penalty

    def is_valid(centers, sides):
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
                dx = abs(centers[i][0] - centers[j][0])
                dy = abs(centers[i][1] - centers[j][1])
                t = (sides[i] + sides[j])/2 - tol
                if dx < t and dy < t:
                    return False
        return True

    def get_score(centers, sides):
        return np.sum(sides)

    def make_grid_init(kx, ky, count):
        idx = np.arange(count)
        cols = idx % kx
        rows = idx // kx
        side_x = 1.0 / kx
        side_y = 1.0 / ky
        cx = (cols + 0.5) * side_x
        cy = (rows + 0.5) * side_y
        centers = np.stack([cx, cy], axis=1)
        sides = np.full(count, min(side_x, side_y) * 0.9)
        return centers.flatten(), sides

    def make_random_init(scale=0.1):
        centers = np.random.rand(n, 2)
        sides = np.random.rand(n) * scale + 0.01
        return centers.flatten(), sides

    inits = []
    inits.append(make_grid_init(3, 4, 10))
    inits.append(make_grid_init(4, 3, 10))
    inits.append(make_grid_init(2, 5, 10))
    inits.append(make_grid_init(5, 2, 10))
    for _ in range(3):
        inits.append(make_random_init())
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 1))

    best_centers = None
    best_sides = None
    best_score = -np.inf

    for centers_init, sides_init in inits:
        x0 = np.concatenate([centers_init, sides_init])
        
        res = minimize(compute_penalty_and_obj, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        x_opt = res.x
        centers = x_opt[:2*n].reshape(n, 2)
        sides = x_opt[2*n:]
        sides = np.maximum(sides, 1e-10)
        
        if is_valid(centers, sides):
            score = get_score(centers, sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()

    if best_centers is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(10, side * 0.9, dtype=np.float64)
        
        if not is_valid(best_centers, best_sides):
             best_sides = best_sides * 0.5

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
