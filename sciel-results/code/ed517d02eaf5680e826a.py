# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


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
    deadline = time.time() + 570  # safety margin before 600s limit

    def _is_feasible(centers, sides, tol=1e-7):
        """Check if the configuration is feasible (vectorized)."""
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # Boundary check
        if np.any(centers[:, 0] - sides/2 < -tol) or np.any(centers[:, 0] + sides/2 > 1 + tol):
            return False
        if np.any(centers[:, 1] - sides/2 < -tol) or np.any(centers[:, 1] + sides/2 > 1 + tol):
            return False
        # Overlap check (vectorized)
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True

    def _objective(x):
        """
        Objective function for minimization.
        x = [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ..., cx_9, cy_9, s_9]
        We minimize -sum(sides) + penalty.
        """
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        
        # Ensure sides are positive (bounds will handle this, but just in case)
        if np.any(sides <= 0):
            return 1e6
            
        sum_sides = np.sum(sides)
        
        penalty = 0.0
        
        # Boundary violations
        for i in range(n):
            cx, cy, s = centers[i, 0], centers[i, 1], sides[i]
            # Left
            if cx - s/2 < 0:
                penalty += (cx - s/2) ** 2
            # Right
            if cx + s/2 > 1:
                penalty += (cx + s/2 - 1) ** 2
            # Bottom
            if cy - s/2 < 0:
                penalty += (cy - s/2) ** 2
            # Top
            if cy + s/2 > 1:
                penalty += (cy + s/2 - 1) ** 2
        
        # Overlap violations
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                tx = (sides[i] + sides[j]) / 2 - dx
                ty = (sides[i] + sides[j]) / 2 - dy
                
                # If they overlap, tx > 0 and ty > 0
                # We penalize the penetration depth
                if tx > 0:
                    penalty += tx ** 2
                if ty > 0:
                    penalty += ty ** 2
        
        # We want to maximize sum_sides, so minimize -sum_sides
        # Penalty must be large enough to dominate sum_sides when infeasible
        # Max sum_sides is 10. Max penalty can be large.
        # Use a large multiplier for penalty to ensure feasibility is prioritized
        return -sum_sides + penalty * 10.0

    # Bounds for each parameter
    # cx, cy in [0, 1]
    # s in [1e-6, 1] (s cannot be 1 if others exist, but bounds are loose)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx
        bounds.append((0.0, 1.0))  # cy
        bounds.append((1e-6, 1.0)) # s

    # Initial best from analytical construction
    centers_init = np.zeros((n, 2), dtype=np.float64)
    sides_init = np.zeros(n, dtype=np.float64)
    
    # Square 0: top-left, side 0.5
    centers_init[0] = [0.25, 0.75]
    sides_init[0] = 0.5
    
    # Square 1: bottom-right, side 0.5
    centers_init[1] = [0.75, 0.25]
    sides_init[1] = 0.5
    
    # Squares 2-5: Top-right 2x2 grid (4 squares of side 0.25)
    for i in range(4):
        col = i % 2
        row = i // 2
        centers_init[2 + i] = [0.5 + (col + 0.5) * 0.25, 0.5 + (row + 0.5) * 0.25]
        sides_init[2 + i] = 0.25
        
    # Squares 6-9: Bottom-left 2x2 grid (4 squares of side 0.25)
    for i in range(4):
        col = i % 2
        row = i // 2
        centers_init[6 + i] = [(col + 0.5) * 0.25, (row + 0.5) * 0.25]
        sides_init[6 + i] = 0.25
        
    best_centers = centers_init.copy()
    best_sides = sides_init.copy()
    best_obj = _objective(np.concatenate([best_centers.flatten(), best_sides]))

    def _update_best(centers, sides):
        nonlocal best_centers, best_sides, best_obj
        if _is_feasible(centers, sides):
            x = np.concatenate([centers.flatten(), sides])
            obj = _objective(x)
            if obj < best_obj:
                best_centers = centers.copy()
                best_sides = sides.copy()
                best_obj = obj

    _update_best(best_centers, best_sides)

    # Generate multiple diverse initial configurations for multi-start
    def _make_config_2big_8small():
        """2 large 0.5 squares + 8 small 0.25 squares."""
        c = np.zeros((n, 2))
        s = np.zeros(n)
        c[0] = [0.25, 0.75]; s[0] = 0.5
        c[1] = [0.75, 0.25]; s[1] = 0.5
        for i in range(4):
            col = i % 2
            row = i // 2
            c[2+i] = [0.5 + (col + 0.5) * 0.25, 0.5 + (row + 0.5) * 0.25]
            s[2+i] = 0.25
        for i in range(4):
            col = i % 2
            row = i // 2
            c[6+i] = [(col + 0.5) * 0.25, (row + 0.5) * 0.25]
            s[6+i] = 0.25
        return c, s

    def _make_config_3big_7small():
        """3 larger squares + 7 smaller squares."""
        c = np.zeros((n, 2))
        s = np.zeros(n)
        # Three squares of side ~0.4
        c[0] = [0.2, 0.8]; s[0] = 0.4
        c[1] = [0.8, 0.6]; s[1] = 0.4
        c[2] = [0.5, 0.2]; s[2] = 0.4
        # Seven squares of side ~0.2
        small_positions = [
            [0.1, 0.4], [0.4, 0.4], [0.7, 0.4],
            [0.1, 0.1], [0.4, 0.1], [0.7, 0.1], [0.9, 0.9]
        ]
        for i, pos in enumerate(small_positions):
            c[3+i] = pos
            s[3+i] = 0.2
        return c, s

    def _make_config_grid():
        """3x3 grid with one extra square."""
        c = np.zeros((n, 2))
        s = np.zeros(n)
        # 3x3 grid of squares with side 1/3
        for i in range(9):
            col = i % 3
            row = i // 3
            c[i] = [(col + 0.5) / 3, (row + 0.5) / 3]
            s[i] = 1/3
        # 10th square in corner
        c[9] = [0.05, 0.95]
        s[9] = 0.1
        return c, s

    def _make_config_perturbed(base_c, base_s, rng, scale=0.05):
        """Perturb a base configuration."""
        c = base_c.copy()
        s = base_s.copy()
        c += rng.normal(0, scale, size=c.shape)
        s *= rng.uniform(0.9, 1.1, size=s.shape)
        # Clamp to valid range
        c[:, 0] = np.clip(c[:, 0], 0, 1)
        c[:, 1] = np.clip(c[:, 1], 0, 1)
        s = np.clip(s, 1e-6, 1)
        return c, s

    # Multi-start local optimization
    def _local_optimize(x0, method='L-BFGS-B'):
        """Run local optimization from initial point x0."""
        try:
            if method == 'L-BFGS-B':
                res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
            elif method == 'Nelder-Mead':
                res = minimize(_objective, x0, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            else:
                res = minimize(_objective, x0, method='Powell',
                              options={'maxiter': 3000, 'ftol': 1e-12})
            
            if res.success or res.fun < _objective(x0):
                c_res = res.x[:2*n].reshape(n, 2)
                s_res = res.x[2*n:]
                _update_best(c_res, s_res)
                return res.x
        except Exception:
            pass
        return x0

    # Generate diverse starting points
    rng = np.random.RandomState(42)
    base_configs = [
        _make_config_2big_8small(),
        _make_config_3big_7small(),
        _make_config_grid(),
    ]
    
    start_points = []
    for base_c, base_s in base_configs:
        start_points.append(np.concatenate([base_c.flatten(), base_s]))
        # Add perturbed versions
        for _ in range(3):
            pc, ps = _make_config_perturbed(base_c, base_s, rng)
            start_points.append(np.concatenate([pc.flatten(), ps]))
    
    # 1. Global Search with Differential Evolution (shorter to save time)
    if time.time() < deadline - 120:
        try:
            de_res = differential_evolution(_objective, bounds, seed=42, maxiter=300, 
                                            popsize=15, tol=1e-8, polish=False)
            if de_res.success:
                x_res = de_res.x
                c_res = x_res[:2*n].reshape(n, 2)
                s_res = x_res[2*n:]
                _update_best(c_res, s_res)
                start_points.append(x_res)
        except Exception as e:
            print(f"DE failed: {e}")

    # 2. Multi-start local optimization
    for sp in start_points:
        if time.time() > deadline - 30:
            break
        _local_optimize(sp, 'L-BFGS-B')
    
    # 3. Additional refinement with different methods
    if time.time() < deadline - 15:
        x0 = np.concatenate([best_centers.flatten(), best_sides])
        _local_optimize(x0, 'Nelder-Mead')
        _local_optimize(x0, 'Powell')

    # Final check: ensure the returned solution is strictly feasible
    if not _is_feasible(best_centers, best_sides):
        best_centers = centers_init.copy()
        best_sides = sides_init.copy()

    return best_centers, best_sides


# EVOLVE-BLOCK-END
