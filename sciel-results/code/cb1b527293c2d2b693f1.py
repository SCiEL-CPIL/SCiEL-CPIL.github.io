# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Uses Differential Evolution for global search and SLSQP for local refinement.
    Returns (centers, sides).
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 570  # safety margin before 630s kill

    def violation(x):
        """Return total constraint violation (0 if feasible)."""
        cx, cy, s = x[:n], x[n:2*n], x[2*n:3*n]
        v = 0.0
        # Boundary: each square inside [-tol, 1+tol]^2
        for i in range(n):
            v += max(0.0, (cx[i] - s[i]/2) - (-tol))  # left edge < -tol
            v += max(0.0, (cx[i] + s[i]/2) - (1.0 + tol))  # right edge > 1+tol
            v += max(0.0, (cy[i] - s[i]/2) - (-tol))  # bottom edge < -tol
            v += max(0.0, (cy[i] + s[i]/2) - (1.0 + tol))  # top edge > 1+tol
        # Non-overlap: for each pair, |dx| >= t or |dy| >= t where t=(si+sj)/2-tol
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                # Violation if both dx < t and dy < t
                if dx < t and dy < t:
                    v += (t - dx) + (t - dy)
        return v

    def objective(x):
        """Negative sum of side lengths (we minimize -sum)."""
        s = x[2*n:3*n]
        return -np.sum(s)

    def penalized_objective(x):
        """Objective with penalty for constraint violation."""
        return objective(x) + 100.0 * violation(x)

    def is_feasible(x):
        return violation(x) < 1e-6

    def refine(x_init):
        """Refine a feasible or near-feasible point using SLSQP."""
        bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(1e-8, 1.0)] * n
        try:
            res = minimize(
                objective, x_init, method='SLSQP', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            x_res = res.x
            x_res[2*n:3*n] = np.maximum(x_res[2*n:3*n], 1e-8)
            if is_feasible(x_res):
                return x_res
        except Exception:
            pass
        return None

    best_x = None
    best_obj = np.inf

    # 1. Try Differential Evolution for global search
    # Bounds for DE
    de_bounds = [(0.0, 1.0)] * (2 * n) + [(0.01, 0.5)] * n
    
    try:
        if time.time() < deadline - 30:
            # DE can be slow, so limit iterations
            de_result = differential_evolution(
                penalized_objective, 
                de_bounds, 
                seed=42, 
                maxiter=50, 
                popsize=15, 
                tol=1e-6,
                polish=False # We will polish manually
            )
            if de_result.success or de_result.fun < best_obj:
                x_de = de_result.x
                x_de[2*n:3*n] = np.maximum(x_de[2*n:3*n], 1e-8)
                if is_feasible(x_de):
                    best_x = x_de
                    best_obj = objective(x_de)
                else:
                    # Try to refine the DE result even if slightly infeasible
                    x_ref = refine(x_de)
                    if x_ref is not None:
                        obj_ref = objective(x_ref)
                        if obj_ref < best_obj:
                            best_obj = obj_ref
                            best_x = x_ref
    except Exception:
        pass

    # 2. Try specific heuristic: 3x3 grid of large squares + 1 small
    # This configuration is known to be strong.
    # 9 squares of side ~0.32, 1 square of side ~0.25
    # Let's generate a few variations of this layout and refine them.
    
    rng = np.random.default_rng(42)
    
    # Heuristic 1: 3x3 grid centered, small square in corner
    # Large squares side s, small square side t
    # To fit 3x3 of s in a 1x1 box, we need 3s <= 1 + gap? 
    # If we place them in a grid with gaps, we can fit the small one.
    
    for s_large in [0.32, 0.33, 0.34]:
        if time.time() > deadline:
            break
        # Construct 3x3 grid
        cx_large = []
        cy_large = []
        s_large_arr = []
        for i in range(3):
            for j in range(3):
                cx_large.append((j + 0.5) * s_large)
                cy_large.append((i + 0.5) * s_large)
                s_large_arr.append(s_large)
        
        # Place 10th square in top-right corner, adjusting for overlap
        # The 3x3 grid occupies [0, 3*s_large] x [0, 3*s_large] if aligned to 0.
        # Let's align the grid to the bottom-left, but shift it to make space.
        # Actually, let's just place the 10th square in the remaining space.
        # If 3*s_large < 1, there is space.
        # Let's try placing the 10th square at (1 - t/2, 1 - t/2)
        # and check if it overlaps.
        
        for t_small in [0.2, 0.25, 0.3]:
            if time.time() > deadline:
                break
            
            x_init = np.zeros(3 * n)
            # First 9
            x_init[:9] = cx_large
            x_init[9:18] = cy_large
            x_init[18:27] = s_large_arr
            
            # 10th
            x_init[9] = 1.0 - t_small/2
            x_init[18] = 1.0 - t_small/2
            x_init[27] = t_small
            
            # Check if this is a valid start (it might overlap with the 9th square of the grid)
            # If it overlaps, SLSQP will try to fix it, but it might get stuck.
            # Let's refine it.
            x_ref = refine(x_init)
            if x_ref is not None:
                obj_ref = objective(x_ref)
                if obj_ref < best_obj:
                    best_obj = obj_ref
                    best_x = x_ref

    # 3. Random starts with SLSQP (fallback)
    n_starts = 20
    for start in range(n_starts):
        if time.time() > deadline:
            break
            
        cx_init = rng.uniform(0.1, 0.9, n)
        cy_init = rng.uniform(0.1, 0.9, n)
        s_init = rng.uniform(0.05, 0.35, n)
        for i in range(n):
            s_init[i] = min(s_init[i], 0.5)
            cx_init[i] = np.clip(cx_init[i], s_init[i]/2 - tol, 1 - s_init[i]/2 + tol)
            cy_init[i] = np.clip(cy_init[i], s_init[i]/2 - tol, 1 - s_init[i]/2 + tol)
        x_init = np.concatenate([cx_init, cy_init, s_init])
        
        x_ref = refine(x_init)
        if x_ref is not None:
            obj_ref = objective(x_ref)
            if obj_ref < best_obj:
                best_obj = obj_ref
                best_x = x_ref

    # If no valid solution found, return a default valid one (4x4 grid)
    if best_x is None:
        side = 1.0 / 4.0
        idx = np.arange(n)
        cols = idx % 4
        rows = idx // 4
        cx0 = (cols + 0.5) * side
        cy0 = (rows + 0.5) * side
        s0 = np.full(n, side)
        best_x = np.concatenate([cx0, cy0, s0])

    cx = best_x[:n]
    cy = best_x[n:2*n]
    s = best_x[2*n:3*n]

    # Final cleanup
    s = np.maximum(s, 1e-8)
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)

    return centers, sides


# EVOLVE-BLOCK-END
