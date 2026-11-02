"""Program for packing 50 equal circles in a unit square; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Optimizes circle packing using L-BFGS-B to minimize constraint violations.
    Treats the problem as a rigid-body assembly where we minimize the 'stress'
    from overlapping circles and boundary violations.
    """
    
    def hex_initial(n):
        """Hexagonal lattice, centered and scaled to fit in [-0.5,0.5]^2."""
        pts = []
        for row in range(12):
            for col in range(12):
                x = col + 0.5 * (row % 2)
                y = row * np.sqrt(3) / 2.0
                pts.append([x, y])
        pts = np.array(pts[:n])
        pts -= pts.mean(axis=0)
        max_abs = np.max(np.abs(pts))
        if max_abs > 0:
            pts *= (0.48 / max_abs)
        return pts
    
    def grid_initial(n):
        """Grid layout."""
        m = int(np.ceil(np.sqrt(n)))
        xs, ys = np.meshgrid(np.linspace(-0.45, 0.45, m), np.linspace(-0.45, 0.45, m))
        pts = np.column_stack([xs.ravel(), ys.ravel()])[:n]
        return pts

    def compute_radius(pts):
        """Compute the packing radius for a configuration."""
        # Pairwise distances
        dists = np.sqrt(((pts[:, np.newaxis] - pts[np.newaxis]) ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        # Boundary distances
        r_bound = 0.5 - np.max(np.abs(pts), axis=1)
        r_bound = np.min(r_bound)
        return min(r_pair, r_bound)

    def objective_and_grad(x_flat, r):
        """
        Objective: Sum of squared violations of distance and boundary constraints.
        Gradient: Analytic gradient of the objective.
        """
        pts = x_flat.reshape(N, 2)
        # Calculate distances
        dists = np.sqrt(((pts[:, np.newaxis] - pts[np.newaxis]) ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        
        # Violations: max(0, 2r - dist)
        # We only care about pairs where dist < 2r
        # For efficiency and smoothness, we use a soft penalty or just sum squares of violations
        # Let v_ij = max(0, 2r - d_ij)
        # Obj = sum v_ij^2 + sum boundary_violations^2
        
        # Boundary violations: max(0, |x_i| - (0.5 - r))
        # Note: The square is [-0.5, 0.5]^2.
        # Distance to boundary is min(0.5 - x, 0.5 + x).
        # Violation if radius > distance to boundary.
        # dist_to_bound_i = min(0.5 - pts[i,0], 0.5 + pts[i,0], 0.5 - pts[i,1], 0.5 + pts[i,1])
        
        # To make it differentiable, we can use the same max(0, ...) trick.
        # However, L-BFGS-B handles bounds naturally. 
        # If we use bounds in minimize, we don't need boundary penalty in objective.
        # But we DO need to ensure circles don't overlap.
        
        # Let's use bounds in the optimizer for the square [-0.5+r, 0.5-r]^2?
        # No, the radius r is fixed during the minimization for a specific binary search step.
        # The centers must be in [-0.5+r, 0.5-r] to be inside the square with radius r.
        
        # So, we can set bounds in `minimize` to be [-0.5+r, 0.5-r].
        # Then we only need to minimize overlap violations.
        
        # Overlap violations
        # v = 2r - dist
        # If v > 0, penalty is v^2.
        # Gradient of v^2 w.r.t x_i:
        # d/dx_i ( (2r - ||x_i - x_j||)^2 ) = 2 * (2r - d) * (-1) * (x_i - x_j)/d
        # = 2 * (2r - d) * (x_j - x_i)/d
        
        obj_val = 0.0
        grad = np.zeros_like(x_flat)
        
        # Vectorized calculation of gradient is complex due to the max(0, ...) non-linearity.
        # We will iterate over pairs. N=50, so 1225 pairs. This is fast enough.
        for i in range(N):
            for j in range(i + 1, N):
                d_vec = pts[i] - pts[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    # Perturb if coincident
                    d_vec = np.array([1e-5, 1e-5])
                    d = np.linalg.norm(d_vec)
                
                if d < 2 * r:
                    violation = 2 * r - d
                    obj_val += violation ** 2
                    # Gradient contribution
                    # d_obj/d_x_i = 2 * violation * (x_i - x_j) / d
                    # d_obj/d_x_j = 2 * violation * (x_j - x_i) / d
                    factor = 2 * violation / d
                    grad[i*2:i*2+2] += factor * d_vec
                    grad[j*2:j*2+2] -= factor * d_vec
        
        return obj_val, grad

    def solve_for_radius(pts_init, r):
        """
        Try to find a configuration with radius r using L-BFGS-B.
        Returns (pos, success).
        """
        x0 = pts_init.flatten()
        # Bounds: centers must be in [-0.5+r, 0.5-r]
        bound_val = 0.5 - r
        bounds = [(-bound_val, bound_val)] * (2 * N)
        
        try:
            res = minimize(
                objective_and_grad, 
                x0, 
                args=(r,), 
                method='L-BFGS-B', 
                jac=True, 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Check if solution is feasible
            pos = res.x.reshape(N, 2)
            # Verify overlaps
            dists = np.sqrt(((pos[:, np.newaxis] - pos[np.newaxis]) ** 2).sum(-1))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            if min_dist >= 2 * r - 1e-8:
                return pos, True
            else:
                return pos, False
        except Exception:
            return pts_init, False

    # Initial configurations
    configs = [
        hex_initial(N),
        grid_initial(N)
    ]
    
    # Add jittered versions
    rng = np.random.default_rng(42)
    for _ in range(3):
        base = hex_initial(N)
        jitter = rng.normal(0, 0.05, size=(N, 2))
        cfg = base + jitter
        cfg = np.clip(cfg, -0.49, 0.49)
        configs.append(cfg)
    
    best_pos = None
    best_r = 0.0
    
    for cfg in configs:
        # Estimate initial radius
        r0 = compute_radius(cfg)
        
        # Binary search for max radius
        # Start slightly below r0 to ensure feasibility, go up to 0.1 (theoretical max for 50 is ~0.071)
        r_lo = r0 * 0.95
        r_hi = 0.10 # Safe upper bound
        
        # Ensure r_lo is feasible
        pos_lo, feas_lo = solve_for_radius(cfg, r_lo)
        if not feas_lo:
            # If even r_lo is not feasible, try to reduce r_lo
            # But usually hex/grid are feasible for their natural radius
            # If not, skip this config
            continue
        
        # Binary search
        for _ in range(30):
            r_mid = (r_lo + r_hi) / 2
            pos_mid, feas_mid = solve_for_radius(pos_lo, r_mid)
            
            if feas_mid:
                r_lo = r_mid
                pos_lo = pos_mid
                if r_mid > best_r:
                    best_r = r_mid
                    best_pos = pos_mid.copy()
            else:
                r_hi = r_mid
            
            if r_hi - r_lo < 1e-6:
                break
    
    # Safety fallback
    if best_pos is None or np.any(~np.isfinite(best_pos)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    # Final check: no coincident centers
    dists = np.sqrt(((best_pos[:, np.newaxis] - best_pos[np.newaxis]) ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    return best_pos
# EVOLVE-BLOCK-END
