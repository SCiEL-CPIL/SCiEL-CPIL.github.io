# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import linprog

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4.
    Returns array of shape (21, 3) with [x, y, r] for each circle.
    The bounding rectangle must have perimeter at most 4.
    
    Approach: 
    1. Iterate over aspect ratios.
    2. For each aspect ratio, use multiple initial configurations.
    3. For each configuration, iteratively:
       a. Compute Delaunay triangulation of centers.
       b. Solve LP to maximize sum of radii given fixed centers and box.
       c. Update centers using gradient ascent on the LP duals (shadow prices).
    4. Return the best found packing.
    """
    N = 21
    
    def get_delaunay_pairs(centers):
        """Get Delaunay neighbor pairs from center positions."""
        try:
            tri = Delaunay(centers)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        edges.add((min(a, b), max(a, b)))
            return list(edges)
        except Exception:
            # Fallback: all pairs if Delaunay fails (expensive but robust)
            return [(i, j) for i in range(len(centers)) for j in range(i + 1, len(centers))]

    def solve_lp(centers, W, H, delaunay_edges):
        """
        Solve LP for optimal radii given fixed centers and box size.
        Returns (radii, duals).
        """
        n = len(centers)
        c = -np.ones(n)
        A_ub_rows = []
        b_ub_vals = []
        
        # Pair constraints: r_i + r_j <= d_ij
        for i, j in delaunay_edges:
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub_rows.append(row)
            b_ub_vals.append(d)
        
        # Boundary constraints: r_i <= dist_to_boundary
        for i in range(n):
            # Left
            row = np.zeros(n); row[i] = 1
            A_ub_rows.append(row); b_ub_vals.append(centers[i][0])
            # Right
            row = np.zeros(n); row[i] = 1
            A_ub_rows.append(row); b_ub_vals.append(W - centers[i][0])
            # Bottom
            row = np.zeros(n); row[i] = 1
            A_ub_rows.append(row); b_ub_vals.append(centers[i][1])
            # Top
            row = np.zeros(n); row[i] = 1
            A_ub_rows.append(row); b_ub_vals.append(H - centers[i][1])
            
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        bounds = [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                duals = res.ineqlin.marginals if hasattr(res, 'ineqlin') else None
                return res.x, duals
        except Exception:
            pass
        return None, None

    def compute_forces(centers, radii, W, H, delaunay_edges, duals):
        """
        Compute forces on centers using LP dual values.
        """
        n = len(centers)
        forces = np.zeros_like(centers)
        
        if duals is not None and len(duals) > 0:
            num_pair_constraints = len(delaunay_edges)
            
            # Pair constraints
            for k, (i, j) in enumerate(delaunay_edges):
                if k < len(duals):
                    dual_val = duals[k]
                    if dual_val > 1e-8:
                        d_vec = centers[i] - centers[j]
                        d = np.linalg.norm(d_vec)
                        if d > 1e-10:
                            direction = d_vec / d
                            forces[i] += direction * dual_val
                            forces[j] -= direction * dual_val
            
            # Boundary constraints
            offset = num_pair_constraints
            for i in range(n):
                # Left: r_i <= x_i
                idx = offset + 4 * i
                if idx < len(duals):
                    dual_val = duals[idx]
                    if dual_val > 1e-8:
                        forces[i][0] += dual_val
                # Right: r_i <= W - x_i
                idx = offset + 4 * i + 1
                if idx < len(duals):
                    dual_val = duals[idx]
                    if dual_val > 1e-8:
                        forces[i][0] -= dual_val
                # Bottom: r_i <= y_i
                idx = offset + 4 * i + 2
                if idx < len(duals):
                    dual_val = duals[idx]
                    if dual_val > 1e-8:
                        forces[i][1] += dual_val
                # Top: r_i <= H - y_i
                idx = offset + 4 * i + 3
                if idx < len(duals):
                    dual_val = duals[idx]
                    if dual_val > 1e-8:
                        forces[i][1] -= dual_val
        else:
            # Fallback: heuristic forces for tight constraints
            for i, j in delaunay_edges:
                d_vec = centers[i] - centers[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    continue
                if abs(radii[i] + radii[j] - d) < 1e-5:
                    direction = d_vec / d
                    forces[i] += direction
                    forces[j] -= direction
            
            for i in range(n):
                if abs(radii[i] - centers[i][0]) < 1e-5:
                    forces[i][0] += 1.0
                if abs(radii[i] - (W - centers[i][0])) < 1e-5:
                    forces[i][0] -= 1.0
                if abs(radii[i] - centers[i][1]) < 1e-5:
                    forces[i][1] += 1.0
                if abs(radii[i] - (H - centers[i][1])) < 1e-5:
                    forces[i][1] -= 1.0
        
        return forces

    def optimize_packing(centers_init, W, H, max_iters=300):
        """
        Iteratively optimize centers to maximize sum of radii.
        """
        centers = centers_init.copy()
        n = len(centers)
        best_val = -np.inf
        best_centers = centers.copy()
        best_radii = None
        
        step_size = 0.05
        stall_count = 0
        delaunay_edges = get_delaunay_pairs(centers)
        
        for it in range(max_iters):
            radii, duals = solve_lp(centers, W, H, delaunay_edges)
            if radii is None:
                # If LP fails, try to separate overlapping centers
                centers += np.random.normal(0, 0.01, size=centers.shape)
                centers[:, 0] = np.clip(centers[:, 0], 0, W)
                centers[:, 1] = np.clip(centers[:, 1], 0, H)
                delaunay_edges = get_delaunay_pairs(centers)
                continue
            
            val = np.sum(radii)
            if val > best_val:
                best_val = val
                best_centers = centers.copy()
                best_radii = radii.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            if stall_count > 50:
                # Shake to escape local minima
                shake_mag = 0.01 * (stall_count / 50.0)
                centers += np.random.normal(0, shake_mag, size=centers.shape)
                centers[:, 0] = np.clip(centers[:, 0], 0, W)
                centers[:, 1] = np.clip(centers[:, 1], 0, H)
                delaunay_edges = get_delaunay_pairs(centers)
                stall_count = 0
                step_size = max(0.01, step_size * 0.5)
                continue
            
            forces = compute_forces(centers, radii, W, H, delaunay_edges, duals)
            
            # Normalize forces to prevent huge steps
            force_norms = np.linalg.norm(forces, axis=1)
            max_norm = np.max(force_norms)
            if max_norm > 1e-10:
                # Scale forces so that max step is step_size
                forces = (forces / max_norm) * step_size
                centers += forces
                
                # Clamp to box [0, W] x [0, H]
                centers[:, 0] = np.clip(centers[:, 0], 0, W)
                centers[:, 1] = np.clip(centers[:, 1], 0, H)
                
                # Recompute Delaunay every 3 iterations to keep structure fresh
                if it % 3 == 0:
                    delaunay_edges = get_delaunay_pairs(centers)
            
            # Decay step size
            step_size *= 0.995
            
            if val - best_val < 1e-9 and it > 100:
                break

        if best_radii is not None:
            return best_val, best_centers, best_radii
        return -np.inf, None, None

    # Aspect ratios to try
    # Perimeter 2(W+H) = 4 => W+H=2
    # Focus on ratios close to 1 and sqrt(2)
    aspects = [1.0, 1.1, 0.9, 1.2, 0.8, 1.414, 0.707, 1.3, 0.77, 1.5, 0.67, 1.6, 0.625, 1.8, 0.55, 2.0, 0.5, 2.5, 0.4]
    
    best_val_global = -np.inf
    best_result_global = None
    
    # Generate initial configurations
    configs = []
    
    # Grids
    for gx, gy in [(5, 5), (4, 6), (7, 3), (6, 4), (3, 7), (2, 11)]:
        cx = np.linspace(0.5/gx, 1 - 0.5/gx, gx)
        cy = np.linspace(0.5/gy, 1 - 0.5/gy, gy)
        grid_pts = np.array([[x, y] for y in cy for x in cx])
        if len(grid_pts) >= 21:
            configs.append(grid_pts[:21])
    
    # Hexagonal-like
    hex_pts = []
    for row in range(6):
        for col in range(4):
            x = (col + 0.5) / 4.0 + (0.5 / 4.0) * (row % 2)
            y = (row + 0.5) / 6.0
            hex_pts.append([x, y])
    if len(hex_pts) >= 21:
        configs.append(np.array(hex_pts[:21]))
        
    # Random seeds
    for seed in [42, 123, 456, 789, 1024]:
        np.random.seed(seed)
        configs.append(np.random.rand(21, 2))
        
    # Optimize for each aspect and config
    for aspect in aspects:
        H = 2.0 / (aspect + 1.0)
        W = 2.0 * aspect / (aspect + 1.0)
        
        for config in configs:
            centers_init = config * np.array([W, H])
            # Ensure centers are within bounds
            centers_init[:, 0] = np.clip(centers_init[:, 0], 0, W)
            centers_init[:, 1] = np.clip(centers_init[:, 1], 0, H)
            
            val, centers, radii = optimize_packing(centers_init, W, H)
            
            if val > best_val_global:
                best_val_global = val
                result = np.zeros((21, 3))
                result[:, :2] = centers
                result[:, 2] = radii
                best_result_global = result

    if best_result_global is not None:
        return best_result_global
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
