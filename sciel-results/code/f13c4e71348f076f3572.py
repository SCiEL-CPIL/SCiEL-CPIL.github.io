import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _hex_lattice_init(spacing, offset_x=0.0, offset_y=0.0):
    """Generate points on a hexagonal lattice inside the unit circle."""
    points = []
    # Estimate range
    k = int(1.0 / spacing) + 3
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing + offset_x
            y = j * spacing * np.sqrt(3.0) / 2.0 + offset_y
            if x*x + y*y <= 1.0:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        return None
    # Select the N points closest to the origin to form a compact cluster
    radii = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(radii)[:N]
    return points[indices]

def _calc_radius(points):
    """Calculate the exact packing radius for a given set of points."""
    # Boundary constraint
    boundary_r = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(boundary_r)
    
    # Pairwise constraint
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _softmin_objective(params, beta):
    """
    Objective function for optimization.
    Minimizes -softmin(constraints).
    """
    points = params.reshape(-1, 2)
    
    # 1. Boundary constraints: 1 - ||p_i||
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary_vals = 1.0 - norms
    
    # 2. Pairwise constraints: ||p_i - p_j|| / 2
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    mask = ~np.eye(N, dtype=bool)
    pair_vals = dists[mask] / 2.0
    
    # Combine all constraints
    all_vals = np.concatenate([boundary_vals, pair_vals])
    
    # Soft min calculation
    min_val = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - min_val))
    sum_exp = np.sum(exp_vals)
    
    soft_min = min_val - np.log(sum_exp) / beta
    
    return -soft_min

def _softmin_gradient(params, beta):
    """
    Gradient of the softmin objective.
    """
    points = params.reshape(-1, 2)
    
    # 1. Boundary constraints
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary_vals = 1.0 - norms
    
    # 2. Pairwise constraints
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    mask = ~np.eye(N, dtype=bool)
    pair_vals = dists[mask] / 2.0
    
    all_vals = np.concatenate([boundary_vals, pair_vals])
    
    min_val = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - min_val))
    sum_exp = np.sum(exp_vals)
    
    # Weights for each constraint
    weights = exp_vals / sum_exp
    
    # Gradient accumulation
    grad = np.zeros_like(points)
    
    # Boundary part
    w_boundary = weights[:N]
    
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    boundary_grad = -points / safe_norms[:, np.newaxis]
    
    grad += -w_boundary[:, np.newaxis] * boundary_grad
    
    # Pairwise part
    w_pair_matrix = np.zeros((N, N))
    idx_i, idx_j = np.where(mask)
    w_pair_vals = weights[N:]
    
    w_pair_matrix[idx_i, idx_j] = w_pair_vals
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_matrix = np.linalg.norm(diff, axis=2)
    
    safe_dists = np.where(dists_matrix > 1e-10, dists_matrix, 1.0)
    
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    weighted_vecs = w_pair_matrix[:, :, np.newaxis] * unit_vecs
    pair_grad = -np.sum(weighted_vecs, axis=1)
    
    grad += pair_grad
    
    return grad.flatten()

def _physics_relax(points, iterations=1000, dt=0.01, cutoff=0.5):
    """
    Grid-based physics relaxation.
    Uses a spatial hash grid to find neighbors within 'cutoff' distance.
    Applies repulsive forces (inverse square) to push points apart.
    Applies boundary force to push points inside.
    """
    pts = points.copy()
    
    # Spatial hash grid
    # Cell size = cutoff
    cell_size = cutoff
    grid = {}
    
    def get_cell(x, y):
        return (int(np.floor(x / cell_size)), int(np.floor(y / cell_size)))
    
    def rebuild_grid(pts):
        grid.clear()
        for i, (x, y) in enumerate(pts):
            c = get_cell(x, y)
            if c not in grid:
                grid[c] = []
            grid[c].append(i)
    
    rebuild_grid(pts)
    
    for _ in range(iterations):
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Rebuild grid every step to ensure correctness (costly but N=50 is small)
        # Actually, for N=50, brute force O(N^2) is ~2500 ops. Grid is overhead.
        # But the prompt asks for grid-based. Let's use grid for neighbor lookup.
        # For N=50, even grid might be slower than vectorized numpy.
        # However, the "physics" part is the key: iterative relaxation.
        
        # Let's stick to a simple vectorized force calculation for speed, 
        # but conceptually it's the same as the grid approach for dense packing.
        # The "grid" idea is to avoid O(N^2) if N was large. Here N=50.
        # Let's implement the grid logic properly to satisfy the directive.
        
        rebuild_grid(pts)
        
        for i in range(N):
            x, y = pts[i]
            c = get_cell(x, y)
            
            # Check 3x3 neighborhood
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nc = (c[0] + dx, c[1] + dy)
                    if nc in grid:
                        for j in grid[nc]:
                            if i == j:
                                continue
                            dx_p = pts[i][0] - pts[j][0]
                            dy_p = pts[i][1] - pts[j][1]
                            dist_sq = dx_p*dx_p + dy_p*dy_p
                            dist = np.sqrt(dist_sq)
                            
                            if dist < cutoff and dist > 1e-6:
                                # Repulsive force: k / dist^2 * direction
                                # Force magnitude decreases with distance
                                # We want to push them apart until they touch (dist ~ 2r)
                                # Simple spring: F = k * (rest_length - dist) if dist < rest_length
                                # Or inverse square: F = k / dist^2
                                # Let's use a soft repulsion: F = 1 / (dist^2 + eps)
                                f_mag = 1.0 / (dist_sq + 1e-4)
                                forces[i][0] += f_mag * dx_p / dist
                                forces[i][1] += f_mag * dy_p / dist
                            
        # Boundary force
        # Push points towards center if they are close to boundary
        # F_boundary = -k * (norm - (1-r)) * (pos/norm)
        # We don't know r exactly, so use a soft boundary at 1.0
        norms = np.hypot(pts[:, 0], pts[:, 1])
        # If norm > 0.8, apply inward force
        mask_boundary = norms > 0.8
        for i in np.where(mask_boundary)[0]:
            f_mag = (norms[i] - 0.8) * 10.0 # Strong pull
            forces[i][0] -= f_mag * pts[i][0] / norms[i]
            forces[i][1] -= f_mag * pts[i][1] / norms[i]
        
        # Update positions
        pts += dt * forces
        
        # Enforce hard boundary (clamp)
        norms = np.hypot(pts[:, 0], pts[:, 1])
        mask_out = norms > 1.0
        if np.any(mask_out):
            pts[mask_out] = pts[mask_out] * (1.0 / norms[mask_out, np.newaxis])
        
        # Optional: Re-center to prevent drift
        # pts -= np.mean(pts, axis=0)
        
    return pts

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    best_radius = 0.0
    best_points = None
    
    # Initialization
    init_strategies = []
    
    spacings = [0.25, 0.26, 0.24, 0.27]
    offsets = [(0,0), (0.01, 0.01), (-0.01, 0.01), (0.01, -0.01)]
    
    for s in spacings:
        for ox, oy in offsets:
            pts = _hex_lattice_init(s, ox, oy)
            if pts is not None and len(pts) == N:
                init_strategies.append(pts)
    
    if not init_strategies:
        pts = np.random.rand(N, 2) * 2 - 1
        pts = pts[np.hypot(pts[:,0], pts[:,1]) <= 1]
        if len(pts) < N:
             while len(pts) < N:
                 new_pts = np.random.rand(N, 2) * 2 - 1
                 pts = np.vstack([pts, new_pts])
        init_strategies.append(pts[:N])

    # Optimization loop
    for init_pts in init_strategies:
        if time.time() - start_time > time_limit:
            break
            
        current_pts = init_pts.copy()
        
        # Stage 1: Physics-based relaxation to quickly improve local packing
        # This is fast and helps escape bad local minima from the lattice init
        current_pts = _physics_relax(current_pts, iterations=500, dt=0.005, cutoff=0.4)
        
        # Stage 2: Multi-stage soft-min optimization
        current_params = current_pts.flatten()
        
        # Annealing schedule for beta
        betas = [10, 50, 200, 1000, 5000]
        
        for beta in betas:
            if time.time() - start_time > time_limit:
                break
                
            try:
                res = minimize(
                    _softmin_objective,
                    current_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=_softmin_gradient,
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.success or res.status in [0, 1, 2, 3, 4, 5, 6]:
                    new_pts = res.x.reshape(-1, 2)
                    
                    if np.all(np.hypot(new_pts[:, 0], new_pts[:, 1]) < 1.0 + 1e-6):
                         current_params = res.x
                         current_pts = new_pts
            except Exception:
                pass
        
        # Stage 3: Final polish with high beta
        if time.time() - start_time < time_limit - 5.0:
            try:
                res = minimize(
                    _softmin_objective,
                    current_params,
                    args=(10000,),
                    method='L-BFGS-B',
                    jac=_softmin_gradient,
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.success or res.status in [0, 1, 2, 3, 4, 5, 6]:
                    new_pts = res.x.reshape(-1, 2)
                    if np.all(np.hypot(new_pts[:, 0], new_pts[:, 1]) < 1.0 + 1e-6):
                        current_pts = new_pts
            except Exception:
                pass

        # Calculate final radius for this strategy
        final_radius = _calc_radius(current_pts)
        
        if final_radius > best_radius:
            best_radius = final_radius
            best_points = current_pts.copy()
            
    # Final check
    if best_points is None:
        pts = _hex_lattice_init(0.25)
        if pts is not None:
            best_points = pts
            best_radius = _calc_radius(pts)
        else:
            best_points = np.zeros((N, 2))
            best_radius = 0.0

    return best_points
# EVOLVE-BLOCK-END
