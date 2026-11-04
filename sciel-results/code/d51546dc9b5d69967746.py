"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi
from scipy.optimize import minimize

def _eval_packing(xy, r):
    """Evaluate a packing: returns (total_radius, is_valid)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _compute_voronoi_inradii(points):
    """
    Given n points in [0,1]^2, compute the inradius of each Voronoi cell.
    The inradius is the maximum radius of a circle centered at the point
    that fits within the Voronoi cell and the unit square.
    """
    n = len(points)
    inradii = np.zeros(n)
    
    # Add boundary points to constrain Voronoi to unit square
    # We'll handle boundaries by computing distance to cell edges and to square boundaries
    
    try:
        vor = Voronoi(points)
    except Exception:
        # Fallback: use min distance to other points and to walls
        for i in range(n):
            min_dist = np.inf
            for j in range(n):
                if i != j:
                    d = np.sqrt((points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2)
                    min_dist = min(min_dist, d)
            # Distance to walls
            wall_dist = min(points[i,0], 1-points[i,0], points[i,1], 1-points[i,1])
            inradii[i] = min(min_dist / 2.0, wall_dist)
        return inradii
    
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            # Unbounded region - circle extends to infinity in Voronoi sense
            # But we're in a bounded square, so use distance to nearest other point and walls
            min_dist = np.inf
            for j in range(n):
                if i != j:
                    d = np.sqrt((points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2)
                    min_dist = min(min_dist, d)
            wall_dist = min(points[i,0], 1-points[i,0], points[i,1], 1-points[i,1])
            inradii[i] = min(min_dist / 2.0, wall_dist)
            continue
        
        # Compute inradius of the polygon defined by region vertices
        # The inradius is the minimum distance from point i to any edge of the cell
        min_dist_to_edge = np.inf
        
        for k in range(len(region)):
            v1 = vor.vertices[region[k]]
            v2 = vor.vertices[region[(k+1) % len(region)]]
            
            # Distance from point i to line segment v1-v2
            px, py = points[i]
            x1, y1 = v1
            x2, y2 = v2
            
            # Line through v1 and v2
            dx = x2 - x1
            dy = y2 - y1
            length_sq = dx*dx + dy*dy
            
            if length_sq < 1e-12:
                dist = np.sqrt((px-x1)**2 + (py-y1)**2)
            else:
                # Projection of point onto line
                t = ((px - x1)*dx + (py - y1)*dy) / length_sq
                t = max(0, min(1, t))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
            
            min_dist_to_edge = min(min_dist_to_edge, dist)
        
        # Also consider distance to square boundaries
        wall_dist = min(points[i,0], 1-points[i,0], points[i,1], 1-points[i,1])
        
        inradii[i] = min(min_dist_to_edge, wall_dist)
    
    return inradii


def _voronoi_objective(points):
    """Objective: negative sum of Voronoi inradii (to minimize)."""
    inradii = _compute_voronoi_inradii(points)
    return -np.sum(inradii)


def _voronoi_gradient(points, eps=1e-7):
    """Numerical gradient of the Voronoi objective."""
    n = len(points)
    grad = np.zeros(2 * n)
    
    base = _voronoi_objective(points)
    
    for k in range(2 * n):
        points_plus = points.copy()
        points_plus[k % n, k // n] += eps
        obj_plus = _voronoi_objective(points_plus)
        
        points_minus = points.copy()
        points_minus[k % n, k // n] -= eps
        obj_minus = _voronoi_objective(points_minus)
        
        grad[k] = (obj_plus - obj_minus) / (2 * eps)
    
    return grad


def _voronoi_optimize(points, deadline, max_iter=200):
    """Optimize points using L-BFGS-B to maximize sum of Voronoi inradii."""
    n = len(points)
    v0 = points.flatten()
    
    bounds = []
    for i in range(n):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    
    def obj(v):
        if time.time() > deadline:
            return float('inf')
        pts = v.reshape(n, 2)
        return _voronoi_objective(pts)
    
    def grad(v):
        if time.time() > deadline:
            return np.zeros(2 * n)
        pts = v.reshape(n, 2)
        return _voronoi_gradient(pts)
    
    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8,
                                   'maxfun': 50000})
        opt_points = result.x.reshape(n, 2)
    except Exception:
        opt_points = points
    
    return opt_points


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                positions.append([x, y])
    while len(positions) < N:
        positions.append([0.5, 0.5])
    return np.array(positions[:N])


def _random_init(n, seed=0):
    """Random initialization of points in [0.05, 0.95]^2."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.05, 0.95, (n, 2))


def _local_search_refine(xy, r, deadline, grid_size=50):
    """
    Local search: iteratively move each circle to best position/radius.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)
    
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))
    
    max_sweeps = 8
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]
            
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()
            
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]
            
            if new_r > r[i] + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True
        
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_xy = xy.copy()
            best_r = r.copy()
        
        if not improved:
            break
    
    return best_xy, best_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    n = N

    # Phase 1: Voronoi-based optimization from multiple initializations
    inits = []
    
    # Grid initializations
    for side in [4, 5, 6]:
        pts = _grid_init(side)
        if len(pts) >= n:
            inits.append(pts[:n])
        else:
            # Pad with random points
            extra = _random_init(n - len(pts), seed=side)
            inits.append(np.vstack([pts, extra]))
    
    # Random initializations
    for seed in [0, 1, 2, 3, 4]:
        inits.append(_random_init(n, seed=seed))
    
    # Run Voronoi optimization on each initialization
    for init_idx, pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        # Voronoi optimization
        opt_pts = _voronoi_optimize(pts.copy(), deadline - 5, max_iter=100)
        
        # Compute inradii from optimized points
        inradii = _compute_voronoi_inradii(opt_pts)
        
        # Check validity
        total, valid = _eval_packing(opt_pts, inradii)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = opt_pts.copy()
            best_r = inradii.copy()
    
    # Phase 2: Local search refinement on best result
    if best_valid and best_xy is not None and time.time() < deadline - 15:
        ls_xy, ls_r, ls_total, ls_valid = _local_search_refine(
            best_xy.copy(), best_r.copy(), deadline, grid_size=60)
        if ls_valid and ls_total > best_total + 1e-10:
            best_total = ls_total
            best_xy = ls_xy
            best_r = ls_r

    # Phase 3: Another round of Voronoi optimization from the refined result
    if best_valid and best_xy is not None and time.time() < deadline - 10:
        opt_pts2 = _voronoi_optimize(best_xy.copy(), deadline - 5, max_iter=80)
        inradii2 = _compute_voronoi_inradii(opt_pts2)
        total2, valid2 = _eval_packing(opt_pts2, inradii2)
        if valid2 and total2 > best_total:
            best_total = total2
            best_xy = opt_pts2.copy()
            best_r = inradii2.copy()
        
        # Final local search
        if time.time() < deadline - 5:
            ls2_xy, ls2_r, ls2_total, ls2_valid = _local_search_refine(
                best_xy.copy(), best_r.copy(), deadline, grid_size=80)
            if ls2_valid and ls2_total > best_total + 1e-10:
                best_total = ls2_total
                best_xy = ls2_xy
                best_r = ls2_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        pts = _grid_init(side)
        if len(pts) < N:
            extra = _random_init(N - len(pts), seed=99)
            pts = np.vstack([pts, extra])
        pts = pts[:N]
        inradii = _compute_voronoi_inradii(pts)
        total, valid = _eval_packing(pts, inradii)
        if valid:
            best_total = total
            best_valid = True
            best_xy = pts
            best_r = inradii
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
