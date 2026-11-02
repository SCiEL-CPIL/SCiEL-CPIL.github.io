"""Heilbronn triangles: maximize smallest triangle area for 11 points in unit equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

def bary_to_cart(bary):
    """Convert barycentric coordinates (u, v) to Cartesian.
    Point = u*V0 + v*V1 + w*V2 where w = 1-u-v.
    So x = v + 0.5*w = v + 0.5*(1-u-v) = 0.5 + v - 0.5*u
       y = w*H = (1-u-v)*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = H * w
    return np.column_stack([x, y])

def cart_to_bary(pts):
    """Convert Cartesian points to barycentric coordinates."""
    x = pts[:, 0]
    y = pts[:, 1]
    w = y / H
    u = 1.0 - w - x + 0.5 * w  # from x = v + 0.5*w, v = x - 0.5*w, u = 1 - v - w
    v = x - 0.5 * w
    return np.column_stack([u, v])

def min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the points.
    pts is (N, 2) array of Cartesian coordinates.
    """
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |det|
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                det = dx1 * dy2 - dy1 * dx2
                area = 0.5 * abs(det)
                if area < min_area:
                    min_area = area
    return min_area

def min_triangle_area_squared(pts):
    """Return squared minimum area (smoother for optimization)."""
    n = len(pts)
    min_sq = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                det = dx1 * dy2 - dy1 * dx2
                sq = 0.25 * det * det
                if sq < min_sq:
                    min_sq = sq
    return min_sq

def objective_neg_min_area_sq(bary_params):
    """Objective: negative of squared minimum triangle area, for minimization."""
    pts = bary_to_cart(bary_params)
    return -min_triangle_area_squared(pts)

def objective_neg_min_area(bary_params):
    """Objective: negative of minimum triangle area."""
    pts = bary_to_cart(bary_params)
    return -min_triangle_area(pts)

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Bounds for barycentric coordinates: each >= 0, and u+v <= 1
    # We use u in [0, 0.99], v in [0, 0.99] and enforce u+v < 1 via projection or just tight bounds
    # Better: use u in [eps, 1-2*eps], v in [eps, 1-2*eps-u]... complex.
    # Simpler: use bounds [0, 0.98] for both, and project if needed.
    # Actually for L-BFGS-B, let's use bounds [0.001, 0.999] for each, and accept that u+v might exceed 1 slightly.
    # Better approach: parameterize with u, v where u>=0, v>=0, u+v<=1.
    # Use transformation: let a = u, b = v, and constrain a+b <= 1.
    # For L-BFGS-B, we can just use bounds and hope, or use a penalty.
    # Let's use bounds [0, 0.99] for u and v, and if u+v > 1, we clip.
    
    # Actually, let's use a cleaner approach: work in Cartesian with bounds,
    # but the triangle constraint is non-rectangular. 
    # Best: stay in barycentric, use bounds [0, 0.99] for u, v, and in objective, 
    # if any point is outside, return a large penalty.
    
    def penalized_objective(bary_params):
        u = bary_params[:, 0]
        v = bary_params[:, 1]
        w = 1.0 - u - v
        # Check if any point is outside the triangle
        if np.any(u < -1e-6) or np.any(v < -1e-6) or np.any(w < -1e-6):
            return 1e10  # large penalty
        pts = bary_to_cart(bary_params)
        return -min_triangle_area_squared(pts)
    
    bounds = [(0.0, 0.99)] * (2 * N)
    
    # Generate multiple starting points
    rng = np.random.RandomState(42)
    
    # Initial best
    best_pts = None
    best_val = -float('inf')
    
    # Starting points: 
    # 1. Random uniform in triangle
    # 2. Grid-like arrangements
    # 3. Perturbed known configurations
    
    starts = []
    
    # Random starts
    for _ in range(30):
        bary = np.zeros((N, 2))
        for _ in range(N):
            u = rng.random()
            v = rng.random() * (1.0 - u)
            bary[_] = [u, v]
        starts.append(bary)
    
    # Grid-like: try to place points in a structured way
    # 3x3 grid plus 2 extra
    grid_starts = []
    # 3x3 grid in barycentric
    for i in range(3):
        for j in range(3):
            u = 0.25 + 0.25 * i
            v = 0.25 + 0.25 * j
            if u + v <= 0.95:
                grid_starts.append([u, v])
    # Add 2 more
    grid_starts.append([0.5, 0.15])
    grid_starts.append([0.15, 0.5])
    if len(grid_starts) >= N:
        starts.append(np.array(grid_starts[:N]))
    
    # Symmetric-ish start: vertices + midpoints + center
    sym = np.array([
        [0.01, 0.01],   # near V0
        [0.01, 0.98],   # near V1
        [0.98, 0.01],   # near V2
        [0.5, 0.01],    # midpoint V0-V1
        [0.01, 0.5],    # midpoint V0-V2
        [0.5, 0.5],     # midpoint V1-V2
        [0.33, 0.33],   # center
        [0.15, 0.15],
        [0.7, 0.15],
        [0.15, 0.7],
        [0.5, 0.25],
    ])
    starts.append(sym)
    
    # More structured starts
    # Hexagonal packing attempt
    hex_start = np.array([
        [0.5, 0.1],
        [0.2, 0.2],
        [0.8, 0.2],
        [0.35, 0.35],
        [0.65, 0.35],
        [0.1, 0.4],
        [0.9, 0.4],
        [0.5, 0.5],
        [0.25, 0.55],
        [0.75, 0.55],
        [0.5, 0.7],
    ])
    starts.append(hex_start)
    
    # Optimize each start
    for idx, x0 in enumerate(starts):
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                penalized_objective,
                x0.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
        
        x_opt = res.x.reshape(N, 2)
        pts = bary_to_cart(x_opt)
        
        # Verify points are inside
        u = x_opt[:, 0]
        v = x_opt[:, 1]
        w = 1.0 - u - v
        if np.any(u < -1e-6) or np.any(v < -1e-6) or np.any(w < -1e-6):
            continue
        
        val = min_triangle_area(pts)
        if val > best_val:
            best_val = val
            best_pts = pts.copy()
    
    # If we still have time, do a second round with more refined starts
    if time.time() < deadline - 30:
        # Use the best solution as a new start with perturbations
        if best_pts is not None:
            best_bary = cart_to_bary(best_pts)
            for _ in range(20):
                if time.time() > deadline:
                    break
                pert = best_bary + rng.normal(0, 0.05, size=(N, 2))
                # Clip to valid range
                pert[:, 0] = np.clip(pert[:, 0], 0.001, 0.99)
                pert[:, 1] = np.clip(pert[:, 1], 0.001, 0.99)
                # Ensure u + v < 1
                sum_uv = pert[:, 0] + pert[:, 1]
                mask = sum_uv > 0.99
                if np.any(mask):
                    scale = 0.99 / sum_uv[mask]
                    pert[mask, 0] *= scale
                    pert[mask, 1] *= scale
                
                try:
                    res = minimize(
                        penalized_objective,
                        pert.flatten(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                except Exception:
                    continue
                
                x_opt = res.x.reshape(N, 2)
                pts = bary_to_cart(x_opt)
                
                u = x_opt[:, 0]
                v = x_opt[:, 1]
                w = 1.0 - u - v
                if np.any(u < -1e-6) or np.any(v < -1e-6) or np.any(w < -1e-6):
                    continue
                
                val = min_triangle_area(pts)
                if val > best_val:
                    best_val = val
                    best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: random points
        pts = np.array([
            [0.5, 0.0],
            [0.25, 0.433],
            [0.75, 0.433],
            [0.5, 0.866],
            [0.1, 0.1],
            [0.9, 0.1],
            [0.5, 0.3],
            [0.3, 0.6],
            [0.7, 0.6],
            [0.5, 0.55],
            [0.5, 0.75],
        ])
        return pts
    
    return best_pts
