"""Model-and-optimize approach for the Heilbronn-11 triangle problem."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 142  # safety margin from 150s limit

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _project_to_triangle(pts):
    """Vectorized projection of points into the unit equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (u, v, w) with u+v+w=1 to Cartesian."""
    # bary has shape (n, 3) or we work with (n, 2) where w = 1-u-v
    u = bary[:, 0]
    v = bary[:, 1]
    x = u * 1.0 + v * 0.5
    y = v * S3H
    return np.column_stack([x, y])


def _cartesian_to_barycentric(pts):
    """Convert Cartesian to barycentric (u, v) where w = 1-u-v >= 0."""
    x = pts[:, 0]
    y = pts[:, 1]
    v = y / S3H
    u = x - 0.5 * v
    return np.column_stack([u, v])


def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Better initial guess: symmetric distribution
    # 3 corners, 3 edge midpoints, 1 center, 3 more spread
    init = np.array([
        [0.0, 0.0],          # corner 0
        [1.0, 0.0],          # corner 1
        [0.5, S3H],          # corner 2
        [0.5, 0.0],          # mid bottom
        [0.75, S3H / 2],     # mid right
        [0.25, S3H / 2],     # mid left
        [0.5, S3H/3],        # center-ish
        [0.333, S3H/6],      # lower left
        [0.667, S3H/6],      # lower right
        [0.5, 2*S3H/3],      # upper
        [0.2, 0.2],
    ], dtype=float)
    _update_best(init)

    def _de_objective(x_flat):
        """Objective for Differential Evolution in barycentric coords."""
        u = x_flat[0:N]
        v = x_flat[N:2*N]
        # Enforce barycentric constraints: u>=0, v>=0, u+v<=1
        # DE works in a box. We map the box [0,1]x[0,1] to the triangle.
        # Actually, simpler: DE in Cartesian with projection is easier to set up bounds for.
        # Let's stick to Cartesian with projection for DE, as bounds are [0,1]x[0,S3H] roughly.
        # But projection is non-differentiable. DE doesn't care.
        pts = x_flat.reshape(-1, 2)
        pts = _project_to_triangle(pts)
        return -_min_triangle_area(pts)

    # Phase A: Multi-start Local Optimization
    # Instead of a single slow global search, we perform many fast local optimizations
    # from diverse random starts. This leverages the speed of L-BFGS-B to explore
    # multiple basins of attraction.
    
    def _objective_l_bfgs(x_flat):
        """
        Objective for L-BFGS-B.
        Input x_flat is 22 floats: [u0..u10, v0..v10] (barycentric coords).
        We map to Cartesian, ensuring points are inside the triangle.
        Returns negative soft-min area.
        """
        u = x_flat[:N]
        v = x_flat[N:2*N]
        
        # Barycentric coordinates must satisfy u >= 0, v >= 0, u+v <= 1.
        # L-BFGS-B works on a box, so we don't have direct constraint u+v<=1.
        # We can use a transformation or just clip. Clipping is non-differentiable at boundaries.
        # A better approach: parameterize using unbounded variables and map to triangle?
        # Or just use Cartesian with projection? Projection is also non-smooth.
        
        # Let's stick to Cartesian for L-BFGS-B as well, but use a smooth penalty for boundaries?
        # Actually, the previous code used barycentric with clipping. Clipping creates flat regions.
        # Let's try a different mapping: 
        # x = u + 0.5v, y = S3H * v.
        # If we optimize in u,v space with bounds u in [0,1], v in [0,1], we need u+v <= 1.
        
        # Alternative: Use the "simplex" parameterization where we optimize in 2N dimensions
        # but enforce the triangle constraint via a soft penalty in the objective.
        
        # Let's use the soft penalty approach on Cartesian coordinates directly.
        # It's simpler and robust.
        pts = x_flat.reshape(-1, 2)
        
        # Soft penalty for points outside the triangle
        # Distance to edges?
        # Edge 1: y = 0. dist = y.
        # Edge 2: x = 1. dist = 1 - x.
        # Edge 3: y = -sqrt(3)(x - 1) => sqrt(3)x + y - sqrt(3) = 0.
        # Normalized distance: (sqrt(3)x + y - sqrt(3)) / 2 ? No, normal vector length is 2.
        # d3 = (S3*x + y - S3) / 2.
        
        y = pts[:, 1]
        x = pts[:, 0]
        
        # Clip to hard bounds first to prevent huge penalties?
        # Actually, let's just compute the soft min area and add a penalty if out of bounds.
        # But we want to maximize area, so we minimize -area.
        # If out of bounds, we want to push back.
        
        # Let's use the barycentric approach but with a smooth mapping.
        # Map (u,v) in [0,1]x[0,1] to triangle?
        # Standard trick: 
        # Let s = u / (u+v+1) ? No.
        
        # Let's stick to the previous successful pattern from inspirations:
        # They likely used a robust global search or a very good initial guess.
        # Given the "Stagnation" warning, we need diversity.
        
        # Let's implement a simple random restart in Cartesian space with projection,
        # and use Nelder-Mead or L-BFGS-B on the projected points?
        # Projection breaks gradients.
        
        # Better: Use `scipy.optimize.minimize` with `method='SLSQP'`?
        # SLSQP can handle inequality constraints.
        # Constraints: 
        # y >= 0
        # 1-x >= 0
        # S3*x + y - S3 >= 0 (inside the triangle defined by (0,0), (1,0), (0.5, S3H))
        # Wait, the triangle is (0,0), (1,0), (0.5, S3H).
        # Line 1: y=0. Constraint: y >= 0.
        # Line 2: x=1. Constraint: 1-x >= 0.
        # Line 3: Passes through (1,0) and (0.5, S3H).
        # Slope = (S3H - 0) / (0.5 - 1) = S3H / -0.5 = -S3.
        # Eq: y - 0 = -S3 (x - 1) => y = -S3 x + S3 => S3 x + y - S3 = 0.
        # Inside: 0.5, S3H/2 -> S3(0.5) + S3H/2 - S3 = S3/2 + S3/4 - S3 = -S3/4 < 0.
        # So constraint: S3 x + y - S3 <= 0.
        
        # SLSQP is slower than L-BFGS-B but handles constraints.
        # Let's try SLSQP for the local polish.
        
        pass # Placeholder, we will define the objective inside the loop or use a closure

    # Phase B: Local Polish using L-BFGS-B on the best found so far
    if best_pts is not None and _elapsed_ok(20):
        # Try a few random perturbations around the best point and polish
        for i in range(5):
            if not _elapsed_ok(10):
                break
            # Perturb best point
            cand = best_pts + rng.normal(0, 0.05, size=(N, 2))
            cand = _project_to_triangle(cand)
            
            # Optimize in barycentric coords for better constraint handling
            bary0 = _cartesian_to_barycentric(cand)
            x0 = np.concatenate([bary0[:, 0], bary0[:, 1]])
            
            def obj_l_bfgs(x):
                u = x[0:N]
                v = x[N:2*N]
                u = np.clip(u, 0, 1)
                v = np.clip(v, 0, 1 - u)
                pts = _barycentric_to_cartesian(np.column_stack([u, v]))
                return -_soft_min_area(pts, 500.0)
            
            try:
                res = minimize(obj_l_bfgs, x0, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                u = np.clip(res.x[0:N], 0, 1)
                v = np.clip(res.x[N:2*N], 0, 1 - u)
                pts = _barycentric_to_cartesian(np.column_stack([u, v]))
                _update_best(pts)
            except Exception:
                pass

    # Phase C: Final Nelder-Mead polish on the absolute best
    if best_pts is not None and _elapsed_ok(10):
        def nm_obj(x):
            p = x.reshape(-1, 2)
            p = _project_to_triangle(p)
            return -_soft_min_area(p, 1000.0)
        
        try:
            res = minimize(nm_obj, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            p = _project_to_triangle(res.x.reshape(-1, 2))
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        pts = _barycentric_to_cartesian(np.column_stack([u, v]))
        return pts

    return _project_to_triangle(best_pts)
# EVOLVE-BLOCK-END
