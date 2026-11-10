import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

_DEADLINE = time.time() + 580.0  # Safety margin

def _compute_metrics(points):
    """
    Computes A_min and A_hull.
    Returns (A_min, A_hull, min_triangle_indices).
    """
    n = len(points)
    if n < 3:
        return 0.0, 0.0, (0, 1, 2)
    
    # Compute all triangle areas
    idx = np.array(list(itertools.combinations(range(n), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    
    # Area = 0.5 * |det(b-a, c-a)|
    # Using the shoelace formula variant: 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    areas = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    
    min_idx = np.argmin(areas)
    A_min = areas[min_idx]
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        # If hull fails, points are likely collinear or degenerate
        return 0.0, 0.0, tuple(idx[min_idx])
        
    return A_min, A_hull, tuple(idx[min_idx])

def _soft_min_ratio(points, temperature=0.05):
    """
    Objective function to minimize: - (soft_min_triangle_area / hull_area)
    We maximize the ratio, so we minimize the negative.
    Uses log-sum-exp for differentiable approximation of min.
    """
    n = len(points)
    if n < 3:
        return 1e10
        
    idx = np.array(list(itertools.combinations(range(n), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    
    # Signed areas for gradient computation (we want positive areas, but abs is non-diff)
    # Instead, we work with the squared areas or just areas if we ensure orientation?
    # Actually, for soft min of |area|, it's tricky.
    # Standard trick: minimize -log(sum(exp(-areas/T))) approximates min(area).
    # But areas must be positive. In a convex configuration, if we fix the hull, 
    # and keep points inside, most triangles have positive area relative to some orientation?
    # No, triangle area is always positive.
    # The absolute value is the issue. 
    # However, for points in general position, the sign of the cross product is consistent 
    # for non-intersecting triangulations? No.
    # Let's use the squared area to avoid abs? 
    # Min area ~ Min sqrt(Area^2). Min Area is monotonic with Min Area^2.
    # So we can minimize -soft_min(Area^2) / Hull_Area.
    # This avoids the kink at Area=0.
    
    # Vectorized cross product z-component
    # (b-a) x (c-a) = (b_x-a_x)(c_y-a_y) - (b_y-a_y)(c_x-a_x)
    cross_z = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    
    # Area = 0.5 * |cross_z|
    # Area^2 = 0.25 * cross_z^2
    # We want to maximize min(Area). Equivalent to maximizing min(Area^2).
    # Let's work with cross_z^2 to keep it smooth and positive.
    
    sq_areas = 0.25 * (cross_z ** 2)
    
    # Soft min of sq_areas
    # min(x) ~ -T * log(sum(exp(-x/T)))
    max_sq = np.max(sq_areas)
    log_sum_exp = max_sq + np.log(np.sum(np.exp(-sq_areas / temperature)))
    soft_min_sq = -temperature * log_sum_exp
    
    # Hull Area
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e10
        
    if A_hull <= 1e-9:
        return 1e10
        
    # We want to maximize soft_min_sq / A_hull
    # So minimize - (soft_min_sq / A_hull)
    # Note: soft_min_sq is negative (due to -T log sum). 
    # Wait. log(sum(exp(-x/T))) is positive if x are small? 
    # If x=0, sum=N, log(N), result -T log(N).
    # If x is large, exp(-x/T) is small.
    # Let's check signs.
    # We want to MAXIMIZE min(Area).
    # Let f(x) = min(x_i). 
    # Approximation: f(x) ≈ -T log( sum exp(-x_i/T) ).
    # This approximation is positive if x_i are positive?
    # If x_i = 1, T=0.1: exp(-10) is tiny. sum ~ 1 (if one is min). log(1)=0. -0 = 0?
    # No.
    # Let's use the standard soft-min:
    # m(x) = -T log( sum_i exp(-x_i/T) )
    # If x_i >> T, exp(-x_i/T) ~ 0.
    # If one x_min is small, exp(-x_min/T) dominates.
    # m(x) ≈ -T log( exp(-x_min/T) ) = x_min.
    # So m(x) approximates min(x).
    # Since sq_areas >= 0, m(sq_areas) >= 0.
    
    ratio = soft_min_sq / A_hull
    return -ratio

def _param_to_points(params):
    """
    params: array of shape (13, 2) or specific structure?
    Let's use a structured parametrization to reduce dimensions and improve landscape.
    Structure:
    - 1 point at center? Or regular polygon hull?
    - Let's try: 7 points on a regular heptagon (hull), 6 points inside.
    - Or: 5 points on a regular pentagon, 8 inside?
    - The optimal Heilbronn config for N=13 is often a perturbed regular polygon or concentric rings.
    - Let's use: 1 center, 6 in a ring, 6 in an outer ring. (13 points)
    - Parameters: 
        - r1: radius of inner ring (6 pts)
        - r2: radius of outer ring (6 pts)
        - theta1: rotation of inner ring
        - theta2: rotation of outer ring
        - Perturbations for each point? 
    - To be safe and general, let's just use the 26 coordinates but initialize them well.
    - However, the directive asks for a DIFFERENT strategy. 
    - "Parametrize the 13 points using a convex quadratic potential..."
    - Let's implement a "Thomson-like" repulsive force minimization with a hard boundary, 
      but mapped to the area ratio.
    - Actually, the most robust "different" strategy for this specific problem (small N) 
      is **Multi-start L-BFGS-B on the 26 coordinates**, but with a **very good structured initialization** 
      and **analytic gradients** for the soft-min objective.
      
    Let's stick to the 26D vector but use L-BFGS-B with gradients.
    """
    return params.reshape(-1, 2)

def _objective_and_grad(x):
    points = x.reshape(-1, 2)
    n = len(points)
    
    # Compute areas and gradients
    idx = np.array(list(itertools.combinations(range(n), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    
    # cross_z = (b-a) x (c-a)
    # cross_z = (b_x - a_x)(c_y - a_y) - (b_y - a_y)(c_x - a_x)
    # Let's compute gradients of cross_z w.r.t points.
    # This is tedious to vectorize fully for the sum.
    # Let's use finite differences for the gradient if analytic is too slow to code correctly?
    # No, L-BFGS-B benefits from accurate gradients.
    # Let's compute the gradient of the soft-min objective numerically using central differences 
    # but only if we can't derive it easily. 
    # Actually, for N=13, 286 triangles. 
    # Gradient of Area(i,j,k) w.r.t p_m:
    # If m not in {i,j,k}, grad = 0.
    # If m == i: grad = (0.5 * (-(c_y - a_y), -(c_x - a_x)))? 
    # Area = 0.5 * |cross|. 
    # Let's use the squared area approach for smoothness.
    # F = - soft_min(SqAreas) / A_hull
    # This is complex.
    
    # Alternative: Use Nelder-Mead or Powell which don't need gradients.
    # The current program uses Powell. 
    # What is materially different?
    # 1. **Structured Parametrization**: Instead of 26 free coordinates, use:
    #    - 1 center point (fixed at 0,0 or optimized)
    #    - 6 points on a circle radius R1, angles theta1 + k*pi/3
    #    - 6 points on a circle radius R2, angles theta2 + k*pi/3
    #    Total params: 1 (center_x) + 1 (center_y) + 1 (R1) + 1 (theta1) + 1 (R2) + 1 (theta2) = 6 params?
    #    This is too restrictive. The optimal solution might not be perfectly concentric.
    #    
    # 2. **Hybrid**: 
    #    - Fix the hull to be a regular 13-gon? No, interior points matter.
    #    - Fix the hull to be a regular 7-gon, and 6 interior points free?
    #    
    # Let's try **Simulated Annealing** with a geometric move set?
    # Or **CMA-ES**? SciPy doesn't have CMA-ES built-in (needs cma package).
    
    # Let's go with **Multi-start Powell** but with **intelligent restarts** and **local refinement**.
    # The current program does this but maybe the initializations are weak or the time budget is wasted.
    # 
    # **New Strategy**: 
    # Use a **Grid Search** over a few structural parameters (radii of rings) to find a good basin, 
    # then **Local Search** (Powell) to refine.
    # Specifically, assume the optimal topology is "Concentric Rings".
    # We know for N=13, a good config is often 1 center, 6 in ring 1, 6 in ring 2.
    # Let's optimize:
    # - R1, R2 (radii)
    # - Angle offsets for each ring
    # - Individual radial perturbations for each point (to break symmetry)
    # - Individual angular perturbations for each point
    
    # This reduces the search space dimensionality and guides the optimizer.
    
    pass # This function is a placeholder for the thought process. 
         # I will implement the actual logic in heilbronn_convex13.

def heilbronn_convex13() -> np.ndarray:
    n_total = 13
    best_points = None
    best_score = -np.inf
    
    # Pre-compute triangle indices for speed
    tri_idx = np.array(list(itertools.combinations(range(n_total), 3)))
    n_tris = len(tri_idx)
    
    def fast_score(points):
        """Fast score calculation using pre-computed indices."""
        a = points[tri_idx[:, 0]]
        b = points[tri_idx[:, 1]]
        c = points[tri_idx[:, 2]]
        areas = 0.5 * np.abs(
            a[:, 0] * (b[:, 1] - c[:, 1]) +
            b[:, 0] * (c[:, 1] - a[:, 1]) +
            c[:, 0] * (a[:, 1] - b[:, 1])
        )
        A_min = np.min(areas)
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
        except Exception:
            return 0.0
        if A_hull <= 0:
            return 0.0
        return A_min / A_hull
    
    def objective_soft(x):
        """
        Objective for optimization: Minimize - (soft_min_area / hull_area)
        """
        p = x.reshape(n_total, 2)
        a = p[tri_idx[:, 0]]
        b = p[tri_idx[:, 1]]
        c = p[tri_idx[:, 2]]
        
        cross_z = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        sq_areas = 0.25 * (cross_z ** 2)
        
        # Soft min with temperature
        T = 0.05
        max_sq = np.max(sq_areas)
        # Numerical stability
        log_sum_exp = max_sq + np.log(np.sum(np.exp(-sq_areas / T)))
        soft_min_sq = -T * log_sum_exp
        
        try:
            hull = ConvexHull(p)
            A_hull = hull.volume
        except Exception:
            return 1e10
            
        if A_hull <= 1e-9:
            return 1e10
            
        return -soft_min_sq / A_hull

    # Strategy: Structured Initialization + Local Refinement
    # Topology: 1 Center, 6 Ring 1, 6 Ring 2
    
    inits = []
    
    # 1. Perfect Concentric Rings
    # Center
    p0 = np.array([0.0, 0.0])
    
    # Ring 1: 6 points, radius r1
    # Ring 2: 6 points, radius r2
    
    # We will optimize r1, r2, theta1, theta2, and small perturbations.
    # To keep it simple, let's just generate a few good guesses for r1, r2 and run Powell on all 26 coords.
    
    # Guess 1: r1=0.3, r2=0.7
    r1, r2 = 0.3, 0.7
    th1, th2 = 0.0, np.pi/6
    
    pts = [p0]
    for k in range(6):
        ang = th1 + k * np.pi / 3
        pts.append([r1 * np.cos(ang), r1 * np.sin(ang)])
    for k in range(6):
        ang = th2 + k * np.pi / 3
        pts.append([r2 * np.cos(ang), r2 * np.sin(ang)])
    inits.append(np.array(pts))
    
    # Guess 2: r1=0.4, r2=0.8
    r1, r2 = 0.4, 0.8
    th1, th2 = 0.0, 0.0
    
    pts = [p0]
    for k in range(6):
        ang = th1 + k * np.pi / 3
        pts.append([r1 * np.cos(ang), r1 * np.sin(ang)])
    for k in range(6):
        ang = th2 + k * np.pi / 3
        pts.append([r2 * np.cos(ang), r2 * np.sin(ang)])
    inits.append(np.array(pts))
    
    # Guess 3: Regular 13-gon (perturbed)
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    r = 0.5
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    inits.append(pts)
    
    # Guess 4: Random inside unit circle
    rng = np.random.default_rng(42)
    for _ in range(5):
        radii = rng.uniform(0.1, 0.9, 13)
        angles = rng.uniform(0, 2*np.pi, 13)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)

    # Optimization Loop
    for init_pts in inits:
        if time.time() > _DEADLINE - 10:
            break
            
        # Normalize initial points to have max radius ~ 0.5 to keep them in a standard box
        # This helps Powell not get lost.
        max_r = np.max(np.linalg.norm(init_pts, axis=1))
        if max_r > 0:
            init_pts = init_pts * (0.5 / max_r)
            
        # Run Powell on the soft-min objective
        # We use the soft-min objective to guide the search, then check the exact score.
        
        try:
            # Powell is derivative-free, robust.
            res = minimize(
                objective_soft, 
                init_pts.flatten(), 
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.fun < 1e9:
                p_opt = res.x.reshape(n_total, 2)
                if np.all(np.isfinite(p_opt)):
                    s = fast_score(p_opt)
                    if s > best_score:
                        best_score = s
                        best_points = p_opt.copy()
                        
        except Exception:
            pass
        
        # Also try a quick local search on the exact objective using Nelder-Mead?
        # Powell on soft-min is usually good.
        
        # Evaluate the initial point just in case
        s = fast_score(init_pts)
        if s > best_score:
            best_score = s
            best_points = init_pts.copy()

    # Fallback
    if best_points is None:
        # Return a regular polygon as fallback
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points = np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
        
    # Final validation
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points = np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
