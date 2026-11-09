import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """
    Fast calculation of minimum triangle area using vectorized operations.
    Points: (N, 2) array. Returns minimum area (not doubled).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            v1 = points[j] - pi
            v2s = points[j + 1:] - pi
            cross_vals = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            if len(areas) > 0:
                cur_min = np.min(areas)
                if cur_min < min_area:
                    min_area = cur_min
    return min_area

def _score(points):
    """
    Score = Min Triangle Area / Convex Hull Area.
    """
    try:
        if np.allclose(points, points[0]):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area < 1e-12:
            return 0.0
            
        min_tri_area = _min_triangle_area_fast(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def _negative_score(points):
    """
    Objective function for minimizer.
    """
    return -_score(points)

def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _random_convex_points(seed):
    """
    Generate random points in convex position.
    Strategy: Pick random angles, sort them, assign random radii.
    This ensures the points are in convex position (no point is inside the convex hull of others)
    if the radii are sufficiently large relative to the angular spread, 
    but strictly speaking, random radii can lead to non-convex positions.
    To guarantee convex position, we can just take random points and compute convex hull,
    but we need exactly N points.
    
    A robust way to generate N points in convex position:
    1. Generate N random angles, sort them.
    2. Generate N random radii from a distribution that keeps them near the boundary.
       Actually, any set of points with distinct angles sorted around the origin 
       forms a convex polygon IF they are all on the boundary of their convex hull.
       If we pick points on a circle, they are convex. 
       If we pick random radii, some might be inside.
       
    Better strategy for initialization:
    Start with a regular polygon and apply a random affine transformation? 
    No, affine transformations preserve the ratio min_area/hull_area? 
    Area scales by |det(A)|. Min triangle area scales by |det(A)|. Hull area scales by |det(A)|.
    So the score is invariant under affine transformations!
    
    Wait, is the score invariant under affine transformations?
    Let A be an affine transformation matrix.
    Area(T) = 0.5 * |det([p1-p0, p2-p0])|
    If points are transformed by M: p' = M p.
    p1'-p0' = M(p1-p0), p2'-p0' = M(p2-p0).
    det([M(p1-p0), M(p2-p0)]) = det(M) * det([p1-p0, p2-p0]).
    So every triangle area scales by |det(M)|.
    The convex hull area also scales by |det(M)|.
    Therefore, the ratio (Min Triangle Area) / (Hull Area) is INVARIANT under linear transformations (and translations).
    
    This is a HUGE insight.
    The problem is effectively about the shape of the point set up to affine equivalence.
    The regular polygon is a specific shape.
    We can parameterize the space of affine-invariant shapes.
    However, since we are just optimizing coordinates, we can fix the scale and rotation to reduce dimensions,
    or just optimize in 2D and rely on the optimizer to find the shape.
    But knowing the invariance helps: we don't need to worry about the points drifting away from the origin or scaling up/down.
    We can normalize the points after each step to keep them in a standard frame (e.g., centroid at 0, unit variance or fixed hull area).
    
    Let's use this invariance to improve stability.
    After generating points, we can normalize them:
    1. Center at origin.
    2. Scale such that the convex hull area is 1 (or some constant).
    Actually, the score is already normalized. So the optimizer doesn't care about scale.
    But numerical stability might suffer if points get very large or small.
    Let's add a normalization step in the score function or wrapper.
    """
    rng = np.random.default_rng(seed)
    # Generate points on a random ellipse or similar?
    # Let's just generate random points and ensure convexity by sorting angles and setting radii to 1 + small noise?
    # If radii are close to 1, they are close to the circle, thus convex.
    angles = np.sort(rng.uniform(0, 2 * np.pi, N))
    # Add small noise to angles to avoid degeneracies
    angles += rng.normal(0, 0.01, N)
    angles = np.sort(angles)
    radii = 1.0 + rng.normal(0, 0.05, N)
    radii = np.abs(radii) # Ensure positive
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _normalize_points(points):
    """
    Normalize points to have centroid at origin and unit variance (or similar) 
    to prevent drift, although score is invariant.
    This helps numerical stability of the optimizer.
    """
    # Center
    points = points - np.mean(points, axis=0)
    # Scale to have mean squared distance 1
    dist_sq = np.sum(points**2, axis=1)
    mean_dist_sq = np.mean(dist_sq)
    if mean_dist_sq > 1e-12:
        points = points / np.sqrt(mean_dist_sq)
    return points

def _negative_score_normalized(points):
    """
    Objective with normalization.
    """
    # Flatten to 1D for optimizer compatibility if needed, but we pass 2D
    # Ensure input is 2D
    if points.ndim == 1:
        points = points.reshape(N, 2)
    
    # Normalize
    pts = _normalize_points(points)
    
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_points = points.copy()
            best_score = score

    # Strategy: 
    # 1. Start with Regular Polygon (known good baseline).
    # 2. Use CMA-ES (or a robust derivative-free optimizer) on the 28 coordinates.
    #    Since the problem is affine invariant, the landscape is somewhat flat in scale/rotation directions.
    #    Normalization helps.
    # 3. Multiple restarts from random convex configurations to escape local optima.
    
    from scipy.optimize import minimize
    
    # Initial best: Regular Polygon
    reg_poly = _regular_polygon(N)
    reg_score = _score(reg_poly)
    update_best(reg_poly, reg_score)
    
    # We will run several CMA-ES instances.
    # CMA-ES is robust but slow. 
    # Alternative: Use a custom evolutionary strategy or just many Nelder-Mead restarts?
    # The prompt asks for a DIFFERENT strategy. The current one is NM on polar.
    # Let's use **Differential Evolution** from scipy? It's global but can be slow.
    # Or **Basinhopping**?
    
    # Let's implement a simple **Particle Swarm Optimization (PSO)** or just **CMA-ES** via scipy.
    # scipy.optimize.minimize with method='Nelder-Mead' is what we want to avoid as the *primary* global search, 
    # though we can use it for local refinement.
    
    # Let's try **CMA-ES** manually or via a simple implementation? 
    # SciPy doesn't have CMA-ES built-in in older versions, but `minimize` with method='CMA-ES' is not standard.
    # However, we can use **Differential Evolution** (method='differential_evolution') which is global.
    
    # DE bounds: 
    # Since we normalize, the scale doesn't matter. 
    # But DE needs bounds. 
    # Let's bound the coordinates to [-2, 2] after normalization? 
    # No, DE operates on the raw variables. 
    # If we use the normalized objective, the variables are the raw coordinates.
    # But the objective is flat along scale/rotation. DE might struggle with flat directions.
    
    # Better approach: 
    # Use **Nelder-Mead** but with **many random restarts** from **diverse initializations** (not just regular polygon).
    # The current program does restarts from regular polygon + noise.
    # My "different strategy" is:
    # 1. Use **Random Affine Transformations** of the regular polygon as diverse starts?
    #    Since score is affine invariant, transforming the regular polygon doesn't change the score! 
    #    So that's useless for exploration.
    
    # 2. Use **Random Convex Configurations** as starts.
    #    This explores shapes far from the regular polygon.
    
    # Let's combine:
    # - Phase 1: Generate 10-20 random convex configurations.
    # - Phase 2: Run Nelder-Mead on each for a short time to find local optima.
    # - Phase 3: Take the best of these and run a longer Nelder-Mead or a different optimizer.
    
    # This is structurally different from "Perturb regular polygon and NM".
    # It is "Sample random shapes, refine each, keep best".
    
    rng = np.random.default_rng(42)
    
    num_restarts = 50 # We have time for many quick NMs
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if i == 0:
            # Start with regular polygon
            x0 = reg_poly.copy()
        else:
            # Random convex-ish points
            x0 = _random_convex_points(rng.integers(0, 2**32))
            
        # Normalize initial point
        x0 = _normalize_points(x0)
        
        # Run Nelder-Mead
        # Use a callback to check time
        def callback(xk):
            if time.time() > deadline - 5:
                raise KeyboardInterrupt()
        
        try:
            res = minimize(_negative_score_normalized, x0, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
                           callback=callback)
            
            opt_pts = _normalize_points(res.x)
            opt_score = _score(opt_pts)
            update_best(opt_pts, opt_score)
            
        except KeyboardInterrupt:
            # Time's up for this restart, but we might have a partial result? 
            # Minimize doesn't return partials easily on exception. 
            # We can just evaluate the current best guess if we had one, but we don't store intermediate states in callback easily.
            # Let's just break.
            pass
        except Exception:
            pass

    # If we still have time, do a final refinement on the best point found
    if time.time() < deadline - 10 and best_points is not None:
        x0 = _normalize_points(best_points)
        def callback_final(xk):
            if time.time() > deadline - 2:
                raise KeyboardInterrupt()
        try:
            res = minimize(_negative_score_normalized, x0, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
                           callback=callback_final)
            opt_pts = _normalize_points(res.x)
            opt_score = _score(opt_pts)
            update_best(opt_pts, opt_score)
        except KeyboardInterrupt:
            pass
        except Exception:
            pass

    if best_points is None:
        best_points = reg_poly
        
    return best_points

# EVOLVE-BLOCK-END
