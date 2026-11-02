"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

def _all_triangle_areas(pts):
    """Vectorized areas of all C(n,3) triangles."""
    n = len(pts)
    parts = []
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            v2 = pts[j + 1:] - pts[i]
            if len(v2) == 0:
                continue
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            parts.append(0.5 * cross)
    return np.concatenate(parts)

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _convex_hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    min_area = np.min(_all_triangle_areas(pts))
    return min_area / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _soft_min_triangle_area(pts, alpha=50.0):
    """
    Smooth approximation of the minimum triangle area.
    Uses -log(sum(exp(-alpha * areas))) / alpha
    This is differentiable and guides optimizers better.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    log_sum = -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            if len(areas) > 0:
                min_a = np.min(areas)
                if min_a > 0:
                    shifted = -alpha * (areas - min_a)
                    log_sum_partial = np.log(np.sum(np.exp(shifted))) - alpha * min_a
                    if log_sum_partial > log_sum:
                        log_sum = log_sum_partial
    if log_sum == -np.inf:
        return 0.0
    return -log_sum / alpha

def _hull_area_safe(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 1.0

def _smooth_score(pts):
    """Smooth score surrogate."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area_safe(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _soft_min_triangle_area(pts, alpha=50.0) / hull_a

def _neg_smooth_score(params):
    pts = params.reshape(-1, 2)
    return -_smooth_score(pts)

def _generate_initializations():
    """Generate diverse starting points."""
    inits = []
    rng = np.random.RandomState(42)
    
    # 1. Regular Polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular Polygon + Center (if N is odd, N-1 on ring, 1 center)
    m = N - 1
    angles_m = np.linspace(0, 2 * np.pi, m, endpoint=False)
    outer = np.column_stack([np.cos(angles_m), np.sin(angles_m)])
    center = np.array([[0.0, 0.0]])
    inits.append(np.vstack([outer, center]))
    
    # 3. Concentric Rings
    # Try splitting N into two rings
    for n1 in range(3, N-2):
        n2 = N - n1
        ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
        r1 = 0.5
        r2 = 1.0
        ring1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
        ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
        inits.append(np.vstack([ring1, ring2]))
        
    # 4. Jittered Regular Polygon
    for jitter in [0.01, 0.05, 0.1]:
        r = 1.0 + jitter * rng.randn(N)
        inits.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
        
    # 5. Random points in unit disk
    for _ in range(5):
        r = rng.rand(N) ** 0.5
        theta = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
    # 6. Grid-like
    xs = np.linspace(-0.5, 0.5, 4)
    ys = np.linspace(-0.5, 0.5, 4)
    xx, yy = np.meshgrid(xs, ys)
    pts_grid = np.column_stack([xx.flatten(), yy.flatten()])
    inits.append(pts_grid[:13])
    
    return inits

def _refine_local(pts, deadline):
    """
    Perform local refinement using a two-stage approach:
    1. Smooth optimization to navigate the landscape.
    2. Exact optimization to polish the result.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Stage 1: Smooth refinement
    # We use Powell as it is robust for non-smooth-ish functions (though we smooth it).
    # We increase maxiter and tighten tolerances compared to the seed.
    try:
        res_smooth = minimize(
            _neg_smooth_score,
            current_pts.flatten(),
            method='Powell',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 50000}
        )
        smooth_pts = res_smooth.x.reshape(-1, 2)
        smooth_score = _score(smooth_pts)
        if smooth_score > current_score:
            current_pts = smooth_pts
            current_score = smooth_score
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return current_pts, current_score

    # Stage 2: Exact refinement
    # Use Nelder-Mead with very tight tolerances to polish the local minimum.
    try:
        res_exact = minimize(
            _neg_score,
            current_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 100000}
        )
        exact_pts = res_exact.x.reshape(-1, 2)
        exact_score = _score(exact_pts)
        if exact_score > current_score:
            current_pts = exact_pts
            current_score = exact_score
    except Exception:
        pass
        
    return current_pts, current_score

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop at 130s to ensure return.
    deadline = start_time + 130.0
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    
    # 1. Evaluate initializations
    inits = _generate_initializations()
    for init_pts in inits:
        if time.time() > deadline - 30.0:
            break
        s = _score(init_pts)
        if s > best_score:
            best_score = s
            best_pts = init_pts.copy()
            
    # 2. Optimization Loop
    # We will try multiple restarts from the best few initializations and the current best
    # Strategy: 
    #   a) Smooth optimization to refine structure
    #   b) Exact optimization to maximize true score
    
    candidates = []
    # Add top 3 initializations
    scored_inits = [(_score(p), p) for p in inits]
    scored_inits.sort(key=lambda x: -x[0])
    for s, p in scored_inits[:5]:  # Take top 5 to ensure diversity in local optima
        candidates.append(p)
    candidates.append(best_pts)
    
    # Remove duplicates roughly
    unique_candidates = []
    for c in candidates:
        if not any(np.allclose(c, u, atol=1e-5) for u in unique_candidates):
            unique_candidates.append(c)
            
    # Refine each candidate
    for init_pts in unique_candidates:
        if time.time() > deadline - 10.0:
            break
            
        refined_pts, refined_score = _refine_local(init_pts, deadline - 5.0)
        
        # Update global best
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()

    # 3. Differential Evolution for global exploration if time permits
    # DE is expensive, so we only run it if we have substantial time left.
    if time.time() < deadline - 40.0:
        try:
            # Use a smaller population and fewer iterations to fit in the time budget
            # while still exploring global space.
            bounds = [(-1.5, 1.5)] * (2 * N)
            res_de = differential_evolution(
                _neg_score,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                init='latinhypercube',
                polish=False # We will polish manually to save time
            )
            de_pts = res_de.x.reshape(-1, 2)
            
            # Polish the DE result
            polished_pts, polished_score = _refine_local(de_pts, deadline - 5.0)
            
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts.copy()
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
