"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill


def _min_triangle_area(pts):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(pts)
    if n < 3:
        return 0.0
    # Vectorized computation of all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj - pi, pk - pi)|
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score_points(pts):
    """Score = min triangle area / convex hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_a


def _neg_score(pts_flat):
    """Negative score for minimization, with penalty for degenerate hulls."""
    pts = pts_flat.reshape(-1, 2)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1.0  # Large penalty
    min_ta = _min_triangle_area(pts)
    return -min_ta / hull_a


def _logsumexp_min_surrogate(pts_flat, temperature=0.01):
    """
    Smooth surrogate: -log(sum(exp(-areas/temperature))) approximates -min(areas).
    This is differentiable and helps gradient-based optimizers.
    We maximize the surrogate (minimize negative).
    """
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    areas = np.zeros((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                areas[idx] = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                idx += 1
    
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1.0
    
    # Normalize areas by hull area to make temperature scale-invariant
    norm_areas = areas / hull_a
    
    # Log-sum-exp trick for numerical stability
    shifted = -norm_areas / temperature
    max_shifted = np.max(shifted)
    lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    
    # We want to maximize the minimum, so minimize -lse (which approximates -min)
    return lse  # Minimizing this maximizes the log-sum-exp of areas


def _neg_surrogate(pts_flat, temperature=0.05):
    """Negative of the surrogate for minimization."""
    return _logsumexp_min_surrogate(pts_flat, temperature)


def _refine_exact(pts_flat, max_iter=200, timeout=10.0):
    """Refine using exact objective with Powell's method."""
    start_time = time.time()
    result = minimize(
        _neg_score,
        pts_flat,
        method='Powell',
        options={'maxiter': max_iter, 'ftol': 1e-10, 'xtol': 1e-10}
    )
    elapsed = time.time() - start_time
    return result.x, elapsed


def _try_surrogate_then_refine(init_pts, temperature=0.05, timeout_surrogate=20.0, timeout_refine=10.0):
    """
    Two-phase optimization:
    1. Optimize the smooth log-sum-exp surrogate with BFGS
    2. Refine with exact objective using Powell
    """
    start_time = time.time()
    init_flat = init_pts.flatten()
    
    # Phase 1: Surrogate optimization
    result1 = minimize(
        lambda x: _neg_surrogate(x, temperature),
        init_flat,
        method='BFGS',
        options={'maxiter': 100, 'gtol': 1e-6}
    )
    elapsed1 = time.time() - start_time
    if elapsed1 > timeout_surrogate:
        return init_pts, _score_points(init_pts)
    
    best_pts = result1.x.reshape(-1, 2)
    best_score = _score_points(best_pts)
    
    # Phase 2: Exact refinement
    remaining = timeout_surrogate + timeout_refine - (time.time() - start_time)
    if remaining > 1.0:
        refined_pts, elapsed2 = _refine_exact(result1.x, max_iter=100, timeout=remaining)
        refined_score = _score_points(refined_pts.reshape(-1, 2))
        if refined_score > best_score:
            best_pts = refined_pts.reshape(-1, 2)
            best_score = refined_score
    
    return best_pts, best_score


def construct():
    """Construct 13 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    
    # Initial candidate: regular 13-gon on unit circle
    angles = 2 * np.pi * np.arange(N) / N
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    hull_a = _hull_area(init_pts)
    if hull_a > 0:
        init_pts = init_pts / np.sqrt(hull_a)
    
    best_pts = init_pts.copy()
    best_score = _score_points(best_pts)
    
    # Try multiple initializations with surrogate + refinement
    rng = np.random.RandomState(42)
    
    # Number of attempts based on available time
    max_attempts = 5
    for attempt in range(max_attempts):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 20.0:
            break
        
        # Generate a perturbed initialization
        if attempt == 0:
            init = best_pts.copy()
        else:
            # Random perturbation of the best
            noise = rng.normal(0, 0.1, size=(N, 2))
            init = best_pts + noise
            # Renormalize
            hull_a = _hull_area(init)
            if hull_a > 0:
                init = init / np.sqrt(hull_a)
        
        # Try different temperatures
        temps = [0.02, 0.05, 0.1]
        temp = temps[attempt % len(temps)]
        
        try:
            result_pts, result_score = _try_surrogate_then_refine(
                init, temperature=temp, timeout_surrogate=15.0, timeout_refine=8.0
            )
            if result_score > best_score:
                best_pts = result_pts.copy()
                best_score = result_score
        except Exception:
            continue
    
    # Final check and return
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS:
        # Return whatever we have
        return best_pts
    
    return best_pts
# EVOLVE-BLOCK-END
