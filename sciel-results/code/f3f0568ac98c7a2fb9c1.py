"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 13

def _triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points."""
    n = len(pts)
    areas = []
    # Vectorized computation for speed
    # We can loop over i, and vectorize j, k
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            # v2 for all k > j
            v2s = pts[j+1:] - pts[i]
            # Cross product z-component
            crosses = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas.extend(0.5 * np.abs(crosses))
    return np.array(areas)

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Compute min triangle area / hull area."""
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
    hull_area = _hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_objective(params, beta):
    """
    Objective function for optimization.
    Minimizes - (soft_min(triangle_areas) / hull_area)
    """
    pts = params.reshape(N, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-9:
        return 1e9
    areas = _triangle_areas(pts)
    # Soft min using log-sum-exp
    # To avoid overflow/underflow, shift by max
    max_area = np.max(areas)
    shifted = -beta * (areas - max_area)
    # exp(shifted) is safe since shifted <= 0
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = max_area - log_sum / beta
    return -soft_min_val / hull_area

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def _halton_sequence(index, base):
    f = 1.0
    r = 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r

def _generate_halton_points(n):
    pts = []
    for i in range(1, n + 1):
        x = _halton_sequence(i, 2)
        y = _halton_sequence(i, 3)
        pts.append([x, y])
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    return pts

def _polish(pts, time_budget):
    """Refine points using L-BFGS-B with high beta."""
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Try a few betas
    betas = [50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() - start_time > time_budget:
            break
            
        def objective(params):
            return _soft_min_objective(params, beta)
        
        try:
            res = minimize(
                objective,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500}
            )
            if res.success:
                new_pts = res.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts
        except Exception:
            pass
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    best_pts = None
    best_score = -1e9
    
    # Initial candidates
    candidates = [
        _regular_polygon(N),
        _generate_halton_points(N)
    ]
    
    # Add some random perturbations of regular polygon
    rng = np.random.default_rng(42)
    base = _regular_polygon(N)
    for _ in range(3):
        noise = rng.normal(0, 0.2, size=N * 2)
        perturbed = base.flatten() + noise
        candidates.append(perturbed.reshape(N, 2))
    
    # Phase 1: Differential Evolution (Global Search)
    # We optimize the 26 coordinates directly.
    # Bounds: [-2, 2] for x and y to allow for various shapes
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    # Run DE with a limited budget
    de_time_budget = 60.0
    if time.time() - start_time < time_limit - de_time_budget:
        try:
            # Use a smaller population and fewer iterations to fit in time
            # popsize=15, maxiter=50
            de_res = differential_evolution(
                _soft_min_objective,
                bounds,
                args=(100.0,), # beta=100 for DE
                maxiter=50,
                popsize=15,
                tol=0.01,
                seed=42,
                polish=False, # We will polish manually
                init='sobol'
            )
            if de_res.success:
                de_pts = de_res.x.reshape(N, 2)
                de_score = _score(de_pts)
                if de_score > best_score:
                    best_score = de_score
                    best_pts = de_pts
        except Exception:
            pass
    
    # Phase 2: Local Polish on best DE result and initial candidates
    remaining_time = time_limit - (time.time() - start_time)
    
    if best_pts is not None:
        polished_pts, polished_score = _polish(best_pts, remaining_time - 10.0)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts
    else:
        # Fallback to polishing the regular polygon if DE failed
        polished_pts, polished_score = _polish(candidates[0], remaining_time - 10.0)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts

    # Phase 3: If we have time, try polishing other candidates
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0:
        for cand in candidates:
            if time.time() - start_time > time_limit - 5.0:
                break
            if np.allclose(cand, best_pts):
                continue
            polished_pts, polished_score = _polish(cand, remaining_time - 5.0)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 5.0:
                break

    if best_pts is None:
        return _regular_polygon(N)
        
    return best_pts

# EVOLVE-BLOCK-END
