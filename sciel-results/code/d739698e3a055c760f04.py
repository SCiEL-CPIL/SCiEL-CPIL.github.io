import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
from scipy.special import logsumexp
import time

N = 13

def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    # Vectorized calculation for speed
    # i, j, k combinations
    # This loop is O(N^3) but N=13 is small (286 triples)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax = pts[j, 0] - pts[i, 0]
                ay = pts[j, 1] - pts[i, 1]
                bx = pts[k, 0] - pts[i, 0]
                by = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(ax * by - ay * bx)
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score_true(params):
    """True score: min triangle area / hull area."""
    pts = params.reshape((N, 2))
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_a = _min_triangle_area(pts)
    return min_a / hull_a

def _score_smooth(params, beta=20.0):
    """Smoothed score for global optimization."""
    pts = params.reshape((N, 2))
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    
    # Calculate all triangle areas
    areas = []
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax = pts[j, 0] - pts[i, 0]
                ay = pts[j, 1] - pts[i, 1]
                bx = pts[k, 0] - pts[i, 0]
                by = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(ax * by - ay * bx)
                areas.append(area)
    
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
        
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # We want to maximize this, so we minimize its negative
    soft_min_val = -1.0 / beta * logsumexp(-beta * areas)
    
    return soft_min_val / hull_a

def _neg_score_true(params):
    return -_score_true(params)

def _neg_score_smooth(params, beta=20.0):
    return -_score_smooth(params, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_score = 0.0
    best_pts = None

    # 1. Initial structured guesses
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    score_reg = _score_true(pts_reg.flatten())
    if score_reg > best_score:
        best_score = score_reg
        best_pts = pts_reg.copy()

    # Concentric rings: 7 outer, 6 inner
    angles7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    outer = np.column_stack([np.cos(angles7), np.sin(angles7)])
    inner = np.column_stack([np.cos(angles6) * 0.5, np.sin(angles6) * 0.5])
    pts_ring = np.vstack([outer, inner])
    score_ring = _score_true(pts_ring.flatten())
    if score_ring > best_score:
        best_score = score_ring
        best_pts = pts_ring.copy()

    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(20):
        pts_rand = rng.uniform(-1, 1, size=(N, 2))
        score_rand = _score_true(pts_rand.flatten())
        if score_rand > best_score:
            best_score = score_rand
            best_pts = pts_rand.copy()

    if time.time() > deadline - 20:
        return best_pts

    # 2. Differential Evolution with Smoothed Objective
    # Use a beta that balances smoothness and accuracy
    # Start with moderate beta
    beta_de = 20.0
    
    def de_objective(x):
        return _neg_score_smooth(x, beta_de)

    try:
        # Bounds: points should stay within a reasonable range
        # If hull area is 1, points are roughly within [-2, 2]
        bounds = [(-3, 3)] * (2 * N)
        
        # DE settings
        # popsize=20, maxiter=100 might be too slow. 
        # Let's monitor time.
        result = differential_evolution(
            de_objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-5,
            polish=True,
            init='sobol',
            updating='deferred'
        )
        
        if result.fun < -best_score:
            candidate_pts = result.x.reshape((N, 2))
            candidate_score = _score_true(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
                
    except Exception:
        pass

    if time.time() > deadline - 10:
        return best_pts

    # 3. Local Refinement with Nelder-Mead on True Objective
    # Nelder-Mead is good for non-smooth functions
    if best_pts is not None:
        starts = [best_pts.flatten()]
        
        # Add a few perturbed starts
        for i in range(5):
            noise = rng.normal(0, 0.1, size=(N, 2))
            starts.append((best_pts + noise).flatten())

        for start in starts:
            if time.time() > deadline - 2:
                break
                
            try:
                res = minimize(
                    _neg_score_true,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                
                if res.fun < -best_score:
                    candidate_pts = res.x.reshape((N, 2))
                    candidate_score = _score_true(candidate_pts)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_pts = candidate_pts.copy()
                        
            except Exception:
                continue

    return best_pts
