"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import itertools

# Precompute all C(13,3) = 286 triangle indices
_TRIPLETS = np.array(list(itertools.combinations(range(N), 3)), dtype=np.intp)

def _compute_score(points):
    """Compute min triangle area / convex hull area. Returns 0 if invalid."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (N, 2):
        return 0.0
    
    # Check for duplicate or nearly-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-10:
                return 0.0
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, this is the area
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute all triangle areas vectorized
    p0 = points[_TRIPLETS[:, 0]]
    p1 = points[_TRIPLETS[:, 1]]
    p2 = points[_TRIPLETS[:, 2]]
    
    # Area = 0.5 * |cross(p1-p0, p2-p0)|
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - \
            (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    
    return min_area / hull_area

def _neg_score(points):
    """Negative score for minimization."""
    return -_compute_score(points)

def _objective(x):
    """Objective for differential evolution. x is flat array of 2*N coordinates."""
    points = x.reshape(N, 2)
    return _neg_score(points)

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Differential Evolution
    bounds = [(0, 1) for _ in range(2 * N)]
    
    # Run DE with time-bounded approach
    try:
        # Use a callback to check time
        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=False,
            init='sobol'
        )
        
        if time.time() > deadline:
            # Use best found so far
            pass
        else:
            cand = result.x.reshape(N, 2)
            s = _compute_score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
                
    except Exception:
        pass
    
    # Strategy 2: Local refinement with L-BFGS from best DE result
    if best_points is not None and time.time() < deadline:
        try:
            res = minimize(
                _neg_score,
                best_points.ravel(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            cand = res.x.reshape(N, 2)
            s = _compute_score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass
    
    # Strategy 3: Try multiple random restarts if time allows
    rng = np.random.RandomState(42)
    while time.time() < deadline - 5:
        # Random initialization near unit square
        init = rng.rand(N, 2)
        try:
            res = minimize(
                _neg_score,
                init.ravel(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            cand = res.x.reshape(N, 2)
            s = _compute_score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass
    
    # Final polish
    if best_points is not None and time.time() < deadline - 2:
        try:
            res = minimize(
                _neg_score,
                best_points.ravel(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            cand = res.x.reshape(N, 2)
            s = _compute_score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: uniform grid-like distribution
        best_points = np.array([[i / (N - 1), j / (N - 1)] for i in range(N) for j in range(N)][:N])
        # Actually just use a simple pattern
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([0.5 + 0.45 * np.cos(angles), 0.5 + 0.45 * np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
