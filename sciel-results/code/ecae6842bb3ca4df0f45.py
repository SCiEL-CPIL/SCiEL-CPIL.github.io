import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13

# EVOLVE-BLOCK-START
def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    # Precompute all pair vectors for efficiency
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / hull area. Higher is better."""
    points = np.asarray(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return -1e10
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective(x):
    """Objective for minimization: negative score."""
    points = x.reshape(-1, 2)
    s = _score(points)
    if s < 0:
        return 1e10
    return -s

def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon centered at origin."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _perturb(points, rng, scale=0.1):
    """Add small random perturbation."""
    return points + rng.uniform(-scale, scale, size=points.shape)

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = -1e10
    
    rng = np.random.RandomState(42)
    
    # Generate initial candidates
    candidates = []
    
    # Regular 13-gon
    candidates.append(_regular_polygon(N))
    
    # Perturbed regular polygons
    for _ in range(5):
        candidates.append(_perturb(_regular_polygon(N), rng, scale=0.05))
    
    # Some random configurations
    for _ in range(3):
        candidates.append(rng.uniform(-1, 1, size=(N, 2)))
    
    # Try each candidate with L-BFGS-B optimization
    for idx, init in enumerate(candidates):
        if time.time() > deadline:
            break
        
        init = np.asarray(init, dtype=float).flatten()
        
        try:
            result = minimize(
                _objective,
                init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < _objective(init):
                opt_points = result.x.reshape(-1, 2)
                s = _score(opt_points)
                
                if s > best_score:
                    best_score = s
                    best_points = opt_points.copy()
                    
        except Exception:
            pass
    
    # If no optimization succeeded, use the best initial candidate
    if best_points is None:
        for init in candidates:
            s = _score(init)
            if s > best_score:
                best_score = s
                best_points = init.copy()
    
    # Final check: ensure we return a valid array
    if best_points is None:
        best_points = _regular_polygon(N)
    
    return np.asarray(best_points, dtype=float)
# EVOLVE-BLOCK-END
