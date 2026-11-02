"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

DEADLINE = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples using vectorization."""
    n = len(pts)
    # Vectorized calculation for speed
    # Create indices for all triples
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p0 = pts[i]
    p1 = pts[j]
    p2 = pts[k]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_area


def _neg_score(pts):
    """Negative score for minimization."""
    return -_score(pts)


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _try_optimize(pts, max_iter=300, xatol=1e-8, fatol=1e-10):
    """Try to optimize a point configuration. Returns improved points or None."""
    def _flatten(p):
        return p.flatten()
    
    def _unflatten(x):
        return x.reshape(N, 2)
    
    try:
        result = minimize(
            _neg_score,
            _flatten(pts),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        if result.fun < 0:
            candidate = _unflatten(result.x)
            return candidate
    except Exception:
        pass
    return None


def construct():
    """Construct 13 points maximizing min triangle area / convex hull area."""
    
    # 1. Start with regular polygon
    pts0 = _regular_polygon(N)
    best_pts = pts0.copy()
    best_score = _score(pts0)
    
    # 2. Local refinement from regular polygon
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    
    if remaining > 5:
        candidate = _try_optimize(best_pts, max_iter=500, xatol=1e-9, fatol=1e-11)
        if candidate is not None:
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_pts = candidate
                best_score = cand_score
    
    # 3. Differential Evolution Global Search
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    
    if remaining > 30:
        bounds = [(-1, 1)] * (2 * N)
        
        def objective_de(x):
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                return 0.0
            points = x.reshape(-1, 2)
            return -_score(points)
            
        try:
            # Use DE to find a good candidate
            # popsize=15, maxiter=50 is a reasonable start
            # tol=1e-6
            res_de = differential_evolution(
                objective_de, 
                bounds, 
                maxiter=50, 
                popsize=20, 
                tol=1e-6, 
                seed=42, 
                polish=False
            )
            
            if res_de.fun < -best_score:
                candidate = res_de.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = candidate
        except Exception:
            pass
        
        # Local refinement from DE result
        if best_score > 0:
            elapsed = time.time() - START_TIME
            remaining = DEADLINE - SAFETY_MARGIN - elapsed
            if remaining > 15:
                candidate = _try_optimize(best_pts, max_iter=500, xatol=1e-9, fatol=1e-11)
                if candidate is not None:
                    cand_score = _score(candidate)
                    if cand_score > best_score:
                        best_pts = candidate
                        best_score = cand_score

    # 4. Multi-start with diverse initializations
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    
    # Generate diverse starting points: perturbations of best, random, different radii
    np.random.seed(42)  # For reproducibility in early iterations
    
    while remaining > 15:
        # Different perturbation strategies
        strategy = int(time.time() * 1000) % 4
        
        if strategy == 0:
            # Small perturbation of current best
            pts_init = best_pts + np.random.normal(0, 0.05, size=(N, 2))
        elif strategy == 1:
            # Larger perturbation
            pts_init = best_pts + np.random.normal(0, 0.2, size=(N, 2))
        elif strategy == 2:
            # Regular polygon with random radius and rotation
            r = 0.5 + np.random.random() * 1.0
            theta = np.random.random() * 2 * np.pi
            pts_init = _regular_polygon(N, radius=r)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            pts_init = pts_init @ rot.T
        else:
            # Random points in unit circle
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0.1, 1.0, N))
            pts_init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Optimize from this starting point
        candidate = _try_optimize(pts_init, max_iter=200, xatol=1e-8, fatol=1e-10)
        if candidate is not None:
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_pts = candidate
                best_score = cand_score
        
        elapsed = time.time() - START_TIME
        remaining = DEADLINE - SAFETY_MARGIN - elapsed
    
    # 5. Final fine-tuning if time remains
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    
    if remaining > 10:
        # Try a few more iterations with tighter tolerances
        for _ in range(3):
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                break
            candidate = _try_optimize(best_pts, max_iter=300, xatol=1e-10, fatol=1e-12)
            if candidate is not None:
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_pts = candidate
                    best_score = cand_score
    
    # Final check to ensure valid output
    if _score(best_pts) <= 0:
        # Fallback to regular polygon if something went wrong
        best_pts = _regular_polygon(N)

    return best_pts
# EVOLVE-BLOCK-END
