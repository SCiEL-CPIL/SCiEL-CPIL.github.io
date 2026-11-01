"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _triangle_areas(points):
    """Compute all triangle areas from a set of points.
    Returns array of areas (unsorted) for all C(n,3) triples."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                                  (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    return np.array(areas)


def _score(points):
    """Compute min triangle area / convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
        areas = _triangle_areas(points)
        min_area = np.min(areas)
        return min_area / hull_area
    except Exception:
        return 0.0


def _neg_score(points):
    return -_score(points)


def _initial_points_ellipse(rng, n):
    """Generate n points on an ellipse-like distribution."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Add some randomness to angles
    angles += rng.uniform(-0.3, 0.3, size=n)
    angles = np.sort(angles)
    # Ellipse with varying aspect ratio
    aspect = rng.uniform(0.7, 1.3)
    points = np.column_stack([
        np.cos(angles) * aspect,
        np.sin(angles)
    ])
    # Add small radial perturbation
    radii = rng.uniform(0.95, 1.05, size=n)
    points *= radii[:, np.newaxis]
    return points


def _initial_points_ring(rng, n):
    """Generate n points: some on outer ring, some on inner ring."""
    outer_count = max(3, int(n * 0.6))
    inner_count = n - outer_count
    outer_angles = np.sort(rng.uniform(0, 2 * np.pi, size=outer_count))
    inner_angles = np.sort(rng.uniform(0, 2 * np.pi, size=inner_count))
    outer_r = rng.uniform(0.9, 1.0)
    inner_r = rng.uniform(0.3, 0.6)
    points = np.vstack([
        np.column_stack([np.cos(outer_angles) * outer_r, 
                         np.sin(outer_angles) * outer_r]),
        np.column_stack([np.cos(inner_angles) * inner_r, 
                         np.sin(inner_angles) * inner_r])
    ])
    # Shuffle
    idx = rng.permutation(n)
    return points[idx]


def _initial_points_random(rng, n):
    """Random points in unit square, then normalize to unit area hull."""
    points = rng.random((n, 2))
    # Shift to center
    points -= points.mean(axis=0)
    return points


def _normalize(points):
    """Normalize points so convex hull has unit area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        area = hull.volume
        if area < 1e-12:
            return points
        scale = 1.0 / np.sqrt(area)
        return points * scale
    except Exception:
        return points


def _basinhopping_step(points, rng, step_size):
    """Perform a basinhopping-like step with random perturbation."""
    n = len(points)
    new_points = points.copy()
    # Perturb each point
    for i in range(n):
        dx = rng.gauss(0, step_size)
        dy = rng.gauss(0, step_size)
        new_points[i] += [dx, dy]
    return new_points


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = -np.inf
    
    # Generate diverse initial points
    initial_sets = []
    for trial in range(5):
        r = trial
        if r == 0:
            pts = _initial_points_ellipse(rng, N)
        elif r == 1:
            pts = _initial_points_ring(rng, N)
        elif r == 2:
            pts = _initial_points_ellipse(rng, N)
        elif r == 3:
            pts = _initial_points_ring(rng, N)
        else:
            pts = _initial_points_random(rng, N)
        
        pts = _normalize(pts)
        initial_sets.append(pts)
    
    # Evaluate initial scores
    for pts in initial_sets:
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Multi-phase refinement
    phase_configs = [
        # (step_size, iterations, method)
        (0.1, 50, 'basinhopping'),
        (0.05, 50, 'basinhopping'),
        (0.02, 50, 'basinhopping'),
        (0.01, 30, 'basinhopping'),
    ]
    
    for step_size, iters, method in phase_configs:
        if time.time() > deadline:
            break
        
        # Use current best as starting point
        current = best_points.copy()
        
        for it in range(iters):
            if time.time() > deadline:
                break
            
            # Perturb
            new_pts = _basinhopping_step(current, rng, step_size)
            new_pts = _normalize(new_pts)
            
            new_s = _score(new_pts)
            
            if new_s > _score(current):
                current = new_pts
                if new_s > best_score:
                    best_score = new_s
                    best_points = new_pts.copy()
        
        # Try Powell optimization on best
        if time.time() > deadline:
            break
        
        try:
            result = scipy.optimize.minimize(
                _neg_score, best_points, method='Powell',
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if result.fun < -best_score:
                best_points = result.x
                best_score = -result.fun
        except Exception:
            pass
        
        # Try Nelder-Mead
        if time.time() > deadline:
            break
        
        try:
            result = scipy.optimize.minimize(
                _neg_score, best_points, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < -best_score:
                best_points = result.x
                best_score = -result.fun
        except Exception:
            pass
    
    # Final check
    if best_points is None:
        return np.random.RandomState(N).random((N, 2))
    
    return best_points

# EVOLVE-BLOCK-END
