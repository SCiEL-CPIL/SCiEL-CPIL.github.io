"""Rewritten program: multi-start local search for circle packing in a unit disk."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE_SECONDS = 135.0  # safety margin before 150s kill


def _score(points):
    """Compute the score: min of half min pairwise distance and min margin to boundary."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points are inside or on the unit circle
    dists_to_center = np.sqrt(np.sum(points**2, axis=1))
    if np.any(dists_to_center > 1.0 + 1e-12):
        return 0.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_margin = 1.0 - np.max(dists_to_center)
    r = min(min_pair_dist / 2.0, min_margin)
    if r < 0:
        return 0.0
    return r


def _objective(points_flat):
    """Objective for minimization: negative score (we want to maximize score)."""
    points = points_flat.reshape(N, 2)
    s = _score(points)
    return -s


def _local_search(initial_points, max_iter=500, tol=1e-12):
    """Run L-BFGS-B local search from initial points."""
    result = minimize(
        _objective,
        initial_points.flatten(),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8}
    )
    best_points = result.x.reshape(N, 2)
    best_score = _score(best_points)
    return best_points, best_score


def _hex_lattice_init():
    """Generate initial points on a hexagonal lattice inside the unit disk."""
    points = []
    spacing = 0.2
    while len(points) < N:
        # Generate hex lattice points
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= (1.0 - spacing/2.0)**2:
                    points.append([x, y])
        if len(points) >= N:
            break
        spacing *= 0.95
    points = np.array(points)
    # Sort by distance from origin and take first N
    radii = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(radii)
    return points[idx[:N]]


def _random_init():
    """Generate random initial points inside the unit disk."""
    angles = np.random.uniform(0, 2*np.pi, N)
    radii = np.random.uniform(0, 1.0, N)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def _ring_init():
    """Generate points on concentric rings."""
    points = []
    # Center point
    points.append([0.0, 0.0])
    # Rings
    n_rings = 7
    for ring in range(1, n_rings):
        r = ring * 0.13
        n_pts = min(6 * ring, N - len(points))
        if n_pts <= 0:
            break
        angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
    points = np.array(points[:N])
    return points


def construct():
    start_time = time.time()
    
    # Try multiple initializations and local searches
    initializers = [
        _hex_lattice_init,
        _ring_init,
        _random_init,
        _random_init,
        _random_init,
    ]
    
    best_points = None
    best_score = 0.0
    
    for init_func in initializers:
        if time.time() - start_time > DEADLINE_SECONDS:
            break
        
        initial = init_func()
        points, score = _local_search(initial, max_iter=300)
        
        if score > best_score:
            best_score = score
            best_points = points
    
    # If we still have time, do additional random restarts
    elapsed = time.time() - start_time
    if elapsed < DEADLINE_SECONDS - 5.0 and best_points is not None:
        remaining = DEADLINE_SECONDS - 5.0 - elapsed
        # Estimate time per iteration
        time_per_iter = max(elapsed / max(len(initializers), 1), 0.1)
        max_extra = int(remaining / time_per_iter)
        max_extra = min(max_extra, 10)
        
        for _ in range(max_extra):
            if time.time() - start_time > DEADLINE_SECONDS:
                break
            initial = _random_init()
            points, score = _local_search(initial, max_iter=200)
            if score > best_score:
                best_score = score
                best_points = points
    
    if best_points is None:
        # Fallback: return hex lattice
        best_points = _hex_lattice_init()
    
    return best_points
