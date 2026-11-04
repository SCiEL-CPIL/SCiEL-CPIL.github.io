"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import time

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """Compute the minimum area over all C(N,3) triangles formed by the points."""
    n = len(points)
    min_area = float('inf')
    # Precompute cross products for all triples
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            pij = points[j] - pi
            for k in range(j + 1, n):
                pik = points[k] - pi
                # Area = 0.5 * |cross(pij, pik)|
                area = 0.5 * abs(pij[0] * pik[1] - pij[1] * pik[0])
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns the score directly (higher is better).
    Returns 0.0 for degenerate/invalid cases.
    """
    points = np.asarray(points)
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Calculate Minimum Triangle Area
    min_area = _min_triangle_area(points)
    
    if min_area <= 0 or not np.isfinite(min_area):
        return 0.0
    
    return min_area / hull_area


def _objective(x):
    """
    Objective function for minimization: negative of the score.
    x is a flattened array of shape (2*N,).
    """
    points = x.reshape(-1, 2)
    s = _score(points)
    if s <= 0:
        return 1e6  # Large penalty for invalid configurations
    return -s


def _generate_initial_points(seed):
    """Generate a diverse set of initial points using various strategies."""
    rng = np.random.RandomState(seed)
    candidates = []
    
    # Strategy 1: Uniform random in unit square
    for _ in range(3):
        pts = rng.uniform(0, 1, (N, 2))
        candidates.append(pts)
    
    # Strategy 2: Points on a circle with small random perturbation
    for _ in range(3):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(-0.1, 0.1, N)
        radii = 1.0 + rng.uniform(-0.05, 0.05, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # Strategy 3: Grid-like arrangement
    for _ in range(2):
        # 4x4 grid minus three points, perturbed
        xs = np.linspace(0, 1, 4)
        ys = np.linspace(0, 1, 4)
        grid = np.array([[x, y] for x in xs for y in ys])
        # Remove three points to get 13 (16 - 3 = 13)
        indices_to_remove = rng.choice(len(grid), size=3, replace=False)
        pts = np.delete(grid, indices_to_remove, axis=0)
        pts = pts + rng.uniform(-0.05, 0.05, (N, 2))
        candidates.append(pts)
    
    # Strategy 4: Fibonacci spiral
    for _ in range(2):
        golden_ratio = (1 + np.sqrt(5)) / 2
        angles = 2 * np.pi * np.arange(N) / golden_ratio
        radii = np.sqrt(np.arange(N) / N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts = pts + rng.uniform(-0.02, 0.02, (N, 2))
        candidates.append(pts)
    
    # Strategy 5: Regular polygon with one point moved inward
    for _ in range(2):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        radii[0] = 0.5  # Move first point inward
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts = pts + rng.uniform(-0.03, 0.03, (N, 2))
        candidates.append(pts)
    
    return candidates


def _polish(points, max_iter=100):
    """
    High-precision polish pass using L-BFGS-B with tighter tolerances.
    """
    x0 = points.flatten()
    try:
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return points


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 5s before 150s limit
    
    # Generate initial candidates
    all_candidates = []
    seed = 0
    while time.time() < deadline and seed < 20:
        candidates = _generate_initial_points(seed)
        all_candidates.extend(candidates)
        seed += 1
    
    # Track best result
    best_score = -1.0
    best_points = None
    
    # Evaluate initial candidates
    for i, pts in enumerate(all_candidates):
        if time.time() > deadline:
            break
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Local optimization on top candidates
    # Sort by score and take top few for polishing
    scored_candidates = []
    for pts in all_candidates:
        s = _score(pts)
        scored_candidates.append((s, pts))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Polish top 10 candidates
    top_k = min(10, len(scored_candidates))
    for i in range(top_k):
        if time.time() > deadline:
            break
        s, pts = scored_candidates[i]
        # L-BFGS-B optimization
        x0 = pts.flatten()
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            opt_pts = result.x.reshape(-1, 2)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Final polish pass on the best point
    if best_points is not None and time.time() < deadline - 5.0:
        polished = _polish(best_points, max_iter=50)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
    
    if best_points is None:
        # Fallback: return random points
        return np.random.uniform(0, 1, (N, 2))
    
    # Ensure the returned points have exactly N points
    if len(best_points) != N:
        # This should not happen with the fixed generator, but as a safeguard
        # If we have more, take first N. If fewer, pad with random? 
        # Better to just return the best valid one found or fallback.
        # Since we know the bug was in generation, and we fixed it, this is just a safety net.
        # If it triggers, it means something else is wrong.
        return np.random.uniform(0, 1, (N, 2))
        
    return best_points

# EVOLVE-BLOCK-END
