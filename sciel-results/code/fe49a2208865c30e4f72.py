"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time

N = 14


# EVOLVE-BLOCK-START
def _min_triangle_area(points):
    """Compute the minimum area over all triples of points (vectorized)."""
    n = len(points)
    # Compute all cross products for triples
    # points shape (n, 2)
    # For each triple (i,j,k), area = 0.5 * |cross(pj-pi, pk-pi)|
    # Vectorize: for fixed i, compute all (j,k) with j>k>i
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        # All other points
        others = np.delete(points, i, axis=0)
        # Compute cross products of all pairs from others
        # others shape (n-1, 2)
        # For each pair (j,k) in others, cross = (others[j]-pi) x (others[k]-pi)
        # This is (n-1)*(n-2)/2 pairs
        # Vectorized: outer product
        v1 = others[:, 0] - pi[0]  # shape (n-1,)
        v2 = others[:, 1] - pi[1]  # shape (n-1,)
        # Cross product matrix: v1[i]*v2[j] - v2[i]*v1[j]
        cross = np.abs(np.outer(v1, v2) - np.outer(v2, v1))
        # Only consider upper triangle (j > i) to avoid duplicates and zeros
        idx = np.triu_indices(n - 1, k=1)
        areas = 0.5 * cross[idx]
        if len(areas) > 0:
            min_in_i = np.min(areas)
            if min_in_i < min_area:
                min_area = min_in_i
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(points):
    """Score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _objective(x):
    """Objective for optimization: minimize negative score (i.e., maximize score)."""
    points = x.reshape(-1, 2)
    return -_score(points)


def construct():
    """Construct 14 points to maximize min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Strategy: Use differential evolution with a Fourier-based parameterization
    # to ensure convexity, then polish with local optimization.
    
    # Fourier harmonic representation:
    # x(t) = a0 + sum_{k=1}^{K} (a_k cos(k*t) + b_k sin(k*t))
    # y(t) = c0 + sum_{k=1}^{K} (c_k cos(k*t) + d_k sin(k*t))
    # With K=3, we have 2 + 4*3 = 14 parameters... but we need 14 points.
    # Actually, let's use a different approach: parameterize on a convex curve.
    
    # Alternative: Use N points on a convex polygon, parameterized by angles and radii.
    # Or simpler: use differential evolution directly on 28 coordinates with penalty for non-convexity.
    
    # Let's use the Fourier approach as suggested by the knowledge:
    # Represent points as samples of a convex curve.
    # A convex curve can be represented as r(theta) = 1 + sum_{k=1}^{K} a_k cos(k*theta) + b_k sin(k*theta)
    # where r(theta) > 0 for all theta.
    
    # With K harmonics, we have 2K+1 parameters (including constant term).
    # Let's use K=5: 11 parameters. Then sample N=14 points at equal angles.
    
    # Actually, let's try a simpler approach first: direct optimization with DE.
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_points = init_points.copy()
    best_score = _score(best_points)
    
    # Define bounds for DE: points in [-2, 2]^2
    bounds = [(-2, 2)] * (2 * N)
    
    # Use differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            polish=True,
            init='sobol'
        )
        
        if time.time() < deadline - 10:
            candidate_points = result.x.reshape(-1, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
    except Exception:
        pass
    
    # Polish with local optimization (Nelder-Mead or L-BFGS-B)
    if time.time() < deadline - 15 and best_points is not None:
        try:
            result2 = minimize(
                _objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            candidate_points = result2.x.reshape(-1, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
        except Exception:
            pass
    
    # Try a few more restarts if time permits
    if time.time() < deadline - 20:
        for seed in [123, 456, 789]:
            if time.time() > deadline - 10:
                break
            try:
                result = differential_evolution(
                    _objective,
                    bounds=bounds,
                    seed=seed,
                    maxiter=30,
                    popsize=12,
                    tol=1e-5,
                    polish=False
                )
                candidate_points = result.x.reshape(-1, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points
                    
                # Quick polish
                if time.time() < deadline - 10:
                    result2 = minimize(
                        _objective,
                        candidate_points.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    candidate_points = result2.x.reshape(-1, 2)
                    candidate_score = _score(candidate_points)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_points = candidate_points
            except Exception:
                pass
    
    return best_points
# EVOLVE-BLOCK-END
