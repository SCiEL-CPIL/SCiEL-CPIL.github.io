N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay
import time


def _covering_radius(points):
    """Compute the covering radius as the max circumradius of convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            tri = points[simplex]
            # Circumradius of triangle on unit sphere
            a, b, c = tri
            ab = np.linalg.norm(a - b)
            bc = np.linalg.norm(b - c)
            ca = np.linalg.norm(c - a)
            # Check for degenerate
            if ab < 1e-10 or bc < 1e-10 or ca < 1e-10:
                continue
            # Circumradius formula: R = abc / (4 * area)
            # Area using cross product
            cross = np.cross(b - a, c - a)
            area = 0.5 * np.linalg.norm(cross)
            if area < 1e-10:
                continue
            R = (ab * bc * ca) / (4.0 * area)
            if R > max_r:
                max_r = R
        return max_r
    except Exception:
        return np.pi


def _smooth_covering_radius(points, alpha=50.0):
    """Smooth approximation of max circumradius for gradient computation.
    Uses the fact that covering radius ~ max over points of distance to nearest point,
    but we use a differentiable proxy: for each point on a fine grid, compute min distance."""
    # Use a grid of test points on the sphere
    # Sample points using Fibonacci sphere for test points
    M = 200  # number of test points
    i = np.arange(M) + 0.5
    z_test = 1.0 - 2.0 * i / M
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i
    r_test = np.sqrt(1.0 - z_test * z_test)
    test_pts = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    # Compute distances from test points to all configuration points
    # dist[i,j] = angular distance from test_pts[i] to points[j]
    dots = test_pts @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (M, N)
    
    # For each test point, find min distance
    min_dists = np.min(dists, axis=1)  # (M,)
    
    # Smooth max approximation: softplus-based
    # Use log-sum-exp for smooth max
    # max(x) ≈ (1/alpha) * log(sum(exp(alpha * x)))
    scaled = alpha * min_dists
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_max = log_sum / alpha
    
    return smooth_max


def _optimize_from_initial(points, time_limit):
    """Optimize starting from given initial points."""
    start_time = time.time()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        pts = pts / norms
        return _smooth_covering_radius(pts)
    
    def gradient(x):
        # Use numerical gradient (L-BFGS can use finite differences)
        return None  # Let L-BFGS compute numerically
    
    best_x = points.flatten().copy()
    best_val = objective(best_x)
    
    # Multiple restarts with perturbation
    n_restarts = 5
    step = 0
    
    while time.time() - start_time < time_limit:
        if step == 0:
            x0 = best_x
        else:
            # Perturb
            noise = 0.1 * np.random.randn(N * 3)
            x0 = best_x + noise
        
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
        
        step += 1
    
    # Final normalization
    pts = best_x.reshape(-1, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    pts = pts / norms
    return pts


def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Stage 1: Good initial points - Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure not in a hemisphere
    # Check if all points in closed hemisphere
    for _ in range(10):
        v = np.random.randn(3)
        v /= np.linalg.norm(v)
        if np.all(np.dot(points, v) >= 0) or np.all(np.dot(points, v) <= 0):
            # Reflect one point
            points[np.random.randint(N)] *= -1
        else:
            break
    
    # Stage 2: Optimize
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        points = _optimize_from_initial(points, min(time_remaining - 5, 120))
    
    # Stage 3: Verify and cleanup
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Check for coincident points
    for i_idx in range(N):
        for j_idx in range(i_idx + 1, N):
            if np.linalg.norm(points[i_idx] - points[j_idx]) < 1e-10:
                points[j_idx] = points[j_idx] + 1e-6 * np.random.randn(3)
                points[j_idx] /= np.linalg.norm(points[j_idx])
    
    # Check hemisphere
    in_hemi = False
    for i_idx in range(N):
        v = -points[i_idx]
        if np.all(np.dot(points, v) >= -1e-8):
            in_hemi = True
            break
    if in_hemi:
        for i_idx in range(N):
            if np.all(np.dot(points, -points[i_idx]) < 0):
                points[i_idx] = -points[i_idx]
                break
    
    # Ensure non-finite
    if not np.all(np.isfinite(points)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return points
# EVOLVE-BLOCK-END
