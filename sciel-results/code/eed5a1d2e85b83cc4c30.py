"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

# EVOLVE-BLOCK-START

def _all_triangle_areas(pts):
    """Compute all triangle areas vectorized. Returns array of shape (C(N,3),)."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas.append(0.5 * abs(cross))
    return np.array(areas)

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return None

def _score(pts):
    """Compute min triangle area / hull area."""
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return -1e9
    hull_area = _hull_area(pts)
    if hull_area is None or hull_area < 1e-9:
        return -1e9
    areas = _all_triangle_areas(pts)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_value(areas, temperature):
    """Compute soft minimum using log-sum-exp trick."""
    if len(areas) == 0:
        return 0.0
    neg_areas = -areas
    max_val = np.max(neg_areas)
    if max_val > 500:
        max_val = 500
    shifted = neg_areas - max_val
    log_sum = max_val + np.log(np.sum(np.exp(shifted)))
    return -temperature * log_sum

def _soft_min_ratio(pts, temperature):
    """Compute soft min triangle area ratio."""
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return -1e9
    hull_area = _hull_area(pts)
    if hull_area is None or hull_area < 1e-9:
        return -1e9
    areas = _all_triangle_areas(pts)
    soft_min = _soft_min_value(areas, temperature)
    return soft_min / hull_area

def _neg_soft_min_ratio(pts, temperature):
    """Negative of soft min ratio for minimization."""
    return -_soft_min_ratio(pts, temperature)

def _halton_sequence(index, base):
    """Generate Halton sequence value."""
    f = 1.0
    r = 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r

def _generate_halton_points(n):
    """Generate n Halton points in [0,1]^2, centered."""
    pts = []
    for i in range(1, n + 1):
        x = _halton_sequence(i, 2)
        y = _halton_sequence(i, 3)
        pts.append([x, y])
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    return pts

def _regular_polygon(n, radius=1.0):
    """Generate regular polygon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def _concentric_rings(n):
    """Generate points in concentric ring patterns."""
    configs = [
        (1, 12),
        (3, 10),
        (5, 8),
    ]
    results = []
    for inner, outer in configs:
        pts = []
        if inner > 0:
            if inner == 1:
                pts.append([0.0, 0.0])
            else:
                inner_angles = np.linspace(0, 2 * np.pi, inner, endpoint=False)
                inner_r = 0.3
                for a in inner_angles:
                    pts.append([inner_r * np.cos(a), inner_r * np.sin(a)])
        outer_angles = np.linspace(0, 2 * np.pi, outer, endpoint=False)
        for a in outer_angles:
            pts.append([np.cos(a), np.sin(a)])
        pts = np.array(pts[:n])
        if len(pts) == n:
            results.append(pts)
    return results

def _structured_grid(n):
    """Generate structured grid-like points."""
    results = []
    for cols in range(3, 6):
        rows = n // cols
        if rows * cols < n:
            continue
        pts = []
        for r in range(rows):
            for c in range(cols):
                if len(pts) >= n:
                    break
                x = (c - (cols - 1) / 2) * 0.6
                y = (r - (rows - 1) / 2) * 0.6
                pts.append([x, y])
        if len(pts) == n:
            pts = np.array(pts)
            results.append(pts)
    return results

def _generate_starts(rng):
    """Generate diverse starting configurations."""
    starts = []
    starts.append(_regular_polygon(N))
    starts.append(_generate_halton_points(N))
    for ring_pts in _concentric_rings(N):
        starts.append(ring_pts)
    for grid_pts in _structured_grid(N):
        starts.append(grid_pts)
    for _ in range(8):
        pts = rng.random((N, 2)) * 2 - 1
        pts -= np.mean(pts, axis=0)
        starts.append(pts)
    base = _regular_polygon(N)
    for _ in range(5):
        noise = rng.normal(0, 0.2, size=N * 2)
        perturbed = base.flatten() + noise
        perturbed = perturbed.reshape(N, 2)
        perturbed -= np.mean(perturbed, axis=0)
        starts.append(perturbed)
    return starts

def _optimize_start(start_pts, start_temp, end_temp, maxiter, time_budget):
    """Optimize from a single starting point with adaptive temperature."""
    start_time = time.time()
    best_result = start_pts.flatten().copy()
    best_score = _score(start_pts)
    
    n_steps = min(maxiter, 200)
    temps = np.linspace(start_temp, end_temp, n_steps)
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        temp = temps[step]
        
        def objective(params):
            pts = params.reshape(N, 2)
            return _neg_soft_min_ratio(pts, temp)
        
        try:
            res = minimize(
                objective,
                best_result,
                method='L-BFGS-B',
                options={'maxiter': max(10, maxiter // n_steps), 'maxfun': 500, 'disp': False}
            )
            if res.success or res.fun < objective(best_result):
                new_pts = res.x.reshape(N, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_result = res.x.copy()
        except Exception:
            pass
    
    return best_result.reshape(N, 2), best_score

def _local_refinement(pts, rng, n_trials, time_budget):
    """Local refinement with random perturbations and optimization."""
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for trial in range(n_trials):
        if time.time() - start_time > time_budget:
            break
        
        noise_scale = 0.05 * (1.0 - trial / n_trials)
        noise = rng.normal(0, noise_scale, size=N * 2)
        perturbed = best_pts.flatten() + noise
        perturbed = perturbed.reshape(N, 2)
        perturbed -= np.mean(perturbed, axis=0)
        
        result, score = _optimize_start(perturbed, 0.01, 0.001, 500, time_budget - (time.time() - start_time))
        
        if score > best_score:
            best_score = score
            best_pts = result
    
    return best_pts, best_score

def _fourier_convex_shape(params, n_points=N, n_modes=3):
    """
    Generate n_points from a convex shape defined by Fourier series of radius.
    params: [r0, a1, b1, a2, b2, a3, b3] (7 params for 3 modes)
    r(theta) = r0 + a1*cos(theta) + b1*sin(theta) + a2*cos(2theta) + b2*sin(2theta) + ...
    Points are placed at evenly spaced angles, with radius from the Fourier series.
    """
    r0 = params[0]
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    radii = np.full(n_points, r0)
    for m in range(1, n_modes + 1):
        a_idx = 1 + 2 * (m - 1)
        b_idx = 2 + 2 * (m - 1)
        if a_idx < len(params):
            a = params[a_idx]
            b = params[b_idx]
            radii += a * np.cos(m * angles) + b * np.sin(m * angles)
    
    # Ensure positive radii
    radii = np.maximum(radii, 0.1)
    
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    pts -= np.mean(pts, axis=0)
    return pts

def _optimize_fourier(rng, time_budget):
    """
    Optimize Fourier parameters to find a good convex configuration.
    Uses differential evolution-like random search over the low-dimensional param space.
    """
    start_time = time.time()
    n_modes = 3
    n_params = 1 + 2 * n_modes  # r0 + a_m, b_m for m=1..n_modes
    
    best_score = -1e9
    best_params = None
    best_pts = None
    
    # Initial random search over Fourier parameters
    n_trials = 50
    
    for trial in range(n_trials):
        if time.time() - start_time > time_budget:
            break
        
        # Random parameters
        r0 = 0.8 + rng.random() * 0.4  # Base radius
        params = np.array([r0])
        for m in range(1, n_modes + 1):
            a = rng.normal(0, 0.15 / m)  # Decay with frequency
            b = rng.normal(0, 0.15 / m)
            params = np.append(params, [a, b])
        
        pts = _fourier_convex_shape(params, N, n_modes)
        score = _score(pts)
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_pts = pts.copy()
    
    # Refine the best Fourier shape with local optimization
    if best_pts is not None:
        remaining = time_budget - (time.time() - start_time)
        if remaining > 5.0:
            result, score = _optimize_start(best_pts, 0.01, 0.001, 200, remaining - 2.0)
            if score > best_score:
                best_score = score
                best_pts = result
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin from 150s kill limit
    
    rng = np.random.default_rng(42)
    
    best_score = -1e9
    best_pts = None
    
    # Generate diverse starts
    starts = _generate_starts(rng)
    
    # Phase 1: Coarse optimization from multiple starts
    for i, start in enumerate(starts):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 10.0:
            break
        
        start_temp = 0.1 if i < 5 else 0.05
        end_temp = 0.01
        maxiter = 100
        
        result, score = _optimize_start(start, start_temp, end_temp, maxiter, remaining - 20.0)
        
        if score > best_score:
            best_score = score
            best_pts = result
    
    # Phase 2: Fourier-based convex shape search
    remaining = time_limit - (time.time() - start_time)
    if remaining > 15.0:
        fourier_pts, fourier_score = _optimize_fourier(rng, remaining - 10.0)
        if fourier_pts is not None and fourier_score > best_score:
            best_score = fourier_score
            best_pts = fourier_pts
    
    # Phase 3: Fine optimization
    if best_pts is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 15.0:
            result, score = _optimize_start(best_pts, 0.01, 0.001, 200, remaining - 10.0)
            if score > best_score:
                best_score = score
                best_pts = result
    
    # Phase 4: Local refinement
    if best_pts is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 10.0:
            best_pts, best_score = _local_refinement(
                best_pts, rng, n_trials=10, time_budget=remaining - 5.0
            )
    
    # Fallback
    if best_pts is None:
        return _regular_polygon(N)
    
    return best_pts

# EVOLVE-BLOCK-END
