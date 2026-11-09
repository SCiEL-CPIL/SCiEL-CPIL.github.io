# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _fast_min_vertex_score(angles):
    """
    Fast scoring: given 20 angles (radians, sorted, in [0, 2pi)),
    compute the minimum vertex score. Returns 0 if invalid.
    Vertices are on unit circle (radius 1), then we normalize.
    """
    n = len(angles)
    # Ensure angles are sorted and in [0, 2pi)
    angles = np.sort(angles)
    # Wrap to [0, 2pi)
    angles = angles % (2 * np.pi)
    angles = np.sort(angles)
    
    x = np.cos(angles)
    y = np.sin(angles)
    pts = np.stack([x, y], axis=1)
    
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_d = dists_to_centroid.mean()
    if mean_d <= 0.01:
        return 0.0
    centered = centered / mean_d
    
    # Check for repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.sum((centered[i] - centered[j]) ** 2) < 1e-12:
                return 0.0
    
    # Check convexity via cross products
    cross_products = np.zeros(n)
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i + 1) % n]
        p2 = centered[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    
    # All must be nonzero and same sign
    if np.any(np.abs(cross_products) < 1e-12):
        return 0.0
    if np.any(cross_products > 0) and np.any(cross_products < 0):
        return 0.0
    
    # Check total turning angle
    angles_turn = np.zeros(n)
    for i in range(n):
        p0 = centered[i]
        p1 = centered[(i + 1) % n]
        p2 = centered[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        n1 = np.sqrt(np.sum(e1 ** 2))
        n2 = np.sqrt(np.sum(e2 ** 2))
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        cos_a = np.dot(e1, e2) / (n1 * n2)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        a = np.arccos(cos_a)
        if cross_products[i] < 0:
            a = -a
        angles_turn[i] = a
    
    total_turn = np.sum(angles_turn)
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return 0.0
    
    # Compute all pairwise distances (n x n matrix)
    diff = centered[:, np.newaxis, :] - centered[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    
    min_score = 1.0
    for i in range(n):
        d = dist_matrix[i]
        d_sorted = np.sort(d)[:19]
        
        if d_sorted[0] < 0.05:
            return 0.0
        
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k + 1]) / 2.0
        if D < 0.1:
            return 0.0
        
        if k == 0:
            four_d = d_sorted[0:4]
        elif k == 17:
            four_d = d_sorted[15:19]
        else:
            four_d = d_sorted[k - 1:k + 3]
        
        ratios = np.maximum(four_d / D, D / four_d)
        raw = np.mean(ratios)
        
        if raw <= 0:
            return 0.0
        vertex_score = min(1.0, 1.0 / raw)
        min_score = min(min_score, vertex_score)
    
    return min_score


def _objective(angles):
    """Objective for minimization: negative of the min vertex score."""
    score = _fast_min_vertex_score(angles)
    return -score


def equidistant_points_polygon20() -> np.ndarray:
    """
    Use Nelder-Mead optimization on angular parameters to find a convex 20-gon
    that maximizes the minimum vertex equidistance score.
    """
    start_time = time.time()
    deadline = start_time + 610.0  # safety margin
    
    n = 20
    best_score = -1.0
    best_angles = None
    
    # Regular polygon as starting point
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Multiple starting points for Nelder-Mead
    rng = np.random.default_rng(123)
    
    starting_points = []
    # Regular polygon
    starting_points.append(base_angles.copy())
    # Slightly perturbed regular polygons
    for _ in range(10):
        perturbed = base_angles + rng.normal(0, 0.05, n)
        perturbed = np.sort(perturbed % (2 * np.pi))
        starting_points.append(perturbed)
    # More diverse starts
    for _ in range(5):
        random_angles = np.sort(rng.uniform(0, 2 * np.pi, n))
        starting_points.append(random_angles)
    
    nelder_mead_options = {
        'maxiter': 500,
        'maxfev': 500,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True,
    }
    
    for sp_idx, start_angles in enumerate(starting_points):
        if time.time() > deadline - 10.0:
            break
        
        # Evaluate starting point
        score = _fast_min_vertex_score(start_angles)
        if score > best_score:
            best_score = score
            best_angles = start_angles.copy()
        
        # Run Nelder-Mead
        try:
            result = minimize(
                _objective,
                start_angles,
                method='Nelder-Mead',
                options=nelder_mead_options
            )
        except Exception:
            continue
        
        opt_angles = np.sort(result.x % (2 * np.pi))
        opt_score = _fast_min_vertex_score(opt_angles)
        if opt_score > best_score:
            best_score = opt_score
            best_angles = opt_angles.copy()
    
    # Additional refinement: try smaller perturbations around best
    if best_angles is not None and time.time() < deadline - 10.0:
        for trial in range(20):
            if time.time() > deadline - 5.0:
                break
            perturbation = best_angles + rng.normal(0, 0.01, n)
            perturbation = np.sort(perturbation % (2 * np.pi))
            score = _fast_min_vertex_score(perturbation)
            if score > best_score:
                best_score = score
                best_angles = perturbation.copy()
    
    # Convert best angles to polygon
    if best_angles is None:
        best_angles = base_angles
    
    x = np.cos(best_angles)
    y = np.sin(best_angles)
    polygon = np.stack([x, y], axis=1)
    
    return polygon


# EVOLVE-BLOCK-END
