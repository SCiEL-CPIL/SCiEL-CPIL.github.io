"""Heilbronn convex regions: place 13 points to maximize minimum triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13
TIME_LIMIT = 110.0  # seconds, safety margin before 120s kill

# Pre-compute triangle indices for N=13
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_I_IDX = _TRI_INDICES[:, 0]
_J_IDX = _TRI_INDICES[:, 1]
_K_IDX = _TRI_INDICES[:, 2]
_NUM_TRI = len(_TRI_INDICES)


def _get_triangle_areas(points):
    """
    Vectorized calculation of areas for all triples.
    points: (N, 2) array
    returns: array of shape (num_triangles,)
    """
    pi = points[_I_IDX]
    pj = points[_J_IDX]
    pk = points[_K_IDX]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas


def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area


def _softmin_score(points, alpha):
    """
    Smooth approximation of the score using softmin of triangle areas.
    This provides a differentiable-like landscape for optimization.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    
    # Softmin: -1/alpha * log(sum(exp(-alpha * areas)))
    a_min = np.min(areas)
    exp_terms = np.exp(-alpha * (areas - a_min))
    log_sum = np.log(np.sum(exp_terms))
    softmin_val = a_min - np.log(np.sum(exp_terms)) / alpha
    
    return softmin_val / hull_area


def _neg_softmin_score(params, alpha):
    """Objective for softmin-based optimization."""
    points = params.reshape(N, 2)
    s = _softmin_score(points, alpha)
    return -s


def _neg_score(params):
    """Objective function for minimization (maximizing score)."""
    points = params.reshape(N, 2)
    s = _score(points)
    return -s


def _fourier_to_points(params):
    """
    Convert Fourier parameters to 13 points on a perturbed regular polygon.
    params: [r0, a1, b1, a2, b2, a3, b3, cx, cy] (9 params)
    - r0: base radius
    - a1, b1: first harmonic amplitudes
    - a2, b2: second harmonic amplitudes
    - a3, b3: third harmonic amplitudes
    - cx, cy: center offset
    """
    r0 = params[0]
    a1, b1 = params[1], params[2]
    a2, b2 = params[3], params[4]
    a3, b3 = params[5], params[6]
    cx, cy = params[7], params[8]
    
    angles = 2 * np.pi * np.arange(N) / N
    # Perturbed radius
    radii = (r0 + a1 * np.cos(angles) + b1 * np.sin(angles) + 
             a2 * np.cos(2 * angles) + b2 * np.sin(2 * angles) +
             a3 * np.cos(3 * angles) + b3 * np.sin(3 * angles))
    
    # Ensure positive radii
    radii = np.maximum(radii, 0.1)
    
    points = np.column_stack([radii * np.cos(angles) + cx, radii * np.sin(angles) + cy])
    return points


def _neg_fourier_score(params, alpha):
    """
    Objective for Fourier-parameterized optimization.
    Uses softmin for smoothness.
    """
    points = _fourier_to_points(params)
    s = _softmin_score(points, alpha)
    return -s


def _neg_fourier_score_hard(params):
    """
    Objective for Fourier-parameterized optimization (hard score).
    """
    points = _fourier_to_points(params)
    s = _score(points)
    return -s


def _optimize_fourier(deadline, rng):
    """
    Optimize using Fourier parameterization.
    This reduces dimensionality from 26 to 9 and enforces smooth convexity.
    """
    best_score = 0.0
    best_params = None
    best_points = None
    
    # Initial parameter guesses
    initial_params_list = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Regular polygon
        [1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.9, 0.03, 0.02, 0.02, 0.01, 0.01, 0.0, 0.0, 0.0],
        [1.1, -0.02, 0.03, 0.01, -0.02, 0.0, 0.01, 0.0, 0.0],
    ]
    
    # Add random initializations
    for _ in range(8):
        p = [1.0 + rng.normal(0, 0.1)]
        for _ in range(6):
            p.append(rng.normal(0, 0.05))
        p.extend([rng.normal(0, 0.05), rng.normal(0, 0.05)])
        initial_params_list.append(p)
    
    for init_params in initial_params_list:
        if time.time() > deadline - 15:
            break
        
        init_params = np.array(init_params, dtype=float)
        
        # Multi-stage optimization in Fourier space
        current_params = init_params.copy()
        current_score = _score(_fourier_to_points(current_params))
        
        # Stage 1: Softmin with various alphas using Nelder-Mead (handles non-smooth better)
        for alpha in [50, 20, 10, 5]:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    _neg_fourier_score,
                    current_params,
                    args=(alpha,),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                )
                cand_params = res.x
                cand_score = _score(_fourier_to_points(cand_params))
                if cand_score > current_score:
                    current_score = cand_score
                    current_params = cand_params
            except Exception:
                pass
        
        # Stage 2: L-BFGS-B polish
        if time.time() < deadline - 8:
            try:
                res = minimize(
                    _neg_fourier_score,
                    current_params,
                    args=(5.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                cand_params = res.x
                cand_score = _score(_fourier_to_points(cand_params))
                if cand_score > current_score:
                    current_score = cand_score
                    current_params = cand_params
            except Exception:
                pass
        
        # Stage 3: Hard score Nelder-Mead
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    _neg_fourier_score_hard,
                    current_params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                cand_params = res.x
                cand_score = _score(_fourier_to_points(cand_params))
                if cand_score > current_score:
                    current_score = cand_score
                    current_params = cand_params
            except Exception:
                pass
        
        # Update global best
        if current_score > best_score:
            best_score = current_score
            best_params = current_params.copy()
            best_points = _fourier_to_points(current_params)
    
    return best_points, best_score


def _optimize_full_space(deadline, rng):
    """
    Optimize in full 26-dimensional space using Nelder-Mead and Powell.
    """
    best_score = 0.0
    best_points = None
    
    # Generate diverse starting points
    start_configs = []
    
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    start_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Random perturbations of regular polygon
    for _ in range(10):
        noise_scale = rng.uniform(0.1, 0.4)
        radii = 1.0 + noise_scale * rng.randn(N)
        angle_offsets = noise_scale * 0.5 * rng.randn(N)
        pts = np.column_stack([
            radii * np.cos(angles + angle_offsets),
            radii * np.sin(angles + angle_offsets)
        ])
        start_configs.append(pts)
    
    # Ring configurations
    for _ in range(5):
        # Two rings: outer and inner
        outer_n = rng.randint(5, 9)
        inner_n = N - outer_n
        outer_angles = 2 * np.pi * np.arange(outer_n) / outer_n + rng.uniform(0, 2*np.pi)
        inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + rng.uniform(0, 2*np.pi)
        outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        inner_radius = rng.uniform(0.3, 0.6)
        inner_pts = inner_radius * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
        pts = np.vstack([outer_pts, inner_pts])
        start_configs.append(pts)
    
    for start_pts in start_configs:
        if time.time() > deadline - 10:
            break
        
        x0 = start_pts.flatten()
        current_score = _score(start_pts)
        
        # Stage 1: Nelder-Mead with softmin (smooth landscape)
        for alpha in [30, 10]:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    _neg_softmin_score,
                    x0,
                    args=(alpha,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
                cand_pts = res.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                if cand_score > current_score:
                    current_score = cand_score
                    x0 = res.x
            except Exception:
                pass
        
        # Stage 2: Powell method (derivative-free, good for non-smooth)
        if time.time() < deadline - 3:
            try:
                res = minimize(
                    _neg_score,
                    x0,
                    method='Powell',
                    options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6}
                )
                cand_pts = res.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                if cand_score > current_score:
                    current_score = cand_score
                    x0 = res.x
            except Exception:
                pass
        
        # Update global best
        if current_score > best_score:
            best_score = current_score
            best_points = x0.reshape(N, 2).copy()
    
    return best_points, best_score


def _refine_with_local_search(points, deadline, rng):
    """
    Local refinement using coordinate descent and random perturbations.
    """
    best_points = points.copy()
    best_score = _score(points)
    
    # Coordinate descent: optimize each point one at a time
    for _ in range(3):
        if time.time() > deadline - 2:
            break
        for i in range(N):
            if time.time() > deadline - 1:
                break
            # Try small perturbations of point i
            for _ in range(20):
                perturbed = best_points.copy()
                perturbed[i] += 0.05 * rng.randn(2)
                s = _score(perturbed)
                if s > best_score:
                    best_score = s
                    best_points = perturbed
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Fourier-parameterized optimization (reduced dimensionality, smooth)
    if time.time() < deadline - 60:
        fourier_points, fourier_score = _optimize_fourier(deadline, rng)
        if fourier_points is not None and fourier_score > best_score:
            best_score = fourier_score
            best_points = fourier_points
    
    # Phase 2: Full-space optimization with diverse starts
    if time.time() < deadline - 30:
        full_points, full_score = _optimize_full_space(deadline, rng)
        if full_points is not None and full_score > best_score:
            best_score = full_score
            best_points = full_points
    
    # Phase 3: Local refinement on the best found
    if best_points is not None and time.time() < deadline - 5:
        refined_points, refined_score = _refine_with_local_search(best_points, deadline, rng)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
    
    # Phase 4: One more round of Fourier optimization from the best
    if best_points is not None and time.time() < deadline - 10:
        # Try to fit the best points to a Fourier representation
        angles = 2 * np.pi * np.arange(N) / N
        # Estimate Fourier coefficients by least squares
        radii = np.linalg.norm(best_points, axis=1)
        A = np.column_stack([
            np.ones(N),
            np.cos(angles), np.sin(angles),
            np.cos(2*angles), np.sin(2*angles),
            np.cos(3*angles), np.sin(3*angles)
        ])
        # Solve for coefficients (ignore cx, cy for simplicity)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, radii, rcond=None)
            cx, cy = np.mean(best_points, axis=0)
            fourier_params = np.concatenate([coeffs, [cx, cy]])
            
            # Optimize from this Fourier representation
            current_params = fourier_params.copy()
            current_score = _score(_fourier_to_points(current_params))
            
            for alpha in [20, 10, 5]:
                if time.time() > deadline - 3:
                    break
                try:
                    res = minimize(
                        _neg_fourier_score,
                        current_params,
                        args=(alpha,),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                    )
                    cand_params = res.x
                    cand_score = _score(_fourier_to_points(cand_params))
                    if cand_score > current_score:
                        current_score = cand_score
                        current_params = cand_params
                except Exception:
                    pass
            
            if current_score > best_score:
                best_score = current_score
                best_points = _fourier_to_points(current_params)
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
