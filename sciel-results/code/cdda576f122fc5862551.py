"""Combined approach: Fourier-parameterized convex boundary + multi-method local refinement."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j + 1:] - x[i]
            dy_ki = y[j + 1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _hull_area_fast(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j + 1:] - x[i]
            dy_ki = y[j + 1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)


def _objective_hard(points_flat):
    """Minimize hull_area / min_triangle_area (scale-invariant)."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    min_area = _min_triangle_area(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


def _objective_soft(points_flat, beta=100.0):
    """Smooth soft-min objective: hull_area / soft_min(triangle areas)."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val


def _fourier_to_points(coeffs):
    """
    Convert Fourier coefficients to points on a convex curve.
    coeffs: array of shape (n_coeffs, 2) where n_coeffs is the number of harmonics.
    We use a parametric representation:
        r(theta) = R0 + sum_{k=1}^{K} (a_k * cos(k*theta) + b_k * sin(k*theta))
    where R0 is the base radius and K is the number of harmonics.
    The points are sampled at N equally spaced angles.
    """
    n_coeffs = len(coeffs)
    # coeffs[0] is [R0, 0], coeffs[1:] are [a_k, b_k] for k=1..K
    R0 = coeffs[0, 0]
    K = n_coeffs - 1
    
    # Sample at N equally spaced angles
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Compute radius at each theta
    radii = np.full(N, R0)
    for k in range(1, K + 1):
        a_k = coeffs[k, 0]
        b_k = coeffs[k, 1]
        radii += a_k * np.cos(k * thetas) + b_k * np.sin(k * thetas)
    
    # Ensure positive radii
    radii = np.maximum(radii, 0.1)
    
    points = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
    return points


def _points_to_fourier(points, n_harmonics=4):
    """
    Convert points to Fourier coefficients (least squares fit).
    This is an approximation; the optimal configuration may not be exactly Fourier.
    """
    # Center the points
    center = np.mean(points, axis=0)
    centered = points - center
    
    # Compute radii and angles
    radii = np.sqrt(np.sum(centered**2, axis=1))
    thetas = np.arctan2(centered[:, 1], centered[:, 0])
    
    # Sort by angle
    order = np.argsort(thetas)
    radii = radii[order]
    thetas = thetas[order]
    
    # Fit Fourier series
    K = n_harmonics
    # Build design matrix
    A = np.zeros((N, K + 1))
    A[:, 0] = 1.0  # R0 term
    for k in range(1, K + 1):
        A[:, k] = np.cos(k * thetas)
    
    # Solve least squares for cos coefficients
    cos_coeffs, _, _, _ = np.linalg.lstsq(A, radii, rcond=None)
    
    # For sin coefficients, we need another fit
    B = np.zeros((N, K))
    for k in range(1, K + 1):
        B[:, k - 1] = np.sin(k * thetas)
    
    # Residual after cos fit
    residual = radii - A @ cos_coeffs
    sin_coeffs, _, _, _ = np.linalg.lstsq(B, residual, rcond=None)
    
    # Build coeffs array
    coeffs = np.zeros((K + 1, 2))
    coeffs[0, 0] = cos_coeffs[0]  # R0
    for k in range(1, K + 1):
        coeffs[k, 0] = cos_coeffs[k]
        coeffs[k, 1] = sin_coeffs[k - 1]
    
    return coeffs


def _fourier_objective(coeffs_flat, n_harmonics=4):
    """Objective for Fourier parameterization: minimize hull_area / min_triangle_area."""
    coeffs = coeffs_flat.reshape(-1, 2)
    points = _fourier_to_points(coeffs)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    min_area = _min_triangle_area(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


def _fourier_objective_soft(coeffs_flat, n_harmonics=4, beta=100.0):
    """Smooth soft-min objective for Fourier parameterization."""
    coeffs = coeffs_flat.reshape(-1, 2)
    points = _fourier_to_points(coeffs)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val


def _coordinate_descent(pts, deadline, max_rounds=30):
    """Cyclic coordinate descent on the hard objective."""
    pts = pts.copy()
    n = len(pts)
    step0 = 0.05
    for rnd in range(max_rounds):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for coord in range(2):
                orig = pts[i, coord]
                step = step0
                for _ in range(20):
                    pts[i, coord] = orig + step
                    s_plus = _score(pts)
                    pts[i, coord] = orig - step
                    s_minus = _score(pts)
                    pts[i, coord] = orig
                    if s_plus > s_minus:
                        pts[i, coord] = orig + step
                    elif s_minus > s_plus:
                        pts[i, coord] = orig - step
                    else:
                        break
                    improved = True
                    step *= 0.5
                    if step < 1e-8:
                        break
        if not improved:
            break
    return pts


def _nm(x0, deadline, max_iter=1500, obj=_objective_hard):
    remaining = deadline - time.time()
    if remaining < 2.0:
        return None
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None


def _lbfgs(x0, deadline, max_iter=3000, obj=_objective_soft):
    remaining = deadline - time.time()
    if remaining < 5.0:
        return None
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res
    except Exception:
        return None


def _powell(x0, deadline, max_iter=1000, obj=_objective_hard):
    remaining = deadline - time.time()
    if remaining < 3.0:
        return None
    try:
        res = minimize(obj, x0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-11, 'ftol': 1e-13})
        return res
    except Exception:
        return None


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Initial good starting point: regular 14-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    # Normalize to unit area
    init_hull_area = _hull_area_fast(init_points)
    if init_hull_area > 1e-12:
        init_points = init_points / np.sqrt(init_hull_area)

    best_points = init_points.copy()
    best_score = _score(init_points)

    # Try multiple initializations
    inits = [init_points.copy()]
    
    # Rotated regular polygons
    for rot in [0.2, 0.5, 1.0, 1.5, 2.0]:
        a = angles + rot
        pts = np.column_stack([np.cos(a), np.sin(a)])
        ha = _hull_area_fast(pts)
        if ha > 1e-12:
            pts = pts / np.sqrt(ha)
        inits.append(pts)
    
    # Two concentric rings (7 inner, 7 outer)
    for r_inner in [0.4, 0.6, 0.8]:
        a_in = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        pts_in = r_inner * np.column_stack([np.cos(a_in), np.sin(a_in)])
        pts_out = np.column_stack([np.cos(a_out), np.sin(a_out)])
        pts = np.vstack([pts_in, pts_out])
        ha = _hull_area_fast(pts)
        if ha > 1e-12:
            pts = pts / np.sqrt(ha)
        inits.append(pts)

    # For each initialization, try multiple optimization strategies
    for init_pts in inits:
        if time.time() > deadline - 20.0:
            break
        
        # 1. Fourier parameterization optimization
        if time.time() < deadline - 15.0:
            try:
                coeffs = _points_to_fourier(init_pts, n_harmonics=4)
                coeffs_flat = coeffs.ravel()
                
                # L-BFGS on Fourier space
                result = _lbfgs(coeffs_flat, deadline, max_iter=2000, obj=_fourier_objective_soft)
                if result is not None:
                    candidate_coeffs = result.x.reshape(-1, 2)
                    candidate_points = _fourier_to_points(candidate_coeffs)
                    candidate_score = _score(candidate_points)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_points = candidate_points.copy()
            except Exception:
                pass
        
        # 2. Cartesian L-BFGS
        if time.time() < deadline - 15.0:
            try:
                x0 = init_pts.ravel()
                result = _lbfgs(x0, deadline, max_iter=2000, obj=_objective_soft)
                if result is not None:
                    candidate_points = result.x.reshape(-1, 2)
                    candidate_score = _score(candidate_points)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_points = candidate_points.copy()
            except Exception:
                pass
        
        # 3. Nelder-Mead on hard objective
        if time.time() < deadline - 10.0:
            try:
                x0 = init_pts.ravel()
                result = _nm(x0, deadline, max_iter=1000, obj=_objective_hard)
                if result is not None:
                    candidate_points = result.x.reshape(-1, 2)
                    candidate_score = _score(candidate_points)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_points = candidate_points.copy()
            except Exception:
                pass
        
        # 4. Powell method
        if time.time() < deadline - 10.0:
            try:
                x0 = init_pts.ravel()
                result = _powell(x0, deadline, max_iter=500, obj=_objective_hard)
                if result is not None:
                    candidate_points = result.x.reshape(-1, 2)
                    candidate_score = _score(candidate_points)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_points = candidate_points.copy()
            except Exception:
                pass

    # Final refinement on the best result
    if time.time() < deadline - 5.0:
        try:
            x0 = best_points.ravel()
            result = _nm(x0, deadline, max_iter=1000, obj=_objective_hard)
            if result is not None:
                candidate_points = result.x.reshape(-1, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
        except Exception:
            pass
    
    # Coordinate descent refinement
    if time.time() < deadline - 5.0:
        try:
            refined = _coordinate_descent(best_points, deadline, max_rounds=20)
            refined_score = _score(refined)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined.copy()
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
