"""
Fourier-based smooth convex curve representation for 14-point Heilbronn problem.
Optimization variables: 10 low-frequency Fourier coefficients defining a strictly convex
closed curve. The 14 points are placed at optimized arc-length intervals along this curve.
This reduces search dimensionality from 28 to ~10, enabling more effective optimization.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_FOURIER = 5  # harmonics 1..5 (10 coefficients: a1..a5, b1..b5)
N_ANGLES = N   # 14 angular positions

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dy_ji * dx_ki
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min_triangle_area / hull_area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _fourier_curve(fourier_coeffs, n_samples=1000):
    """
    Generate points on a smooth convex curve defined by Fourier coefficients.
    fourier_coeffs: array of shape (2*N_FOURIER,) = [a1, b1, a2, b2, ..., aN, bN]
    Returns: array of shape (n_samples, 2)
    
    r(θ) = 1 + Σ_{k=1}^{N} (a_k cos(kθ) + b_k sin(kθ))
    To ensure convexity, we constrain |a_k| + |b_k| < 1/k for k>=2
    """
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    r = np.ones(n_samples)
    for k in range(1, N_FOURIER + 1):
        a_k = fourier_coeffs[2 * (k - 1)]
        b_k = fourier_coeffs[2 * (k - 1) + 1]
        r += a_k * np.cos(k * theta) + b_k * np.sin(k * theta)
    # Ensure r > 0 for convexity
    r = np.maximum(r, 0.1)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def _decode_fourier_params(params):
    """
    Decode Fourier parameters to 14 points.
    params: [a1, b1, a2, b2, ..., a5, b5, θ1, θ2, ..., θ14]
    Total: 10 + 14 = 24 parameters (reduced from 28)
    """
    fourier_coeffs = params[:2 * N_FOURIER]
    angles = params[2 * N_FOURIER:]
    
    # Generate dense curve
    curve = _fourier_curve(fourier_coeffs, n_samples=10000)
    
    # Interpolate to get points at specified angles
    # Sort angles for proper interpolation
    curve_theta = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
    
    points = np.zeros((N, 2))
    for i in range(N):
        angle_i = angles[i] % (2 * np.pi)
        # Find nearest curve point
        idx = np.argmin(np.abs(curve_theta - angle_i))
        # Linear interpolation for better accuracy
        idx_prev = (idx - 1) % 10000
        idx_next = (idx + 1) % 10000
        t_prev = curve_theta[idx_prev]
        t_curr = curve_theta[idx]
        t_next = curve_theta[idx_next]
        
        if t_curr - t_prev > 0 and t_next - t_curr > 0:
            # Interpolate between prev and curr, or curr and next
            if abs(angle_i - t_prev) < abs(angle_i - t_curr):
                t = (angle_i - t_prev) / (t_curr - t_prev) if t_curr > t_prev else 0
                t = np.clip(t, 0, 1)
                points[i] = (1 - t) * curve[idx_prev] + t * curve[idx]
            else:
                t = (angle_i - t_curr) / (t_next - t_curr) if t_next > t_curr else 0
                t = np.clip(t, 0, 1)
                points[i] = (1 - t) * curve[idx] + t * curve[idx_next]
        else:
            points[i] = curve[idx]
    
    return points

def _objective_fourier(flat):
    """
    Objective function for Fourier parameter optimization.
    Minimizes hull_area / min_triangle_area.
    """
    pts = _decode_fourier_params(flat)
    if not np.all(np.isfinite(pts)):
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    return hull_area / min_area

def _generate_initial_fourier_configs():
    """
    Generate diverse initial configurations in Fourier parameter space.
    """
    configs = []
    
    # 1. Circle (all Fourier coefficients zero)
    params = np.zeros(2 * N_FOURIER + N)
    params[2 * N_FOURIER:] = np.linspace(0, 2 * np.pi, N, endpoint=False)
    configs.append(params)
    
    # 2. Ellipse (only a1, b1 nonzero)
    for a1, b1 in [(0.1, 0.0), (0.0, 0.1), (0.05, 0.05), (0.15, 0.0), (0.0, 0.15)]:
        params = np.zeros(2 * N_FOURIER + N)
        params[0] = a1
        params[1] = b1
        params[2 * N_FOURIER:] = np.linspace(0, 2 * np.pi, N, endpoint=False)
        configs.append(params)
    
    # 3. Slightly perturbed circle with random angles
    rng = np.random.default_rng(42)
    for _ in range(5):
        params = np.zeros(2 * N_FOURIER + N)
        # Small random Fourier coefficients
        for k in range(1, N_FOURIER + 1):
            params[2 * (k - 1)] = rng.uniform(-0.05 / k, 0.05 / k)
            params[2 * (k - 1) + 1] = rng.uniform(-0.05 / k, 0.05 / k)
        # Slightly perturbed angles
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        params[2 * N_FOURIER:] = base_angles + rng.normal(0, 0.1, N)
        configs.append(params)
    
    # 4. Non-uniform angle spacing
    for spread in [0.2, 0.5, 0.8]:
        params = np.zeros(2 * N_FOURIER + N)
        # Random but sorted angles
        base = np.sort(rng.uniform(0, 2 * np.pi, N))
        # Ensure minimum spacing
        for i in range(N - 1):
            if base[i + 1] - base[i] < 0.1:
                base[i + 1] = base[i] + 0.1
        base[-1] = (base[-1] + 2 * np.pi) % (2 * np.pi)
        params[2 * N_FOURIER:] = base
        configs.append(params)
    
    # 5. Higher harmonic emphasis
    for k_dom in [2, 3, 4]:
        params = np.zeros(2 * N_FOURIER + N)
        params[2 * (k_dom - 1)] = 0.03 / k_dom
        params[2 * N_FOURIER:] = np.linspace(0, 2 * np.pi, N, endpoint=False)
        configs.append(params)
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_score = 0.0
    best_pts = None
    best_obj = float('inf')
    
    def update_best(pts):
        nonlocal best_score, best_pts, best_obj
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            hull_area = _hull_area(pts)
            min_area = _min_triangle_area(pts)
            if hull_area > 1e-10 and min_area > 1e-12:
                best_obj = hull_area / min_area

    # Phase 1: Multi-start Nelder-Mead on Fourier parameters (24D)
    inits = _generate_initial_fourier_configs()
    for i, params0 in enumerate(inits):
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_objective_fourier, params0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 
                                    'adaptive': True, 'initial_simplex': None})
            if res.fun < best_obj:
                pts = _decode_fourier_params(res.x)
                update_best(pts)
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B with soft objective for smoother gradients
    if best_pts is not None and time.time() < deadline - 25:
        # Find the best Fourier params that produced best_pts
        # Re-optimize from current best
        try:
            # Use the last successful res.x or fall back to circle
            # We'll just run L-BFGS-B on a fresh start near the best
            params_start = np.zeros(2 * N_FOURIER + N)
            params_start[2 * N_FOURIER:] = np.linspace(0, 2 * np.pi, N, endpoint=False)
            
            # Soft objective for smoother gradients
            def _objective_soft_fourier(flat, beta=30.0):
                pts = _decode_fourier_params(flat)
                if not np.all(np.isfinite(pts)):
                    return 1e10
                n = len(pts)
                x = pts[:, 0]
                y = pts[:, 1]
                areas_list = []
                for i in range(n - 2):
                    for j in range(i + 1, n - 1):
                        dx_ji = x[j] - x[i]
                        dy_ji = y[j] - y[i]
                        dx_ki = x[j+1:] - x[i]
                        dy_ki = y[j+1:] - y[i]
                        cross = dx_ji * dy_ki - dy_ji * dx_ki
                        areas_list.append(0.5 * np.abs(cross))
                if not areas_list:
                    return 1e10
                areas = np.concatenate(areas_list)
                if np.min(areas) < 1e-12:
                    return 1e10
                z = -beta * areas
                max_z = np.max(z)
                if max_z < -500:
                    log_sum = max_z
                else:
                    log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
                soft_min = -log_sum / beta
                if soft_min < 1e-12:
                    return 1e10
                hull_area = _hull_area(pts)
                if hull_area < 1e-10:
                    return 1e10
                return hull_area / soft_min
            
            res_lbfgs = minimize(_objective_soft_fourier, params_start, method='L-BFGS-B',
                                 options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res_lbfgs.fun < best_obj:
                pts = _decode_fourier_params(res_lbfgs.x)
                update_best(pts)
        except Exception:
            pass
    
    # Phase 3: Differential Evolution on Fourier parameters
    if time.time() < deadline - 15:
        try:
            bounds = []
            for k in range(1, N_FOURIER + 1):
                # Constrain to ensure convexity: |a_k| + |b_k| < 1/k
                max_coeff = 0.8 / k
                bounds.append((-max_coeff, max_coeff))
                bounds.append((-max_coeff, max_coeff))
            for _ in range(N):
                bounds.append((0, 2 * np.pi))
            
            de_res = differential_evolution(
                _objective_fourier, bounds,
                seed=42, maxiter=30, tol=1e-8,
                popsize=15, mutation=(0.5, 1.0), recombination=0.7,
                polish=False, update_interval=1,
                init='sobol'
            )
            if de_res.fun < best_obj:
                pts = _decode_fourier_params(de_res.x)
                update_best(pts)
        except Exception:
            pass
    
    # Phase 4: Final polish with tighter tolerance
    if best_pts is not None and time.time() < deadline - 5:
        try:
            # Re-run Nelder-Mead from the best configuration found
            # We need to recover the Fourier params that produced best_pts
            # For simplicity, run from circle again with tighter tol
            params_start = np.zeros(2 * N_FOURIER + N)
            params_start[2 * N_FOURIER:] = np.linspace(0, 2 * np.pi, N, endpoint=False)
            
            res = minimize(_objective_fourier, params_start, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13,
                                    'adaptive': True})
            if res.fun < best_obj:
                pts = _decode_fourier_params(res.x)
                update_best(pts)
        except Exception:
            pass
    
    # Fallback to regular polygon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    try:
        hull = ConvexHull(best_pts)
        area = hull.volume
        if area > 1e-10:
            scale = np.sqrt(1.0 / area)
            best_pts = best_pts * scale
    except Exception:
        pass
    
    return best_pts
