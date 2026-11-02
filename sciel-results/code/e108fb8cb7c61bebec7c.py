"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _min_triangle_area_from_angles(angles):
    """
    Compute minimum triangle area for points on unit circle at given angles.
    Points are at (cos(angle), sin(angle)).
    Area of triangle with angles a, b, c (sorted) is:
      0.5 * |sin(b-a) + sin(c-b) + sin(a-c)|
    But since they're on unit circle, the area is 0.5 * |sin(b-a)*sin(c-a)*sin(c-b)|... 
    Actually, for three points on unit circle at angles a < b < c, the triangle area is:
      0.5 * |sin(b-a) + sin(c-b) + sin(a-(c))|  -- no, let me use the standard formula.
    
    For points P1=(cos a, sin a), P2=(cos b, sin b), P3=(cos c, sin c):
    Area = 0.5 * |det([P2-P1, P3-P1])|
    = 0.5 * |(cos b - cos a)(sin c - sin a) - (cos c - cos a)(sin b - sin a)|
    
    Using trig identities, this simplifies to:
    0.5 * |sin(b-a) + sin(c-b) - sin(c-a)|  ... let me just compute it numerically.
    """
    n = len(angles)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    x = pts[:, 0]
    y = pts[:, 1]
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _score_from_angles(angles):
    """
    Score for points on unit circle: min triangle area / convex hull area.
    For points on unit circle, the convex hull is the polygon connecting them in order.
    The area of this polygon can be computed.
    """
    n = len(angles)
    # Sort angles to get convex hull order
    sorted_angles = np.sort(angles)
    pts = np.column_stack([np.cos(sorted_angles), np.sin(sorted_angles)])
    
    # Convex hull area for points on unit circle in angular order
    # Use shoelace formula
    x = pts[:, 0]
    y = pts[:, 1]
    # Shoelace formula
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
    
    if area < 1e-10:
        return 0.0
    
    min_area = _min_triangle_area_from_angles(angles)
    return min_area / area

def _objective_angles(angles):
    """
    Minimize 1/score = hull_area / min_triangle_area for points on unit circle.
    """
    n = len(angles)
    sorted_angles = np.sort(angles)
    pts = np.column_stack([np.cos(sorted_angles), np.sin(sorted_angles)])
    
    # Check for near-duplicate angles
    diffs = np.diff(sorted_angles)
    diffs = np.append(diffs, sorted_angles[0] + 2*np.pi - sorted_angles[-1])
    if np.min(diffs) < 1e-8:
        return 1e10
    
    # Convex hull area via shoelace
    x = pts[:, 0]
    y = pts[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    hull_area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
    
    if hull_area < 1e-10:
        return 1e10
    
    # Min triangle area
    min_area = _min_triangle_area_from_angles(angles)
    
    if min_area < 1e-12:
        return 1e10
    
    return hull_area / min_area

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
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
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area_fast(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Minimum triangle area divided by convex hull area."""
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _normalize_unit_area(points):
    """Scale points so convex hull area = 1."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points

def construct():
    """
    Generate 14 points to maximize minimum triangle area / convex hull area.
    
    Strategy: Use angular parameterization for convex position.
    Points are placed on a unit circle at angles theta_1 < theta_2 < ... < theta_14.
    This guarantees convex position and reduces the problem to 14 angular variables.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    best_angles = None
    
    # --- Phase 1: Optimize angular parameters using Nelder-Mead ---
    
    # Initial guess: equally spaced angles
    base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Multiple restarts with different perturbations
    rng = np.random.default_rng(12345)
    
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            x0 = base_angles.copy()
        else:
            # Random perturbation of equally spaced angles
            perturbation = rng.normal(0, 0.1 + 0.05*restart, size=N)
            x0 = base_angles + perturbation
            # Normalize to [0, 2*pi)
            x0 = np.sort(x0 % (2*np.pi))
        
        # Run Nelder-Mead on the angular objective
        try:
            res = minimize(
                _objective_angles,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if res.fun < 1e9:  # Valid result
                angles_opt = np.sort(res.x % (2*np.pi))
                score_val = _score_from_angles(angles_opt)
                
                if score_val > best_score:
                    best_score = score_val
                    best_angles = angles_opt.copy()
                    
        except Exception:
            pass
    
    # --- Phase 2: Powell polish on best angles ---
    if best_angles is not None and time.time() < deadline - 15:
        try:
            res_powell = minimize(
                _objective_angles,
                best_angles,
                method='Powell',
                options={
                    'maxiter': 5000,
                    'xtol': 1e-12,
                    'ftol': 1e-14
                }
            )
            if res_powell.fun < 1e9:
                angles_opt = np.sort(res_powell.x % (2*np.pi))
                score_val = _score_from_angles(angles_opt)
                if score_val > best_score:
                    best_score = score_val
                    best_angles = angles_opt.copy()
        except Exception:
            pass
    
    # --- Phase 3: Final NM polish ---
    if best_angles is not None and time.time() < deadline - 10:
        try:
            res_final = minimize(
                _objective_angles,
                best_angles,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if res_final.fun < 1e9:
                angles_opt = np.sort(res_final.x % (2*np.pi))
                score_val = _score_from_angles(angles_opt)
                if score_val > best_score:
                    best_score = score_val
                    best_angles = angles_opt.copy()
        except Exception:
            pass
    
    # --- Convert angles to points ---
    if best_angles is None:
        # Fallback: regular 14-gon
        best_angles = base_angles.copy()
    
    final_points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    
    # Normalize to unit area
    final_points = _normalize_unit_area(final_points)
    
    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            final_points = _normalize_unit_area(
                np.column_stack([np.cos(base_angles), np.sin(base_angles)])
            )
    except Exception:
        final_points = _normalize_unit_area(
            np.column_stack([np.cos(base_angles), np.sin(base_angles)])
        )
    
    return final_points
# EVOLVE-BLOCK-END
