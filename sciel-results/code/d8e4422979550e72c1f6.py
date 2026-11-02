"""Convex-position parameterization for Heilbronn n=14: angles+radii enforce convex order."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _build_points(angles, radii):
    """Build points from angular order and radii (ensures convex position)."""
    cx = np.cos(angles) * radii
    cy = np.sin(angles) * radii
    return np.column_stack([cx, cy])

def _min_tri_area_vec(pts):
    """Vectorized min triangle area for N=14 points."""
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    min_a = float('inf')
    for i in range(n):
        xi, yi = x[i], y[i]
        dx = x[i+1:] - xi
        dy = y[i+1:] - yi
        for j in range(1, n - i):
            dxj, dyj = dx[j-1], dy[j-1]
            if j < n - i:
                cross = dxj * dy[j:] - dx[j:] * dyj
                a = 0.5 * np.abs(cross)
                if len(a) > 0:
                    m = float(np.min(a))
                    if m < min_a:
                        min_a = m
    return min_a

def _score_points(pts):
    """Min triangle area / convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-12:
        return 0.0
    min_a = _min_tri_area_vec(pts)
    if min_a < 1e-14:
        return 0.0
    return min_a / hull_area

def _neg_score_params(theta):
    """Objective: negative score from angular params theta (length N).
    theta[0:N//2] = angle offsets from regular polygon
    theta[N//2:N] = log(radii)
    """
    if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
        return 1e10
    n = N
    half = n // 2
    # Angle offsets (wrap around)
    angle_offsets = theta[:half]
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    # Spread offsets to all n angles via interpolation
    # Use cumulative approach: each offset affects its angle
    full_offsets = np.zeros(n)
    full_offsets[:half] = angle_offsets
    full_offsets[half:] = theta[half:n-half] if n > half else angle_offsets
    # Simpler: just use first n values as offsets directly
    offsets = theta[:n] if len(theta) >= n else np.concatenate([theta, np.zeros(n - len(theta))])
    angles = base_angles + offsets
    angles = np.sort(angles)  # enforce order
    # Wrap to [0, 2pi)
    angles = angles - angles[0]
    angles[0] = 0.0
    # Radii from second half
    log_r = theta[n:] if len(theta) > n else np.zeros(n - len(theta))
    radii = np.exp(np.clip(log_r, -3, 3))
    pts = _build_points(angles, radii)
    s = _score_points(pts)
    if s < 1e-12:
        return 1e10
    return -s

def _neg_score_flat(pts_flat):
    """Objective on flat 2N-point coordinates."""
    if np.any(np.isnan(pts_flat)) or np.any(np.isinf(pts_flat)):
        return 1e10
    pts = pts_flat.reshape(N, 2)
    s = _score_points(pts)
    if s < 1e-12:
        return 1e10
    return -s

def _regular_n_gon():
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _perturb_angles(radii, seed):
    """Generate perturbed angle/radius configuration."""
    rng = np.random.RandomState(seed)
    base = np.linspace(0, 2*np.pi, N, endpoint=False)
    offsets = rng.normal(0, 0.1, N)
    angles = base + offsets
    angles = np.sort(angles)
    r = radii * np.exp(rng.normal(0, 0.05, N))
    return angles, r

def _score_from_angles_radii(angles, radii):
    pts = _build_points(angles, radii)
    return _score_points(pts)

def _refine_angular(angles, radii, deadline):
    """Nelder-Mead refinement in (angle_offsets, log_radii) space."""
    base = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0 = np.concatenate([angles - base, np.log(radii)])
    
    def obj(theta):
        if time.time() > deadline:
            return 1e10
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            return 1e10
        # Extract angles and radii
        offsets = theta[:N]
        log_r = theta[N:2*N]
        a = base + offsets
        a = np.sort(a)
        a = a - a[0]
        r = np.exp(np.clip(log_r, -3, 3))
        pts = _build_points(a, r)
        s = _score_points(pts)
        if s < 1e-12:
            return 1e10
        return -s
    
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        if res.fun < 1e9:
            offsets = res.x[:N]
            log_r = res.x[N:2*N]
            a = base + offsets
            a = np.sort(a)
            a = a - a[0]
            r = np.exp(np.clip(log_r, -3, 3))
            return a, r, -res.fun
    except Exception:
        pass
    return angles, radii, _score_from_angles_radii(angles, radii)

def _refine_cartesian(pts, deadline):
    """Nelder-Mead refinement in Cartesian space."""
    x0 = pts.flatten().copy()
    def obj(flat):
        if time.time() > deadline:
            return 1e10
        if np.any(np.isnan(flat)) or np.any(np.isinf(flat)):
            return 1e10
        p = flat.reshape(N, 2)
        s = _score_points(p)
        if s < 1e-12:
            return 1e10
        return -s
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        if res.fun < 1e9:
            return res.x.reshape(N, 2), -res.fun
    except Exception:
        pass
    return pts, _score_points(pts)

def _refine_powell(pts, deadline):
    """Powell refinement in Cartesian space."""
    x0 = pts.flatten().copy()
    def obj(flat):
        if time.time() > deadline:
            return 1e10
        if np.any(np.isnan(flat)) or np.any(np.isinf(flat)):
            return 1e10
        p = flat.reshape(N, 2)
        s = _score_points(p)
        if s < 1e-12:
            return 1e10
        return -s
    try:
        res = minimize(
            obj, x0, method='Powell',
            options={'maxiter': 1500, 'xtol': 1e-10, 'ftol': 1e-12}
        )
        if res.fun < 1e9:
            return res.x.reshape(N, 2), -res.fun
    except Exception:
        pass
    return pts, _score_points(pts)

def construct():
    """Place 14 points in convex position to maximize min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pts = None
    best_score = 0.0
    
    # Initial: regular 14-gon
    reg_pts = _regular_n_gon()
    s_reg = _score_points(reg_pts)
    best_pts = reg_pts.copy()
    best_score = s_reg
    
    # Multi-restart strategy: perturb regular polygon and refine
    rng = np.random.RandomState(42)
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            radii = np.ones(N)
        else:
            seed = 100 + restart * 7
            angles, radii = _perturb_angles(np.ones(N), seed)
        
        # Angular refinement
        angles, radii, s_ang = _refine_angular(angles, radii, deadline)
        if s_ang > best_score:
            best_score = s_ang
            best_pts = _build_points(angles, radii).copy()
        
        # Cartesian refinement from best angular result
        if time.time() > deadline - 10:
            break
        pts_c, s_c = _refine_cartesian(best_pts, deadline)
        if s_c > best_score:
            best_score = s_c
            best_pts = pts_c.copy()
        
        # Powell polish
        if time.time() > deadline - 5:
            break
        pts_p, s_p = _refine_powell(best_pts, deadline)
        if s_p > best_score:
            best_score = s_p
            best_pts = pts_p.copy()
    
    # Final: try a few more perturbations of the best result
    for extra in range(4):
        if time.time() > deadline - 3:
            break
        perturbed = best_pts + rng.normal(0, 0.02, (N, 2))
        pts_e, s_e = _refine_cartesian(perturbed, deadline)
        if s_e > best_score:
            best_score = s_e
            best_pts = pts_e.copy()
    
    # Normalize to unit area
    try:
        hull = ConvexHull(best_pts)
        hull_area = hull.volume
    except Exception:
        best_pts = _regular_n_gon()
        hull = ConvexHull(best_pts)
        hull_area = hull.volume
    
    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        best_pts = best_pts * scale
    
    return best_pts
# EVOLVE-BLOCK-END
