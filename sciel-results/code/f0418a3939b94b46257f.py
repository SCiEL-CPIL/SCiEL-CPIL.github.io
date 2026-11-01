import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# EVOLVE-BLOCK-START
import random

# Pre-compute triangle indices for vectorized area calculation
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def calc_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        if len(points) < 3:
            return 0.0
        
        # Calculate convex hull area
        hull = scipy.spatial.ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        
        if hull_area < 1e-10:
            return 0.0
            
        # Vectorized triangle area calculation
        pi = points[_I]
        pj = points[_J]
        pk = points[_K]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        min_area = 0.5 * np.abs(cross).min()
        
        return min_area / hull_area
    except Exception:
        return 0.0

def objective(points_flat):
    """
    Objective function for minimization.
    Minimizes negative score.
    """
    points = points_flat.reshape(-1, 2)
    score = calc_score(points)
    return -score

def _normalize(points):
    """Scale so convex hull area = 1."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        ha = hull.volume
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _concentric_circles(n, seed=42):
    """Points on two concentric circles."""
    rng = np.random.RandomState(seed)
    # 7 points on outer, 7 on inner
    outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rng.uniform(0, 2 * np.pi)
    inner_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rng.uniform(0, 2 * np.pi)
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    pts = np.vstack([outer, inner])
    return pts

def _perturbed_polygon(n, base_radius=1.0, noise_scale=0.05, seed=None, rotation=0.0):
    """Perturbed regular polygon."""
    rng = np.random.RandomState(seed)
    base = _regular_polygon(n, base_radius, rotation)
    noise = rng.uniform(-noise_scale, noise_scale, (n, 2))
    return base + noise

def _two_ring_points(outer_angles, inner_angles, inner_radius=0.5):
    """
    Build 14 points: 7 on unit circle at outer_angles, 7 at inner_radius at inner_angles.
    Returns (14, 2) array.
    """
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    return np.vstack([outer, inner])

def _angular_objective(params):
    """
    params: [d1_1, ..., d1_7, d2_1, ..., d2_7, inner_radius]
    d1_i = angular gap between consecutive outer points (sum to 2*pi)
    d2_i = angular gap between consecutive inner points (sum to 2*pi)
    inner_radius = radius of inner circle
    
    We use the first 6 gaps for each ring (7th is determined by sum constraint).
    So params has 6 + 6 + 1 = 13 free variables.
    Actually let's use all 7 + 7 + 1 = 15 but with a penalty if sums don't equal 2*pi.
    
    Better: use 6 gaps for outer (7th = 2*pi - sum), 6 for inner (7th = 2*pi - sum), plus radius.
    Total: 13 parameters.
    """
    try:
        d1 = params[:6]
        d2 = params[6:12]
        r_inner = params[12]
        
        # Ensure positive gaps
        if np.any(d1 <= 0.01) or np.any(d2 <= 0.01):
            return 100.0  # Large penalty
        
        # Compute 7th gap
        d1_full = np.append(d1, 2 * np.pi - np.sum(d1))
        d2_full = np.append(d2, 2 * np.pi - np.sum(d2))
        
        if d1_full[-1] <= 0.01 or d2_full[-1] <= 0.01:
            return 100.0
        
        # Build angles
        outer_angles = np.cumsum(np.append([0.0], d1_full[:-1]))
        inner_angles = np.cumsum(np.append([0.0], d2_full[:-1]))
        
        # Build points
        pts = _two_ring_points(outer_angles, inner_angles, r_inner)
        
        # Normalize
        pts = _normalize(pts)
        
        # Score
        sc = calc_score(pts)
        return -sc  # Minimize negative score
    except Exception:
        return 100.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_points = None
    
    def _try_optimize_2d(init_points, method='Nelder-Mead', maxiter=2000, ftol=1e-10, xtol=1e-8):
        """Try optimizing in full 2D space from init_points."""
        nonlocal best_score, best_points
        if time.time() > deadline:
            return None
        try:
            init_norm = _normalize(init_points)
            init_score = calc_score(init_norm)
            
            if init_score > best_score:
                best_score = init_score
                best_points = init_norm.copy()
            
            if method == 'Nelder-Mead':
                res = scipy.optimize.minimize(
                    objective,
                    init_norm.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': xtol, 'fatol': ftol, 'adaptive': True}
                )
            elif method == 'Powell':
                res = scipy.optimize.minimize(
                    objective,
                    init_norm.flatten(),
                    method='Powell',
                    options={'maxiter': maxiter, 'ftol': ftol, 'xtol': xtol}
                )
            else:
                return None
            
            opt_pts = _normalize(res.x.reshape(-1, 2))
            sc = calc_score(opt_pts)
            
            if sc > best_score:
                best_score = sc
                best_points = opt_pts.copy()
            
            return (sc, opt_pts)
        except Exception:
            return None
    
    def _try_optimize_angular(init_params, method='Nelder-Mead', maxiter=5000, ftol=1e-12, xtol=1e-10):
        """Try optimizing in angular parameter space."""
        nonlocal best_score, best_points
        if time.time() > deadline:
            return None
        try:
            # Evaluate initial
            val = _angular_objective(init_params)
            if val < -best_score:  # -val > best_score
                # Reconstruct points
                d1 = init_params[:6]
                d2 = init_params[6:12]
                r_inner = init_params[12]
                d1_full = np.append(d1, 2 * np.pi - np.sum(d1))
                d2_full = np.append(d2, 2 * np.pi - np.sum(d2))
                if d1_full[-1] > 0.01 and d2_full[-1] > 0.01:
                    outer_angles = np.cumsum(np.append([0.0], d1_full[:-1]))
                    inner_angles = np.cumsum(np.append([0.0], d2_full[:-1]))
                    pts = _normalize(_two_ring_points(outer_angles, inner_angles, r_inner))
                    sc = calc_score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_points = pts.copy()
            
            res = scipy.optimize.minimize(
                _angular_objective,
                init_params,
                method=method,
                options={'maxiter': maxiter, 'xatol': xtol, 'fatol': ftol, 'adaptive': True}
            )
            
            # Reconstruct and evaluate
            d1 = res.x[:6]
            d2 = res.x[6:12]
            r_inner = res.x[12]
            d1_full = np.append(d1, 2 * np.pi - np.sum(d1))
            d2_full = np.append(d2, 2 * np.pi - np.sum(d2))
            if d1_full[-1] > 0.01 and d2_full[-1] > 0.01:
                outer_angles = np.cumsum(np.append([0.0], d1_full[:-1]))
                inner_angles = np.cumsum(np.append([0.0], d2_full[:-1]))
                pts = _normalize(_two_ring_points(outer_angles, inner_angles, r_inner))
                sc = calc_score(pts)
                
                if sc > best_score:
                    best_score = sc
                    best_points = pts.copy()
                
                return (sc, pts)
        except Exception:
            return None
    
    # Phase 1: Optimize in angular space (13D)
    # Start with regular heptagons
    equal_gap = 2 * np.pi / 7
    
    # Try different inner radii
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        if time.time() > deadline:
            break
        init_params = np.concatenate([
            np.full(6, equal_gap),  # 6 outer gaps
            np.full(6, equal_gap),  # 6 inner gaps
            [r_inner]
        ])
        _try_optimize_angular(init_params, 'Nelder-Mead', maxiter=8000, ftol=1e-12, xtol=1e-10)
    
    # Try with some asymmetry in angles
    for seed in range(5):
        if time.time() > deadline:
            break
        rng = np.random.RandomState(seed)
        # Perturb the regular gaps
        d1 = equal_gap * (1 + rng.uniform(-0.2, 0.2, 6))
        d2 = equal_gap * (1 + rng.uniform(-0.2, 0.2, 6))
        r_inner = rng.uniform(0.3, 0.7)
        init_params = np.concatenate([d1, d2, [r_inner]])
        _try_optimize_angular(init_params, 'Nelder-Mead', maxiter=5000, ftol=1e-12, xtol=1e-10)
    
    # Phase 2: Refine best in full 2D space
    if best_points is not None:
        if time.time() < deadline - 20:
            _try_optimize_2d(best_points, 'Nelder-Mead', maxiter=3000, ftol=1e-11, xtol=1e-9)
        
        if time.time() < deadline - 15:
            _try_optimize_2d(best_points, 'Powell', maxiter=300, ftol=1e-10, xtol=1e-8)
    
    # Phase 3: Additional 2D optimizations from diverse starts if time permits
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0, 0.0))
    
    # Try a few more diverse 2D starts
    diverse_starts = []
    for rot in [0.0, np.pi / 14, np.pi / 7]:
        diverse_starts.append(_regular_polygon(N, 1.0, rot))
    for seed in range(5):
        diverse_starts.append(_perturbed_polygon(N, 1.0, 0.1, seed=seed + 200, rotation=seed * 0.2))
    for seed in range(3):
        diverse_starts.append(_concentric_circles(N, seed=seed + 50))
    
    for init in diverse_starts:
        if time.time() > deadline - 10:
            break
        _try_optimize_2d(init, 'Nelder-Mead', maxiter=2000, ftol=1e-11, xtol=1e-9)
    
    # Final normalization
    if best_points is not None:
        best_points = _normalize(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
