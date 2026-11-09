import numpy as np
import time
from scipy.optimize import minimize

def _vertex_score(v, pts):
    """Compute the vertex score for vertex v given all points pts."""
    diffs = pts - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    mask = np.ones(len(pts), dtype=bool)
    mask[np.argmin(dists)] = False
    d = np.sort(dists[mask])
    if np.any(d < 0.05):
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k - 1:k + 3]
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(pts):
    scores = [_vertex_score(pts[i], pts) for i in range(len(pts))]
    return min(scores)

def _is_valid(pts):
    if len(pts) != 20:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    for i in range(20):
        for j in range(i + 1, 20):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-12:
                return False
    signs = []
    for i in range(20):
        p0 = pts[i]
        p1 = pts[(i + 1) % 20]
        p2 = pts[(i + 2) % 20]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-14:
            return False
        signs.append(np.sign(cross))
    if not all(s == signs[0] for s in signs):
        return False
    return True

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return pts_norm, False
    pts_norm = pts_norm / mean_dist
    return pts_norm, True

def _convexity_penalty(pts):
    """Return a penalty that is 0 if convex, positive otherwise."""
    penalty = 0.0
    for i in range(20):
        p0 = pts[i]
        p1 = pts[(i + 1) % 20]
        p2 = pts[(i + 2) % 20]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        # We want all crosses to have the same sign (positive for CCW)
        # Penalize negative crosses
        if cross < 0:
            penalty += -cross
        # Also penalize very small crosses (near-collinear)
        if abs(cross) < 1e-6:
            penalty += 1e-6 - abs(cross)
    return penalty

def _objective(x, start_time, deadline):
    """Objective function for optimization. x is 20 angles or 40 (radius, angle) pairs."""
    if time.time() > deadline:
        return 1e6
    
    n = 20
    # x contains 20 radii and 20 angles (40 params total)
    radii = x[:n]
    angles = x[n:]
    
    # Ensure positive radii
    radii = np.maximum(radii, 0.1)
    
    pts = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    
    norm_pts, valid = _normalize(pts)
    if not valid:
        return 1e6
    
    # Check convexity with penalty
    cp = _convexity_penalty(norm_pts)
    if cp > 1e-4:
        # Strong penalty for non-convex
        return 1e3 * cp + 1.0
    
    if not _is_valid(norm_pts):
        return 1.0 + cp
    
    score = _min_vertex_score(norm_pts)
    # We want to maximize score, so minimize negative score
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin

    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])

    norm_pts, valid = _normalize(pts)
    if valid and _is_valid(norm_pts):
        best_score = _min_vertex_score(norm_pts)
        best_pts = pts.copy()
    else:
        best_score = 0.0
        best_pts = pts.copy()

    # Try Nelder-Mead optimization with convexity-aware objective
    if time.time() < deadline - 60:
        # Initial guess: regular polygon with unit radii
        x0 = np.concatenate([np.ones(n), angles])
        
        # Try multiple restarts with different initial perturbations
        rng = np.random.default_rng(42)
        
        for restart in range(5):
            if time.time() > deadline - 30:
                break
            
            if restart == 0:
                x_init = x0.copy()
            else:
                # Perturb angles and radii
                angle_perturb = rng.normal(0, 0.1, size=n)
                radius_perturb = rng.normal(0, 0.1, size=n)
                x_init = np.concatenate([np.ones(n) + radius_perturb, angles + angle_perturb])
            
            # Use Nelder-Mead with adaptive parameters
            try:
                result = minimize(
                    _objective,
                    x_init,
                    args=(start_time, deadline),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True,
                        'initial_simplex': None  # Let scipy choose
                    }
                )
            except Exception:
                continue
            
            if result.success or result.fun < -best_score:
                # Reconstruct points
                radii = np.maximum(result.x[:n], 0.1)
                trial_angles = result.x[n:]
                trial_pts = np.column_stack([
                    radii * np.cos(trial_angles),
                    radii * np.sin(trial_angles)
                ])
                
                norm_trial, v = _normalize(trial_pts)
                if v and _is_valid(norm_trial):
                    score = _min_vertex_score(norm_trial)
                    if score > best_score:
                        best_score = score
                        best_pts = trial_pts.copy()
    
    # Final local search with small perturbations
    if time.time() < deadline - 10:
        step = 0.02
        for iteration in range(2000):
            if time.time() > deadline - 5:
                break
            
            # Perturb radii and angles slightly
            perturb_r = rng.normal(0, step, size=n)
            perturb_a = rng.normal(0, step * 0.5, size=n)
            
            # Extract current radii and angles from best_pts
            cur_radii = np.linalg.norm(best_pts, axis=1)
            cur_angles = np.arctan2(best_pts[:, 1], best_pts[:, 0])
            
            trial_radii = np.maximum(cur_radii + perturb_r, 0.5)
            trial_angles = cur_angles + perturb_a
            
            trial_pts = np.column_stack([
                trial_radii * np.cos(trial_angles),
                trial_radii * np.sin(trial_angles)
            ])
            
            norm_trial, v = _normalize(trial_pts)
            if not v or not _is_valid(norm_trial):
                continue
            
            score = _min_vertex_score(norm_trial)
            if score > best_score:
                best_score = score
                best_pts = trial_pts.copy()
                step *= 0.95
            else:
                step = min(step * 1.03, 0.1)

    return best_pts
