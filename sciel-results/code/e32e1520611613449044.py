# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _vertex_score(v, pts):
    """Compute the vertex score for vertex v given all points."""
    dists = np.sqrt(np.sum((pts - v)**2, axis=1))
    # Remove self
    idx = np.argmin(dists)
    dists = np.delete(dists, idx)
    dists = np.sort(dists)
    n = len(dists)  # 19
    if np.any(dists < 0.05):
        return 0.0
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = np.diff(dists)  # length 18
    k = np.argmin(gaps)
    D = (dists[k] + dists[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        sel = dists[0:4]
    elif k == 17:
        sel = dists[15:19]
    else:
        sel = dists[k-1:k+3]
    ratios = np.maximum(sel / D, D / sel)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _polygon_score(angles):
    """Compute the overall polygon score from angles."""
    # angles: array of 20 angles, first is 0
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Check convexity: all cross products same sign
    cross = np.zeros(20)
    for i in range(20):
        p0 = pts[i]
        p1 = pts[(i+1)%20]
        p2 = pts[(i+2)%20]
        cross[i] = (p1[0]-p0[0])*(p2[1]-p1[1]) - (p1[1]-p0[1])*(p2[0]-p1[0])
    if np.any(cross == 0) or np.any(cross > 0) and np.any(cross < 0):
        return 0.0
    # Normalization
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(pts_norm**2, axis=1)))
    if mean_dist <= 0.01:
        return 0.0
    pts_norm = pts_norm / mean_dist
    # Compute min vertex score
    scores = [_vertex_score(pts_norm[i], pts_norm) for i in range(20)]
    return min(scores)

def _objective(params):
    """Objective to minimize (negative of score)."""
    n = 20
    angles = np.zeros(n)
    angles[0] = 0.0
    # params are 19 values in (0, 1), map to (0, 2pi)
    angles[1:] = 2.0 * np.pi * params
    # Ensure sorted
    angles[1:] = np.sort(angles[1:])
    # Ensure strictly increasing and within (0, 2pi)
    if angles[1] <= 0 or angles[-1] >= 2*np.pi:
        return 1.0  # bad
    for i in range(1, n):
        if angles[i] <= angles[i-1]:
            return 1.0
    score = _polygon_score(angles)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0
    
    n = 20
    # Initial guess: regular polygon
    angles_reg = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Convert to params
    params_reg = angles_reg[1:] / (2*np.pi)
    
    best_score = _polygon_score(angles_reg)
    best_angles = angles_reg.copy()
    
    # Differential evolution
    bounds = [(0.001, 0.999)] * (n-1)
    
    if time.time() < deadline - 60:
        try:
            res = differential_evolution(
                _objective, bounds,
                maxiter=100, popsize=15,
                tol=1e-6, mutation=(0.5, 1.0),
                recombination=0.7, seed=42,
                polish=False,
                update_interval=10,
                callback=lambda x, convergence: (time.time() > deadline - 90)
            )
            if res.fun < 0:  # valid
                params_opt = res.x
                angles_opt = np.zeros(n)
                angles_opt[1:] = np.sort(2*np.pi * params_opt)
                score_opt = _polygon_score(angles_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_angles = angles_opt.copy()
        except Exception:
            pass
    
    # Nelder-Mead polish from best point
    if time.time() < deadline - 30:
        try:
            params_best = best_angles[1:] / (2*np.pi)
            res_nm = minimize(
                _objective, params_best,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_nm.fun < 0:
                params_nm = res_nm.x
                angles_nm = np.zeros(n)
                angles_nm[1:] = np.sort(2*np.pi * np.clip(params_nm, 0.001, 0.999))
                score_nm = _polygon_score(angles_nm)
                if score_nm > best_score:
                    best_score = score_nm
                    best_angles = angles_nm.copy()
        except Exception:
            pass
    
    # Multiple NM restarts from perturbed best
    if time.time() < deadline - 20:
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            try:
                noise = np.random.uniform(-0.01, 0.01, size=n-1)
                params_pert = np.clip(best_angles[1:]/(2*np.pi) + noise, 0.001, 0.999)
                res_r = minimize(
                    _objective, params_pert,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res_r.fun < 0:
                    params_r = res_r.x
                    angles_r = np.zeros(n)
                    angles_r[1:] = np.sort(2*np.pi * np.clip(params_r, 0.001, 0.999))
                    score_r = _polygon_score(angles_r)
                    if score_r > best_score:
                        best_score = score_r
                        best_angles = angles_r.copy()
            except Exception:
                pass
    
    # Final: build polygon from best angles
    pts = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    return pts

# EVOLVE-BLOCK-END
