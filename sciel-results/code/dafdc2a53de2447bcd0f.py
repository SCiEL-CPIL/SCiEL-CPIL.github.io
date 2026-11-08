"""Pack 12 unit hexagons in minimal enclosing hexagon."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    theta = np.radians(angle_deg)
    angles = theta + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _hex_face_normals(angle_deg):
    theta = np.radians(angle_deg)
    angles = np.pi / 6 + theta + np.pi / 3 * np.arange(6)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _min_side_for_points(points, outer_angle, outer_cx, outer_cy):
    """Min side length of hexagon (center, rotation fixed) to contain all points."""
    normals = _hex_face_normals(outer_angle)
    dx = points[:, 0] - outer_cx
    dy = points[:, 1] - outer_cy
    projections = np.abs(dx * normals[:, 0] + dy * normals[:, 1])
    max_proj = np.max(projections)
    return max_proj * 2.0 / np.sqrt(3.0)

def _objective(params):
    """Outer hexagon side length."""
    pos = params[:N*2].reshape(N, 2)
    angles = params[N*2:N*2+N]
    outer_cx = params[N*2+N]
    outer_cy = params[N*2+N+1]
    outer_angle = params[N*2+N+2]
    
    # Collect all vertices of inner hexagons
    all_verts = []
    for i in range(N):
        v = _hex_vertices(pos[i,0], pos[i,1], angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    return _min_side_for_points(all_verts, outer_angle, outer_cx, outer_cy)

def _overlap_penalty(pos, angles, weight=500.0):
    """Penalty for overlapping inner hexagons."""
    penalty = 0.0
    min_dist = np.sqrt(3.0)  # minimum center distance for non-overlapping unit hexagons
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    return weight * penalty

def _penalized(params, weight=500.0):
    base = _objective(params)
    pos = params[:N*2].reshape(N, 2)
    angles = params[N*2:N*2+N]
    return base + _overlap_penalty(pos, angles, weight)

def _honeycomb_init(seed=0):
    rng = np.random.RandomState(seed)
    d = np.sqrt(3.0)
    pos = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # 1 central
    pos[0] = [0, 0]
    # 6 in first ring
    for k in range(6):
        a = k * np.pi / 3
        pos[k+1] = [d * np.cos(a), d * np.sin(a)]
    # 5 in second ring
    for k in range(5):
        a = k * 2 * np.pi / 5 + np.pi/5
        r = 2 * d
        pos[k+7] = [r * np.cos(a), r * np.sin(a)]
    
    # Perturb
    pos += rng.uniform(-0.02, 0.02, size=pos.shape)
    angles += rng.uniform(-5, 5, size=N)
    
    outer_cx = 0.0
    outer_cy = 0.0
    outer_angle = 0.0
    
    params = np.concatenate([pos.flatten(), angles, [outer_cx, outer_cy, outer_angle]])
    return params

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_params = None
    
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        p0 = _honeycomb_init(seed=r * 13 + 3)
        
        try:
            res = minimize(
                _penalized, p0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Also try without penalty if close to valid
        if time.time() - start_time > DEADLINE - 5:
            break
    
    if best_params is None:
        best_params = _honeycomb_init(0)
        best_score = _objective(best_params)
    
    # Final polish: optimize without penalty (assuming no overlaps)
    if time.time() - start_time < DEADLINE - 5:
        try:
            res2 = minimize(
                _objective, best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    pos = best_params[:N*2].reshape(N, 2)
    angles = best_params[N*2:N*2+N]
    outer_cx = best_params[N*2+N]
    outer_cy = best_params[N*2+N+1]
    outer_angle = best_params[N*2+N+2]
    
    # Compute final side
    all_verts = []
    for i in range(N):
        all_verts.append(_hex_vertices(pos[i,0], pos[i,1], angles[i]))
    all_verts = np.vstack(all_verts)
    side = _min_side_for_points(all_verts, outer_angle, outer_cx, outer_cy)
    
    inner = [[float(pos[i,0]), float(pos[i,1]), float(angles[i])] for i in range(N)]
    return inner, (float(outer_cx), float(outer_cy)), float(side), float(outer_angle)
