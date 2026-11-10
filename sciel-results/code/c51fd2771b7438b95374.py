# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import differential_evolution, minimize


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _face_normals(a_deg):
    a = np.deg2rad(a_deg)
    mid_angles = a + np.deg2rad(30 + 60 * np.arange(6))
    return np.column_stack([np.cos(mid_angles), np.sin(mid_angles)])


def _pair_gap(verts_a, verts_b, norms_a, norms_b):
    """
    Check separation using the 12 edge normals (6 from each hexagon).
    Returns the maximum gap over all 12 normals.
    If > 0, the hexagons are separated by at least that amount on some normal.
    """
    # Project onto normals of A
    proj_a = verts_a @ norms_a.T  # (6, 6)
    proj_b = verts_b @ norms_a.T  # (6, 6)
    gap_a = np.maximum(
        proj_b.min(axis=0) - proj_a.max(axis=0),
        proj_a.min(axis=0) - proj_b.max(axis=0)
    )
    
    # Project onto normals of B
    proj_a2 = verts_a @ norms_b.T  # (6, 6)
    proj_b2 = verts_b @ norms_b.T  # (6, 6)
    gap_b = np.maximum(
        proj_b2.min(axis=0) - proj_a2.max(axis=0),
        proj_a2.min(axis=0) - proj_b2.max(axis=0)
    )
    
    return max(gap_a.max(), gap_b.max())


def _objective(params, time_limit=600.0):
    """
    Objective function for optimization.
    Returns the side length of the minimal enclosing hexagon if valid,
    or a large penalty if invalid.
    """
    start_time = time.time()
    
    verts = np.empty((11, 6, 2))
    norms = np.empty((11, 6, 2))
    centers = np.empty((11, 2))
    
    for i in range(11):
        cx, cy, ad = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        verts[i] = _hex_vertices(cx, cy, 1.0, ad)
        norms[i] = _face_normals(ad)
        centers[i] = [cx, cy]
    
    # Check pairwise separation
    min_gap = np.inf
    for i in range(11):
        for j in range(i + 1, 11):
            g = _pair_gap(verts[i], verts[j], norms[i], norms[j])
            if g < min_gap:
                min_gap = g
            if min_gap < -1.0:
                # Heavy overlap, early exit
                return 1e10
    
    if min_gap < 1e-6:
        # Overlap penalty
        return 1e6 * (1e-6 - min_gap) + 100.0
    
    # Compute minimal enclosing regular hexagon
    all_verts = verts.reshape(-1, 2)
    centroid = all_verts.mean(axis=0)
    shifted = all_verts - centroid
    
    # Optimize orientation in [0, 60)
    best_side = np.inf
    for a_deg in np.linspace(0, 60, 121):
        n = _face_normals(a_deg)
        dots = shifted @ n.T  # (66, 6)
        r = dots.max()
        side = 2.0 * r / np.sqrt(3.0)
        if side < best_side:
            best_side = side
    
    return best_side


def _refine(params, time_budget):
    """
    Local refinement using scipy.optimize.minimize.
    """
    start_time = time.time()
    
    def obj(x):
        if time.time() - start_time > time_budget:
            return 1e10
        return _objective(x)
    
    try:
        res = minimize(
            obj, params,
            method='Nelder-Mead',
            options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10, 'adaptive': True}
        )
        if res.success or res.fun < _objective(params):
            return res.x
    except Exception:
        pass
    
    return params


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, side).
    """
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    # Initial guess: flower-like arrangement
    init = []
    init.append([0.0, 0.0, 0.0])  # center
    for k in range(6):
        ang = np.deg2rad(60 * k)
        init.append([2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0])
    for k in [0, 2, 4, 5]:
        ang = np.deg2rad(60 * k + 30)
        init.append([4.0 * np.cos(ang), 4.0 * np.sin(ang), 0.0])
    
    init_params = np.array(init).flatten()
    
    # Bounds
    bounds = []
    for i in range(11):
        bounds.append((-6.0, 6.0))
        bounds.append((-6.0, 6.0))
        bounds.append((-180.0, 180.0))
    
    best_params = init_params.copy()
    best_side = _objective(best_params)
    
    # Differential Evolution
    if time.time() - start_time < time_limit - safety_margin:
        def de_obj(x):
            if time.time() - start_time > time_limit - safety_margin:
                return 1e10
            return _objective(x)
        
        try:
            result = differential_evolution(
                de_obj,
                bounds,
                x0=init_params,
                maxiter=150,
                popsize=12,
                tol=1e-8,
                seed=42,
                polish=False,
                init='sobol',
                workers=-1,
            )
            if result.fun < best_side:
                best_params = result.x
                best_side = result.fun
        except Exception:
            pass
    
    # Local refinement
    if time.time() - start_time < time_limit - safety_margin:
        remaining = time_limit - safety_margin - (time.time() - start_time)
        if remaining > 5:
            best_params = _refine(best_params, remaining)
            new_side = _objective(best_params)
            if new_side < best_side:
                best_side = new_side
    
    # Build result
    inner_hex_data = np.empty((11, 3))
    for i in range(11):
        inner_hex_data[i] = best_params[3 * i:3 * i + 3]
    
    # Compute outer hexagon
    verts = np.empty((11, 6, 2))
    for i in range(11):
        cx, cy, ad = best_params[3 * i], best_params[3 * i + 1], best_params[3 * i + 2]
        verts[i] = _hex_vertices(cx, cy, 1.0, ad)
    
    all_verts = verts.reshape(-1, 2)
    centroid = all_verts.mean(axis=0)
    shifted = all_verts - centroid
    
    best_a = 0.0
    best_side_final = np.inf
    for a_deg in np.linspace(0, 60, 241):
        n = _face_normals(a_deg)
        dots = shifted @ n.T
        r = dots.max()
        side = 2.0 * r / np.sqrt(3.0)
        if side < best_side_final:
            best_side_final = side
            best_a = a_deg
    
    # Shift inner hexagons so centroid is at origin
    inner_hex_data[:, 0] -= centroid[0]
    inner_hex_data[:, 1] -= centroid[1]
    
    outer_hex_data = np.array([0.0, 0.0, best_a])
    outer_hex_side_length = float(best_side_final)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
