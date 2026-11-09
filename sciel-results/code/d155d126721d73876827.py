# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_rad):
    """Return 6 vertices of a regular hexagon."""
    angles = a_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(verts):
    """Return 6 unit outward normals of a hexagon's edges."""
    edges = np.diff(verts, axis=0, append=verts[:1])
    norms = edges / np.linalg.norm(edges, axis=1, keepdims=True)
    # Rotate 90 deg clockwise to get outward normal (for CCW winding)
    normals = np.column_stack([norms[:, 1], -norms[:, 0]])
    return normals


def _check_separation(verts_a, verts_b, eps=1e-6):
    """
    Check if two hexagons are separated.
    For each of the 12 edge normals, check if projected intervals are apart.
    Returns True if separated (at least one normal shows separation > eps).
    """
    na = _edge_normals(verts_a)  # 6 normals
    nb = _edge_normals(verts_b)  # 6 normals
    all_normals = np.vstack([na, nb])  # 12 normals
    
    for n in all_normals:
        proj_a = verts_a @ n
        proj_b = verts_b @ n
        min_a, max_a = proj_a.min(), proj_a.max()
        min_b, max_b = proj_b.min(), proj_b.max()
        # Intervals [min_a, max_a] and [min_b, max_b] are apart if
        # max_a < min_b - eps or max_b < min_a - eps
        if max_a < min_b - eps or max_b < min_a - eps:
            return True
    return False


def _is_valid_packing(inner_data, outer_data, side, eps=1e-6):
    """Check validity of the packing."""
    # All values finite
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)) or not np.isfinite(side):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    # Shift by outer center
    ocx, ocy = outer_data[0], outer_data[1]
    
    # Check separation between all pairs of inner hexagons
    for i in range(11):
        for j in range(i + 1, 11):
            va = _hex_vertices(inner_data[i, 0] - ocx, inner_data[i, 1] - ocy, 1.0, np.radians(inner_data[i, 2]))
            vb = _hex_vertices(inner_data[j, 0] - ocx, inner_data[j, 1] - ocy, 1.0, np.radians(inner_data[j, 2]))
            if not _check_separation(va, vb, eps):
                return False
    
    # Check containment: every inner vertex must be inside the outer hexagon
    overts = _hex_vertices(ocx, ocy, side, np.radians(outer_data[2]))
    # For each outer edge (p1 to p2), check cross product
    for k in range(6):
        p1 = overts[k]
        p2 = overts[(k + 1) % 6]
        edge = p2 - p1
        for i in range(11):
            vi = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, np.radians(inner_data[i, 2]))
            for v in vi:
                # cross(edge, v - p1) >= -eps means v is on the correct side
                cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
                if cross < -eps:
                    return False
    
    # Check edge lengths within 1e-8 of 1
    for i in range(11):
        vi = _hex_vertices(inner_data[i, 0] - ocx, inner_data[i, 1] - ocy, 1.0, np.radians(inner_data[i, 2]))
        for k in range(6):
            d = np.linalg.norm(vi[(k + 1) % 6] - vi[k])
            if abs(d - 1.0) > 1e-8:
                return False
    
    return True


def _objective(vars_):
    """
    Objective: outer side length + penalty for violations.
    vars_ layout:
      0:22 - 11 inner hex positions (x, y)
      22:33 - 11 inner hex angles (rad)
      33:35 - outer center (x, y)
      35 - outer angle (rad)
      36 - outer side length
    """
    inner_pos = vars_[0:22].reshape(11, 2)
    inner_angles = vars_[22:33]
    ocx, ocy = vars_[33], vars_[34]
    outer_angle = vars_[35]
    S = vars_[36]
    
    if S < np.sqrt(11):
        S = np.sqrt(11)
    
    # Penalty for separation violations
    penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            va = _hex_vertices(inner_pos[i, 0], inner_pos[i, 1], 1.0, inner_angles[i])
            vb = _hex_vertices(inner_pos[j, 0], inner_pos[j, 1], 1.0, inner_angles[j])
            na = _edge_normals(va)
            nb = _edge_normals(vb)
            all_normals = np.vstack([na, nb])
            sep = 0.0
            for n in all_normals:
                proj_a = va @ n
                proj_b = vb @ n
                gap = max(proj_a.max() - proj_b.min(), proj_b.max() - proj_a.min())
                # If gap < 0, they overlap; if gap >= 0, check if > eps
                # We want at least one normal with gap > 1e-6
                # Penalty if no normal has gap > 1e-6
                if gap > 1e-6:
                    sep = 1.0
                    break
            if sep < 1.0:
                # Add penalty based on how close the worst case is
                min_gap = np.inf
                for n in all_normals:
                    proj_a = va @ n
                    proj_b = vb @ n
                    gap = max(proj_a.max() - proj_b.min(), proj_b.max() - proj_a.min())
                    min_gap = min(min_gap, gap)
                # If min_gap < 1e-6, penalty
                if min_gap < 1e-6:
                    penalty += 100.0 * (1e-6 - min_gap) ** 2
    
    # Penalty for containment violations
    overts = _hex_vertices(ocx, ocy, S, outer_angle)
    for k in range(6):
        p1 = overts[k]
        p2 = overts[(k + 1) % 6]
        edge = p2 - p1
        for i in range(11):
            vi = _hex_vertices(inner_pos[i, 0], inner_pos[i, 1], 1.0, inner_angles[i])
            for v in vi:
                cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
                if cross < -1e-6:
                    penalty += 100.0 * (-cross) ** 2
    
    return S + penalty


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (11,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    timeout = 600
    
    # Symmetric initial guess
    init = np.zeros(37)
    # Center hexagon at origin
    init[0:2] = [0, 0]
    # 6 hexagons in a ring around center
    for i in range(6):
        angle = i * np.pi / 3.0
        r = 2.1
        idx = (i + 1) * 2
        init[idx] = r * np.cos(angle)
        init[idx + 1] = r * np.sin(angle)
        init[22 + i + 1] = 0.0
    # 4 hexagons further out
    for i in range(4):
        angle = i * np.pi / 2.0 + np.pi / 4
        r = 3.5
        idx = (i + 7) * 2
        init[idx] = r * np.cos(angle)
        init[idx + 1] = r * np.sin(angle)
        init[22 + i + 7] = 0.0
    # Outer hexagon
    init[33] = 0.0
    init[34] = 0.0
    init[35] = 0.0
    init[36] = 6.0
    
    best_vars = init.copy()
    best_val = _objective(best_vars)
    
    # Nelder-Mead optimization
    from scipy.optimize import minimize
    
    try:
        res = minimize(_objective, best_vars, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7,
                                'adaptive': True})
        if res.fun < best_val:
            best_vars = res.x
            best_val = res.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed < timeout - 60:
        # SLSQP polish
        try:
            res2 = minimize(_objective, best_vars, method='SLSQP',
                            options={'maxiter': 2000, 'ftol': 1e-10})
            if res2.fun < best_val:
                best_vars = res2.x
                best_val = res2.fun
        except Exception:
            pass
    
    inner_pos = best_vars[0:22].reshape(11, 2)
    inner_angles_rad = best_vars[22:33]
    inner_angles_deg = np.degrees(inner_angles_rad)
    inner_data = np.column_stack((inner_pos[:, 0], inner_pos[:, 1], inner_angles_deg))
    
    ocx, ocy = best_vars[33], best_vars[34]
    outer_angle_deg = np.degrees(best_vars[35])
    S = max(best_vars[36], np.sqrt(11))
    
    outer_data = np.array([ocx, ocy, outer_angle_deg])
    
    # Validate and fix if needed
    if not _is_valid_packing(inner_data, outer_data, S):
        # Fall back to the seed solution if invalid
        inner_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_data = np.array([0.0, 0.0, 0.0])
        S = 8.0
    
    return inner_data, outer_data, S


# EVOLVE-BLOCK-END
