# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Get 6 vertices of a regular hexagon with center (cx,cy), side s, angle a_deg."""
    a = np.radians(a_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _all_inner_vertices(inner_data):
    """Get all vertices of all inner hexagons."""
    verts_list = []
    for i in range(len(inner_data)):
        cx, cy, a = inner_data[i]
        verts = _hex_vertices(cx, cy, 1.0, a)
        verts_list.append(verts)
    return np.vstack(verts_list)


def _outer_hex_halfplanes(cx, cy, a_deg):
    """Get the 6 half-plane constraints for an outer hexagon.
    Returns (normals, offsets) where normal·(p - center) <= offset for each edge.
    offset = side * sqrt(3)/2 for a regular hexagon with side length 'side'.
    """
    a = np.radians(a_deg)
    normals = []
    for i in range(6):
        angle = a + i * np.pi / 3
        v1 = np.array([np.cos(angle), np.sin(angle)])
        v2 = np.array([np.cos(angle + np.pi / 3), np.sin(angle + np.pi / 3)])
        edge = v2 - v1
        n = np.array([edge[1], -edge[0]])
        mid = (v1 + v2) / 2
        if np.dot(n, mid) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)


def _side_length_given_outer(all_verts, cx, cy, a_deg):
    """Compute minimum side length of outer hexagon centered at (cx,cy) with angle a_deg 
    that encloses all vertices in all_verts."""
    normals = _outer_hex_halfplanes(cx, cy, a_deg)
    rel = all_verts - np.array([cx, cy])
    # For each normal, max projection
    projs = rel @ normals.T  # shape (n_verts, 6)
    max_projs = projs.max(axis=0)  # shape (6,)
    # side = max_proj / cos(pi/6) = max_proj * 2/sqrt(3)
    side = np.max(max_projs) * 2.0 / np.sqrt(3)
    return side


def _min_enclosing_hexagon_fast(inner_data):
    """Find minimum side length enclosing hexagon by optimizing center and angle."""
    all_verts = _all_inner_vertices(inner_data)
    
    def _obj(params):
        cx, cy, a_deg = params
        return _side_length_given_outer(all_verts, cx, cy, a_deg)
    
    # Try multiple initial angles
    best_side = np.inf
    best_params = None
    
    for a0 in [0, 30, 60, 90, 120, 150]:
        try:
            res = minimize(_obj, [0, 0, a0], method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except:
            pass
    
    return best_side, best_params


def _pair_separated(hex1, hex2, tol=1e-6):
    """Check if two unit hexagons are separated (not intersecting)."""
    cx1, cy1, a1 = hex1
    cx2, cy2, a2 = hex2
    
    verts1 = _hex_vertices(cx1, cy1, 1.0, a1)
    verts2 = _hex_vertices(cx2, cy2, 1.0, a2)
    
    # Get all 12 normals
    normals1 = _outer_hex_halfplanes(0, 0, a1)
    normals2 = _outer_hex_halfplanes(0, 0, a2)
    all_normals = np.vstack([normals1, normals2])
    
    for n in all_normals:
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    
    return False


def _check_all_separated(inner_data, tol=1e-6):
    """Check that all pairs of inner hexagons are separated."""
    n = len(inner_data)
    for i in range(n):
        for j in range(i + 1, n):
            if not _pair_separated(inner_data[i], inner_data[j], tol):
                return False
    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon's side length (maximizing 1/side).
    """
    start_time = time.time()
    deadline = start_time + 590  # Leave 10s margin
    
    # Initial configuration: hexagonal grid with small gap
    eps = 1e-4
    d = np.sqrt(3) + eps  # distance between centers of touching hexagons
    
    angles = np.radians([0, 60, 120, 180, 240, 300])
    
    # 7 hexagons: center + 6 neighbors
    positions = np.array([
        [0, 0, 0],
        [d * np.cos(angles[0]), d * np.sin(angles[0]), 0],
        [d * np.cos(angles[1]), d * np.sin(angles[1]), 0],
        [d * np.cos(angles[2]), d * np.sin(angles[2]), 0],
        [d * np.cos(angles[3]), d * np.sin(angles[3]), 0],
        [d * np.cos(angles[4]), d * np.sin(angles[4]), 0],
        [d * np.cos(angles[5]), d * np.sin(angles[5]), 0],
    ])
    
    # Add 4 more at second ring
    d2 = 2 * d
    additional = []
    for i in [0, 2, 4, 1]:  # 4 positions from second ring
        additional.append([d2 * np.cos(angles[i]), d2 * np.sin(angles[i]), 0])
    
    positions = np.vstack([positions, np.array(additional)])
    
    # Evaluate initial
    best_side, best_params = _min_enclosing_hexagon_fast(positions)
    best_result = positions.copy()
    
    if not _check_all_separated(positions, tol=1e-6):
        # Increase eps and retry
        for eps_try in [1e-3, 1e-2, 0.1]:
            d = np.sqrt(3) + eps_try
            positions = np.array([
                [0, 0, 0],
                [d * np.cos(angles[0]), d * np.sin(angles[0]), 0],
                [d * np.cos(angles[1]), d * np.sin(angles[1]), 0],
                [d * np.cos(angles[2]), d * np.sin(angles[2]), 0],
                [d * np.cos(angles[3]), d * np.sin(angles[3]), 0],
                [d * np.cos(angles[4]), d * np.sin(angles[4]), 0],
                [d * np.cos(angles[5]), d * np.sin(angles[5]), 0],
            ])
            d2 = 2 * d
            additional = []
            for i in [0, 2, 4, 1]:
                additional.append([d2 * np.cos(angles[i]), d2 * np.sin(angles[i]), 0])
            positions = np.vstack([positions, np.array(additional)])
            
            if _check_all_separated(positions, tol=1e-6):
                best_side, best_params = _min_enclosing_hexagon_fast(positions)
                best_result = positions.copy()
                break
    
    # Objective function for optimization
    def _full_objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            return 1e10
        
        inner = params.reshape(11, 3)
        
        # Check separation
        if not _check_all_separated(inner, tol=1e-6):
            return 1e8
        
        # Compute enclosing hexagon side
        side, _ = _min_enclosing_hexagon_fast(inner)
        return side
    
    # Differential evolution
    bounds = [(-8, 8), (-8, 8), (0, 60)] * 11
    
    if time.time() - start_time < deadline - 120:
        try:
            de_result = differential_evolution(
                _full_objective, bounds,
                seed=42, maxiter=80, popsize=12,
                tol=1e-5, mutation=(0.5, 1.5), recombination=0.7,
                polish=False, init='sobol'
            )
            
            if de_result.fun < best_side:
                best_side = de_result.fun
                best_result = de_result.x.reshape(11, 3)
        except Exception as e:
            pass
    
    # Nelder-Mead refinement
    if best_result is not None and time.time() - start_time < deadline - 30:
        try:
            nm_result = minimize(
                _full_objective, best_result.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if nm_result.fun < best_side:
                best_side = nm_result.fun
                best_result = nm_result.x.reshape(11, 3)
        except Exception as e:
            pass
    
    # Final high-precision polish
    if best_result is not None and time.time() - start_time < deadline - 10:
        try:
            nm_result2 = minimize(
                _full_objective, best_result.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            if nm_result2.fun < best_side:
                best_side = nm_result2.fun
                best_result = nm_result2.x.reshape(11, 3)
        except Exception as e:
            pass
    
    # Compute final outer hexagon
    _, outer_params = _min_enclosing_hexagon_fast(best_result)
    outer_cx, outer_cy, outer_angle = outer_params
    outer_side = best_side
    
    if outer_side < np.sqrt(11):
        outer_side = np.sqrt(11)
    
    inner_hex_data = best_result
    outer_hex_data = np.array([outer_cx, outer_cy, outer_angle])
    
    return inner_hex_data, outer_hex_data, float(outer_side)


# EVOLVE-BLOCK-END
