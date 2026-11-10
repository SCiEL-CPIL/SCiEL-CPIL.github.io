# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Return the 6 vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _min_outer_side_for_inner(inner_hex_data, outer_center):
    """
    Compute the minimum side length of a regular hexagon (centered at outer_center, angle 0)
    that contains all inner hexagons (unit side).
    Returns (side, ok) where ok is True if all vertices are inside.
    """
    cx, cy, _ = outer_center
    # For a regular hexagon with side S and angle 0, vertices at angles 0,60,...,300
    # Edge normals are at angles 30+60k. Point p is inside if dot(p-center, n_k) <= S*sqrt(3)/2
    normals = []
    for k in range(6):
        ang = np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    normals = np.array(normals)  # shape (6, 2)
    
    max_proj = 0.0
    all_finite = True
    for i in range(len(inner_hex_data)):
        x, y, a_deg = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        if not np.all(np.isfinite(verts)):
            all_finite = False
            break
        shifted = verts - np.array([cx, cy])
        projs = shifted @ normals.T
        m = np.max(projs)
        if m > max_proj:
            max_proj = m
    
    if not all_finite:
        return float('inf'), False
    
    side = max_proj * 2.0 / np.sqrt(3.0)
    return side, True


def _check_validity(inner_hex_data, outer_hex_data, side):
    """
    Check all validity conditions. Returns True if valid.
    """
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data))):
        return False
    if not np.isfinite(side):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    ox, oy, oa = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, side, oa)
    
    # Check every inner vertex is inside outer hexagon
    for i in range(len(inner_hex_data)):
        x, y, a_deg = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        for p in verts:
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j + 1) % 6]
                e = p2 - p1
                v = p - p1
                cross = e[0] * v[1] - e[1] * v[0]
                if cross < -1e-6:
                    return False
    
    # Check every pair of inner hexagons is separated
    n_inner = len(inner_hex_data)
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            x1, y1, a1 = inner_hex_data[i]
            x2, y2, a2 = inner_hex_data[j]
            v1 = _hex_vertices(x1, y1, 1.0, a1)
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            
            def get_normals(verts):
                norms = []
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    e = p2 - p1
                    n = np.array([e[1], -e[0]])
                    n = n / np.linalg.norm(n)
                    norms.append(n)
                return np.array(norms)
            
            n1 = get_normals(v1)
            n2 = get_normals(v2)
            all_normals = np.vstack([n1, n2])
            
            separated = False
            for n in all_normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def _objective(params, outer_angle):
    """
    Objective: minimize the outer hexagon side length.
    params: 33 values (x, y, angle_deg) for 11 inner hexagons.
    outer_angle: rotation of the outer hexagon in degrees.
    """
    inner = params.reshape(11, 3)
    inner[:, 2] = inner[:, 2] % 360.0
    
    # Compute the minimum side for this outer angle
    # For outer hexagon with angle oa, edge normals are at oa + 30 + 60k
    cx, cy = 0.0, 0.0
    normals = []
    for k in range(6):
        ang = np.radians(outer_angle + 30 + 60 * k)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    normals = np.array(normals)
    
    max_proj = 0.0
    for i in range(len(inner)):
        x, y, a_deg = inner[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        if not np.all(np.isfinite(verts)):
            return 1e10
        shifted = verts - np.array([cx, cy])
        projs = shifted @ normals.T
        m = np.max(projs)
        if m > max_proj:
            max_proj = m
    
    if not np.isfinite(max_proj):
        return 1e10
    
    side = max_proj * 2.0 / np.sqrt(3.0)
    
    # Check separation constraints - add penalty if violated
    penalty = 0.0
    n_inner = len(inner)
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            x1, y1, a1 = inner[i]
            x2, y2, a2 = inner[j]
            v1 = _hex_vertices(x1, y1, 1.0, a1)
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            
            def get_normals(verts):
                norms = []
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    e = p2 - p1
                    n = np.array([e[1], -e[0]])
                    n = n / np.linalg.norm(n)
                    norms.append(n)
                return np.array(norms)
            
            n1 = get_normals(v1)
            n2 = get_normals(v2)
            all_normals = np.vstack([n1, n2])
            
            min_gap = float('inf')
            for n in all_normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                gap1 = min2 - max1
                gap2 = min1 - max2
                gap = max(gap1, gap2)
                if gap < min_gap:
                    min_gap = gap
            
            if min_gap < 1e-6:
                penalty += (1e-6 - min_gap) ** 2
    
    return side + 100.0 * penalty


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill
    
    # Initial seed: honeycomb-like arrangement
    inner_seed = np.array(
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
    
    best_inner = inner_seed.copy()
    best_outer = np.array([0.0, 0.0, 0.0])
    best_side = _min_outer_side_for_inner(inner_seed, best_outer)[0]
    
    if not np.isfinite(best_side):
        best_side = 8.0
    
    # Try different outer hexagon angles
    for outer_angle in [0, 15, 30, 45, 60]:
        if time.time() > deadline - 30.0:
            break
        
        # Use differential evolution for global optimization
        bounds = [(-6, 6)] * 11 + [(-6, 6)] * 11 + [(-180, 180)] * 11
        
        try:
            result = differential_evolution(
                _objective, bounds, args=(outer_angle,),
                seed=42, maxiter=100, tol=1e-6, popsize=15,
                mutation=(0.3, 1.0), recombination=0.8,
                polish=True, init='latinhypercube'
            )
            
            if result.success or result.fun < best_side:
                inner_opt = result.x.reshape(11, 3)
                inner_opt[:, 2] = inner_opt % 360.0
                
                # Compute the actual side with this outer angle
                cx, cy = 0.0, 0.0
                normals = []
                for k in range(6):
                    ang = np.radians(outer_angle + 30 + 60 * k)
                    normals.append(np.array([np.cos(ang), np.sin(ang)]))
                normals = np.array(normals)
                
                max_proj = 0.0
                for i in range(len(inner_opt)):
                    x, y, a_deg = inner_opt[i]
                    verts = _hex_vertices(x, y, 1.0, a_deg)
                    shifted = verts - np.array([cx, cy])
                    projs = shifted @ normals.T
                    m = np.max(projs)
                    if m > max_proj:
                        max_proj = m
                
                side_opt = max_proj * 2.0 / np.sqrt(3.0)
                
                if np.isfinite(side_opt) and side_opt < best_side:
                    # Check validity
                    outer_hex = np.array([0.0, 0.0, float(outer_angle)])
                    if _check_validity(inner_opt, outer_hex, side_opt):
                        best_inner = inner_opt
                        best_outer = outer_hex
                        best_side = side_opt
        except Exception:
            continue
    
    # Final local refinement using Nelder-Mead
    if time.time() < deadline - 20.0:
        try:
            res = minimize(
                lambda p: _objective(p, best_outer[2]),
                best_inner.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            inner_opt = res.x.reshape(11, 3)
            inner_opt[:, 2] = inner_opt % 360.0
            
            cx, cy = 0.0, 0.0
            normals = []
            for k in range(6):
                ang = np.radians(best_outer[2] + 30 + 60 * k)
                normals.append(np.array([np.cos(ang), np.sin(ang)]))
            normals = np.array(normals)
            
            max_proj = 0.0
            for i in range(len(inner_opt)):
                x, y, a_deg = inner_opt[i]
                verts = _hex_vertices(x, y, 1.0, a_deg)
                shifted = verts - np.array([cx, cy])
                projs = shifted @ normals.T
                m = np.max(projs)
                if m > max_proj:
                    max_proj = m
            
            side_opt = max_proj * 2.0 / np.sqrt(3.0)
            
            if np.isfinite(side_opt) and side_opt < best_side:
                if _check_validity(inner_opt, best_outer, side_opt):
                    best_inner = inner_opt
                    best_side = side_opt
        except Exception:
            pass
    
    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END
