# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


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
    # Outer hexagon vertices for side length S (angle 0, center at (cx,cy))
    # We need to find minimum S such that all inner vertices are inside.
    # A point (px, py) is inside a regular hexagon of side S centered at (cx,cy) with angle 0
    # if and only if for all 6 edge normals n_i, dot((px-cx, py-cy), n_i) <= S * apothem_factor
    # Actually, for a regular hexagon with side S and angle 0 (flat-top or pointy-top?),
    # Let's use the vertex definition: vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    # The apothem (distance from center to edge midpoint) is S * sqrt(3)/2.
    # The edge normals point outward. For a hexagon with vertices at angle 0, 60, ..., the edges
    # are between consecutive vertices. The normal to the edge between vertex i and i+1
    # is at angle 30 + 60*i degrees.
    # A point p is inside if for all 6 normals n_k (unit vectors at 30+60k degrees),
    # dot(p - center, n_k) <= S * sqrt(3)/2.
    # So S >= max_k dot(p - center, n_k) / (sqrt(3)/2) = 2/sqrt(3) * max_k dot(p - center, n_k)
    
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
        # Project onto each normal
        projs = shifted @ normals.T  # shape (6, 6): 6 vertices x 6 normals
        # We need max over all vertices and all normals
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
    # All values finite
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data))):
        return False
    if not np.isfinite(side):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    ox, oy, oa = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, side, oa)
    
    # Check every inner vertex is inside outer hexagon
    # Using the cross product condition: for each outer edge from p1 to p2,
    # cross(p2 - p1, p - p1) >= -1e-6
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
    # For each pair, check on at least one of their 12 edge normals the projected intervals are apart by > 1e-6
    n_inner = len(inner_hex_data)
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            x1, y1, a1 = inner_hex_data[i]
            x2, y2, a2 = inner_hex_data[j]
            v1 = _hex_vertices(x1, y1, 1.0, a1)
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            
            # Get edge normals for both hexagons
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
                # Intervals are apart if max1 < min2 - 1e-6 or max2 < min1 - 1e-6
                if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


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
    deadline = start_time + 600.0  # 600s wall-clock limit
    
    # Initial seed: the grid arrangement from the current program
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
    
    # Compute initial side
    outer_seed = np.array([0.0, 0.0, 0.0])
    side_seed, ok_seed = _min_outer_side_for_inner(inner_seed, outer_seed)
    
    if not ok_seed or not np.isfinite(side_seed):
        # Fallback
        return inner_seed, outer_seed, 8.0
    
    best_inner = inner_seed.copy()
    best_outer = outer_seed.copy()
    best_side = side_seed
    best_score = 1.0 / best_side
    
    # Try to improve using optimization
    # We'll optimize the positions and angles of the 11 inner hexagons
    # to minimize the required outer side length.
    
    def objective(params):
        # params: 11 * 3 = 33 values: (x, y, angle_deg) for each inner hex
        inner = params.reshape(11, 3)
        # Keep angles small to avoid wrap-around issues
        inner[:, 2] = inner[:, 2] % 360.0
        
        side, ok = _min_outer_side_for_inner(inner, np.array([0.0, 0.0, 0.0]))
        if not ok or not np.isfinite(side):
            return 1e10
        return side
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > deadline - 30.0:
            break
        
        # Perturb the seed
        if restart == 0:
            x0 = inner_seed.flatten()
        else:
            # Random perturbation
            scale = 0.5
            x0 = inner_seed.flatten() + rng.normal(0, scale, size=33)
        
        # Use Nelder-Mead or L-BFGS-B
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < best_side:
                inner_opt = res.x.reshape(11, 3)
                inner_opt[:, 2] = inner_opt[:, 2] % 360.0
                side_opt, ok_opt = _min_outer_side_for_inner(inner_opt, np.array([0.0, 0.0, 0.0]))
                
                if ok_opt and np.isfinite(side_opt) and side_opt < best_side:
                    # Check validity
                    if _check_validity(inner_opt, np.array([0.0, 0.0, 0.0]), side_opt):
                        best_inner = inner_opt
                        best_outer = np.array([0.0, 0.0, 0.0])
                        best_side = side_opt
                        best_score = 1.0 / best_side
        except Exception:
            continue
    
    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END
