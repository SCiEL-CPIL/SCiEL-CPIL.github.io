# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _unit_hex_vertices():
    angles = np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _hexagon_edge_normals(angle_deg):
    """Return 6 outward unit edge normals for a unit hexagon with given angle."""
    a = np.deg2rad(angle_deg)
    # Edge normals are at angle + 30 + 60k
    na = a + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([np.cos(na), np.sin(na)])


def _min_s_support(centers, angles_deg, outer_angle_deg):
    """
    Compute minimum outer hex side length s such that all inner hexagons fit.
    Uses support function: for each of the 6 outer edge normals, find max
    projection of all inner hexagon vertices, then s = 2*max_support/sqrt(3).
    """
    outer_angle = np.deg2rad(outer_angle_deg)
    # Outer hexagon edge normals
    on_a = outer_angle + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    outer_normals = np.column_stack([np.cos(on_a), np.sin(on_a)])  # (6,2)
    
    max_support = 0.0
    for i in range(len(centers)):
        # Vertices of inner hex i
        a_i = np.deg2rad(angles_deg[i])
        va = a_i + np.deg2rad(60.0 * np.arange(6))
        verts = centers[i] + np.column_stack([np.cos(va), np.sin(va)])  # (6,2)
        # Project onto each outer normal
        proj = verts @ outer_normals.T  # (6, 6)
        max_per_normal = np.max(proj, axis=0)  # (6,)
        ms = np.max(max_per_normal)
        if ms > max_support:
            max_support = ms
    
    s = 2.0 * max_support / np.sqrt(3.0)
    return s


def _min_s_support_fast(centers_flat, outer_angle_deg):
    """
    Fast version: centers_flat is (12,2), all angles assumed 0.
    """
    outer_angle = np.deg2rad(outer_angle_deg)
    on_a = outer_angle + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    outer_normals = np.column_stack([np.cos(on_a), np.sin(on_a)])  # (6,2)
    
    # All vertices of all 12 unit hexagons (angle 0)
    # unit hex vertices
    angles0 = np.deg2rad(60.0 * np.arange(6))
    uverts = np.column_stack([np.cos(angles0), np.sin(angles0)])  # (6,2)
    
    # For each hex, vertices = center + uverts
    # Projection onto outer_normals: (center @ outer_normals.T) + (uverts @ outer_normals.T)
    # uverts @ outer_normals.T is (6,6), same for all
    uv_proj = uverts @ outer_normals.T  # (6,6)
    
    max_support = 0.0
    for i in range(12):
        c = centers_flat[i]  # (2,)
        c_proj = c @ outer_normals.T  # (6,)
        # Max vertex projection per normal
        max_per_normal = np.max(uv_proj, axis=0) + c_proj  # (6,)
        ms = np.max(max_per_normal)
        if ms > max_support:
            max_support = ms
    
    s = 2.0 * max_support / np.sqrt(3.0)
    return s


def _pairwise_min_distance(centers, angles_deg):
    """
    Compute minimum distance between any pair of hexagon edges (for overlap detection).
    Uses SAT: for each pair, check all 12 normals (6 from each hex).
    Returns the minimum "gap" (negative means overlap).
    """
    n = len(centers)
    min_gap = np.inf
    
    for i in range(n):
        for j in range(i + 1, n):
            # Get vertices
            a_i = np.deg2rad(angles_deg[i])
            va_i = a_i + np.deg2rad(60.0 * np.arange(6))
            vi = centers[i] + np.column_stack([np.cos(va_i), np.sin(va_i)])
            
            a_j = np.deg2rad(angles_deg[j])
            va_j = a_j + np.deg2rad(60.0 * np.arange(6))
            vj = centers[j] + np.column_stack([np.cos(va_j), np.sin(va_j)])
            
            # Normals of hex i
            ni_a = a_i + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
            ni = np.column_stack([np.cos(ni_a), np.sin(ni_a)])
            
            # Normals of hex j
            nj_a = a_j + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
            nj = np.column_stack([np.cos(nj_a), np.sin(nj_a)])
            
            pair_gap = np.inf
            for k in range(6):
                # Normal from hex i
                n_vec = ni[k]
                proj_i = vi @ n_vec
                proj_j = vj @ n_vec
                gap = max(np.min(proj_j) - np.max(proj_i), np.min(proj_i) - np.max(proj_j))
                # gap > 0 means separated, gap < 0 means overlap
                if gap < pair_gap:
                    pair_gap = gap
                if pair_gap < 0:
                    # Overlap found, no need to check more
                    break
            if pair_gap < 0:
                break
            for k in range(6):
                n_vec = nj[k]
                proj_i = vi @ n_vec
                proj_j = vj @ n_vec
                gap = max(np.min(proj_j) - np.max(proj_i), np.min(proj_i) - np.max(proj_j))
                if gap < pair_gap:
                    pair_gap = gap
                if pair_gap < 0:
                    break
            if pair_gap < min_gap:
                min_gap = pair_gap
            if min_gap < 0:
                break
        if min_gap < 0:
            break
    
    return min_gap


def _objective(params):
    """
    Objective: minimize s + penalty for overlaps.
    params: [x0,y0, ..., x11,y11, outer_angle]  (25 params)
    All inner hex angles fixed at 0.
    """
    centers = params[:24].reshape(12, 2)
    outer_angle = params[24]
    
    s = _min_s_support_fast(centers, outer_angle)
    
    # Overlap penalty: use a smooth approximation
    # Compute center-to-center distances; for hexagons with same orientation,
    # the minimum edge-to-edge gap is approximately (dist - 2*apothem) where
    # apothem = sqrt(3)/2 for unit hex, but this is only valid for parallel edges.
    # Better: use a simpler penalty based on center distance.
    # For two unit hexagons at angle 0, they just touch when center distance = 2*sqrt(3)
    # (the distance between centers of adjacent hexagons in a hex lattice).
    # Actually for same-orientation unit hexagons, the minimum center distance
    # for non-overlap depends on direction. The worst case is along the flat-to-flat
    # direction: distance = 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732? No.
    # For a regular hexagon with side 1, the distance from center to a vertex is 1,
    # and center to edge midpoint is sqrt(3)/2.
    # Two identical hexagons touching edge-to-edge: center distance = 2 * sqrt(3)/2 = sqrt(3).
    # But touching vertex-to-vertex: center distance = 2.
    # The minimum non-overlap center distance for same-orientation hexagons is sqrt(3).
    
    # Use a smooth penalty: for each pair, if dist < sqrt(3), add penalty
    sqrt3 = np.sqrt(3.0)
    penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < sqrt3:
                # Smooth quadratic penalty
                penalty += (sqrt3 - d) ** 2
    
    return s + 10.0 * penalty


def _make_initial_grid():
    """
    Create initial hexagonal lattice arrangement of 12 unit hexagons.
    Center + 6 ring-1 + 5 ring-2 (at the 30+60k positions).
    Returns (12,2) centers and outer angle.
    """
    sqrt3 = np.sqrt(3.0)
    pts = np.array([[0.0, 0.0]])
    for i in range(6):
        ang = np.deg2rad(60.0 * i)
        pts = np.vstack([pts, [2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)]])
    for i in range(5):
        ang = np.deg2rad(30.0 + 60.0 * i)
        pts = np.vstack([pts, [2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)]])
    
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    return pts, 0.0


def _make_initial_grid_alt():
    """Alternative: 3x4 rectangular-ish arrangement."""
    sqrt3 = np.sqrt(3.0)
    # Hex lattice: rows offset by sqrt(3)/2 in y, 1.5 in x
    # Let's do a compact 12-hex cluster
    pts = []
    # Row 0: 4 hexes
    for i in range(4):
        pts.append([i * 2.0, 0.0])
    # Row 1: 4 hexes, offset
    for i in range(4):
        pts.append([i * 2.0 + 1.0, sqrt3])
    # Row 2: 4 hexes
    for i in range(4):
        pts.append([i * 2.0, 2.0 * sqrt3])
    pts = np.array(pts)
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    return pts, 0.0


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    best_s = np.inf
    best_params = None
    best_centers = None
    best_outer_angle = 0.0
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Config 1: hexagonal lattice cluster
    c1, a1 = _make_initial_grid()
    initial_configs.append((c1, a1))
    
    # Config 2: rectangular grid
    c2, a2 = _make_initial_grid_alt()
    initial_configs.append((c2, a2))
    
    # Config 3: hexagonal lattice with different ring-2 selection
    sqrt3 = np.sqrt(3.0)
    pts3 = np.array([[0.0, 0.0]])
    for i in range(6):
        ang = np.deg2rad(60.0 * i)
        pts3 = np.vstack([pts3, [2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)]])
    # Use the 6 at distance 4*sqrt(3) but only 5
    for i in range(5):
        ang = np.deg2rad(60.0 * i)
        pts3 = np.vstack([pts3, [4.0 * sqrt3 * np.cos(ang), 4.0 * sqrt3 * np.sin(ang)]])
    centroid3 = np.mean(pts3, axis=0)
    initial_configs.append((pts3 - centroid3, 0.0))
    
    # Config 4: 2x6 grid
    pts4 = []
    for row in range(2):
        for col in range(6):
            x = col * 2.0
            y = row * 2.0 * sqrt3
            pts4.append([x, y])
    pts4 = np.array(pts4)
    centroid4 = np.mean(pts4, axis=0)
    initial_configs.append((pts4 - centroid4, 0.0))
    
    for centers_init, outer_angle_init in initial_configs:
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Build initial params
        params_init = np.concatenate([centers_init.flatten(), [outer_angle_init]])
        
        # Try multiple outer angles for the initial grid
        for oa in [0.0, 15.0, 30.0, 45.0]:
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break
            
            params_init = np.concatenate([centers_init.flatten(), [oa]])
            
            try:
                result = minimize(
                    _objective,
                    params_init,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 2000,
                        'maxfun': 50000,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                    }
                )
                
                if result is not None and result.fun < best_s:
                    best_s = result.fun
                    best_params = result.x
                    best_centers = result.x[:24].reshape(12, 2)
                    best_outer_angle = result.x[24]
            except Exception:
                pass
        
        # Also try Nelder-Mead from the best L-BFGS-B result if we have one
        if best_params is not None:
            elapsed = time.time() - _START_TIME
            if elapsed < _DEADLINE - _SAFETY_MARGIN:
                try:
                    result2 = minimize(
                        _objective,
                        best_params,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 3000,
                            'maxfev': 30000,
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True,
                        }
                    )
                    if result2 is not None and result2.fun < best_s:
                        best_s = result2.fun
                        best_params = result2.x
                        best_centers = result2.x[:24].reshape(12, 2)
                        best_outer_angle = result2.x[24]
                except Exception:
                    pass
    
    # If optimization failed, fall back to the initial grid
    if best_centers is None:
        best_centers = initial_configs[0][0]
        best_outer_angle = initial_configs[0][1]
    
    # Center the configuration at origin
    centroid = np.mean(best_centers, axis=0)
    best_centers = best_centers - centroid
    
    # Compute final s
    s_final = _min_s_support_fast(best_centers, best_outer_angle)
    
    # Verify separation (SAT check)
    angles_deg = np.zeros(12)
    min_gap = _pairwise_min_distance(best_centers, angles_deg)
    
    if min_gap < -1e-6:
        # Overlap detected! Fall back to a known-valid configuration
        sqrt3 = np.sqrt(3.0)
        # Use the hexagonal lattice which is guaranteed valid
        pts_fb = np.array([[0.0, 0.0]])
        for i in range(6):
            ang = np.deg2rad(60.0 * i)
            pts_fb = np.vstack([pts_fb, [2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)]])
        for i in range(5):
            ang = np.deg2rad(30.0 + 60.0 * i)
            pts_fb = np.vstack([pts_fb, [2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)]])
        centroid_fb = np.mean(pts_fb, axis=0)
        best_centers = pts_fb - centroid_fb
        best_outer_angle = 0.0
        s_final = _min_s_support_fast(best_centers, best_outer_angle)
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    outer_hex_side_length = s_final
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
