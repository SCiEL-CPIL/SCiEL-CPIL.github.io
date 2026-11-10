# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, s, a_deg):
    a_rad = np.deg2rad(a_deg)
    angles = a_rad + np.deg2rad(60 * np.arange(6))
    x = cx + s * np.cos(angles)
    y = cy + s * np.sin(angles)
    return np.column_stack((x, y))

def _check_separation(hex1, hex2):
    """Check if two unit hexagons are separated by SAT on 12 edge normals."""
    def get_normals(vertices):
        normals = np.zeros((6, 2))
        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            normals[i] = n
        return normals

    n1 = get_normals(hex1)
    n2 = get_normals(hex2)
    all_normals = np.vstack([n1, n2])
    norms = np.linalg.norm(all_normals, axis=1, keepdims=True)
    all_normals = all_normals / norms

    for n in all_normals:
        proj1 = hex1 @ n
        proj2 = hex2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False

def _required_outer_side(all_vertices, outer_center, outer_angle_deg):
    """Compute minimum outer hexagon side length to contain all_vertices."""
    a_rad = np.deg2rad(outer_angle_deg)
    normal_angles = a_rad + np.deg2rad(30) + np.deg2rad(60 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    rel_vertices = all_vertices - outer_center
    proj = rel_vertices @ normals.T
    max_proj = np.max(proj)
    if max_proj < 0:
        return 0.0
    S = (2.0 / np.sqrt(3.0)) * max_proj
    return S

def _validate(inner_hex_data, outer_hex_data, side):
    """Full validation check."""
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data)) and np.isfinite(side)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False

    inner_verts = []
    for i in range(11):
        cx, cy, ang = inner_hex_data[i]
        inner_verts.append(_hex_vertices(cx, cy, 1.0, ang))

    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_separation(inner_verts[i], inner_verts[j]):
                return False

    ox, oy, oang = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, side, oang)
    shifted_outer = outer_verts - np.array([ox, oy])
    shifted_inner = []
    for v in inner_verts:
        shifted_inner.append(v - np.array([ox, oy]))

    for iv in shifted_inner:
        for p in iv:
            for k in range(6):
                p1 = shifted_outer[k]
                p2 = shifted_outer[(k + 1) % 6]
                cross_val = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross_val < -1e-6:
                    return False

    for sv in shifted_inner:
        for k in range(6):
            d = np.linalg.norm(sv[(k+1)%6] - sv[k])
            if abs(d - 1.0) > 1e-8:
                return False

    return True

def _build_hex_cluster_11():
    """Build the 1+6+4 hexagon cluster with exact touching geometry."""
    h = np.sqrt(3)
    R_sat = 3.0  # Distance for satellites to touch ring hexagons
    centers = np.array([
        [0.0, 0.0],
        [h, 0.0],
        [h*np.cos(np.deg2rad(60)), h*np.sin(np.deg2rad(60))],
        [h*np.cos(np.deg2rad(120)), h*np.sin(np.deg2rad(120))],
        [h*np.cos(np.deg2rad(180)), h*np.sin(np.deg2rad(180))],
        [h*np.cos(np.deg2rad(240)), h*np.sin(np.deg2rad(240))],
        [h*np.cos(np.deg2rad(300)), h*np.sin(np.deg2rad(300))],
        [R_sat*np.cos(np.deg2rad(30)), R_sat*np.sin(np.deg2rad(30))],
        [R_sat*np.cos(np.deg2rad(150)), R_sat*np.sin(np.deg2rad(150))],
        [R_sat*np.cos(np.deg2rad(210)), R_sat*np.sin(np.deg2rad(210))],
        [R_sat*np.cos(np.deg2rad(330)), R_sat*np.sin(np.deg2rad(330))],
    ])
    return centers

def _build_staggered_cluster_11():
    """Build a staggered row cluster: rows of 4,4,3 with offset."""
    # Row spacing: for unit hexagons, row vertical spacing is h = sqrt(3)
    h = np.sqrt(3)
    # Row 0 (bottom): 4 hexagons at y=0
    # Row 1 (middle): 4 hexagons at y=h, offset by 0.5
    # Row 2 (top): 3 hexagons at y=2h, offset by 0.5
    centers = []
    # Row 0: x = -1.5, -0.5, 0.5, 1.5 at y=0
    for i, x in enumerate([-1.5, -0.5, 0.5, 1.5]):
        centers.append([x, 0.0])
    # Row 1: x = -1.0, 0.0, 1.0, 2.0 at y=h (shifted right to fit 4)
    # Actually for proper staggering: offset by 0.5
    for i, x in enumerate([-1.0, 0.0, 1.0, 2.0]):
        centers.append([x, h])
    # Row 2: x = -0.5, 0.5, 1.5 at y=2h
    for i, x in enumerate([-0.5, 0.5, 1.5]):
        centers.append([x, 2*h])
    
    # Center the cluster
    centroid = np.mean(centers, axis=0)
    centers = np.array(centers) - centroid
    return centers

def _build_compact_cluster_11():
    """Build a more compact 11-hex cluster: 1+6+4 but with tighter geometry."""
    # This is essentially the same as the standard cluster but we can
    # try to optimize the satellite placement
    h = np.sqrt(3)
    # Try placing satellites at slightly different angles
    centers = [
        [0.0, 0.0],
        [h, 0.0],
        [h*np.cos(np.deg2rad(60)), h*np.sin(np.deg2rad(60))],
        [h*np.cos(np.deg2rad(120)), h*np.sin(np.deg2rad(120))],
        [h*np.cos(np.deg2rad(180)), h*np.sin(np.deg2rad(180))],
        [h*np.cos(np.deg2rad(240)), h*np.sin(np.deg2rad(240))],
        [h*np.cos(np.deg2rad(300)), h*np.sin(np.deg2rad(300))],
    ]
    # Add 4 satellites in a more compact arrangement
    # Try placing them at 45-degree angles with optimized radius
    R = 2.8  # Slightly closer than 3.0 to test
    for ang_deg in [45, 135, 225, 315]:
        centers.append([R*np.cos(np.deg2rad(ang_deg)), R*np.sin(np.deg2rad(ang_deg))])
    
    centroid = np.mean(centers, axis=0)
    centers = np.array(centers) - centroid
    return centers

def _optimize_global_transform(centers, deadline):
    """Optimize the global rotation and translation of the cluster to minimize enclosing hexagon."""
    from scipy.optimize import minimize
    
    inner_verts = [_hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0) for i in range(11)]
    
    # Check internal separation
    valid_internal = True
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_separation(inner_verts[i], inner_verts[j]):
                valid_internal = False
                break
        if not valid_internal:
            break
    
    if not valid_internal:
        return None, None, None
    
    all_inner_vertices = np.vstack(inner_verts)
    centroid = np.mean(centers, axis=0)
    
    def obj(params):
        if time.time() > deadline:
            return 1e10
        ox, oy, oang = params
        S = _required_outer_side(all_inner_vertices, np.array([ox, oy]), oang)
        return S
    
    best_S = float('inf')
    best_params = None
    
    # Try multiple starting angles
    for ang0 in np.linspace(0, 60, 13):
        x0 = [centroid[0], centroid[1], float(ang0)]
        try:
            res = minimize(obj, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    if best_params is not None:
        ox, oy, oang = best_params
        # The inner hexagons should be shifted so that the outer center is at (ox, oy)
        # The inner hexagon centers in the final frame are: centers + (ox - centroid[0], oy - centroid[1])
        shift = np.array([ox - centroid[0], oy - centroid[1]])
        shifted_centers = centers + shift
        return shifted_centers, np.array([ox, oy, oang]), best_S
    
    return None, None, None

def _try_local_refinement(centers, outer_hex_data, side, deadline):
    """Attempt to locally refine positions for a tighter packing."""
    # This is a placeholder for more sophisticated local optimization
    # For now, we just return the input
    return centers, outer_hex_data, side

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 630s limit
    
    # Generate candidate configurations
    configs = []
    
    # Config 1: Standard hexagonal cluster (1+6+4)
    configs.append(_build_hex_cluster_11())
    
    # Config 2: Staggered rows
    if time.time() < deadline:
        configs.append(_build_staggered_cluster_11())
    
    # Config 3: Compact variant
    if time.time() < deadline:
        configs.append(_build_compact_cluster_11())
    
    best_S = float('inf')
    best_inner = None
    best_outer = None
    
    for centers in configs:
        if time.time() > deadline:
            break
        
        result = _optimize_global_transform(centers, deadline)
        if result[0] is not None:
            shifted_centers, outer_hex_data, S = result
            if S < best_S:
                best_S = S
                best_inner = shifted_centers.copy()
                best_outer = outer_hex_data.copy()
    
    if best_inner is None:
        # Fallback: use the standard cluster with a generous outer hexagon
        best_inner = _build_hex_cluster_11()
        centroid = np.mean(best_inner, axis=0)
        best_outer = np.array([centroid[0], centroid[1], 0.0])
        inner_verts = [_hex_vertices(best_inner[i, 0], best_inner[i, 1], 1.0, 0.0) for i in range(11)]
        all_inner_vertices = np.vstack(inner_verts)
        best_S = _required_outer_side(all_inner_vertices, np.array([centroid[0], centroid[1]]), 0.0)
    
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [best_inner[i, 0], best_inner[i, 1], 0.0]
    
    outer_hex_data = best_outer
    
    # Validation with fallback
    if not _validate(inner_hex_data, outer_hex_data, best_S):
        best_S += 1e-4
        if not _validate(inner_hex_data, outer_hex_data, best_S):
            best_S += 1e-3
            if not _validate(inner_hex_data, outer_hex_data, best_S):
                # Last resort: very generous outer hexagon
                best_S = 10.0
    
    return inner_hex_data, outer_hex_data, best_S

# EVOLVE-BLOCK-END
