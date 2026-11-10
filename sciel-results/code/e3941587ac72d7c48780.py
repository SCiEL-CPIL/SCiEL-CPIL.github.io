# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon with center (cx,cy), side d, rotation angle_deg."""
    angles = np.deg2rad(angle_deg) + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _all_inner_vertices(inner_hex_data):
    """Compute all 72 vertices of 12 unit hexagons."""
    verts_list = []
    for i in range(12):
        cx, cy, ang = inner_hex_data[i]
        verts_list.append(_hex_vertices(cx, cy, 1.0, ang))
    return np.vstack(verts_list)  # (72, 2)


def _min_s_given(verts, outer_center, outer_angle_deg):
    """
    Given all inner vertices (72,2) and outer hexagon center/angle,
    compute minimum side s of outer hexagon that contains all vertices.
    
    For a regular hexagon with center (ox,oy), side s, angle a:
    vertices at angles a + 60*i. Edge k connects v_k to v_{k+1}.
    The outward normal to edge k is at angle a + 60*k + 30 degrees.
    A point p is inside iff dot(outward_normal_k, p - center) <= s * sqrt(3)/2 for all k.
    """
    rel = verts - outer_center  # (72, 2)
    a = np.deg2rad(outer_angle_deg)
    normal_angles = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)
    
    # For each normal, compute max dot product over all vertices
    # (72, 6) matrix of dot products
    dots = rel @ normals.T  # (72, 6)
    max_dots = np.max(dots, axis=0)  # (6,)
    
    # s >= max_dots[k] / (sqrt(3)/2) for each k
    s_vals = max_dots / (np.sqrt(3.0) / 2.0)
    s = np.max(s_vals)
    return max(s, 0.0)


def _check_separation(inner_hex_data, tol=1e-6):
    """Check that no two inner hexagons overlap using SAT."""
    n = len(inner_hex_data)
    verts_list = []
    for i in range(n):
        cx, cy, ang = inner_hex_data[i]
        verts_list.append(_hex_vertices(cx, cy, 1.0, ang))
    
    for i in range(n):
        for j in range(i + 1, n):
            v1 = verts_list[i]
            v2 = verts_list[j]
            # Get all edge normals from both hexagons
            edges = []
            for k in range(6):
                edges.append(v1[(k+1) % 6] - v1[k])
                edges.append(v2[(k+1) % 6] - v2[k])
            
            for edge in edges:
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                gap1 = min2 - max1
                gap2 = min1 - max2
                if gap1 > tol or gap2 > tol:
                    break
            else:
                # No separating axis found - they overlap
                return False
    return True


def _validate(inner_hex_data, outer_hex_data, s):
    """Full validation against task constraints."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12.0) - 1e-12:
        return False
    
    outer_verts = _hex_vertices(outer_hex_data[0], outer_hex_data[1], s, outer_hex_data[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    oc = outer_hex_data[:2]
    shifted = inner_hex_data.copy()
    shifted[:, 0] -= oc[0]
    shifted[:, 1] -= oc[1]
    
    # Check unit edge lengths
    for i in range(12):
        verts = _hex_vertices(shifted[i, 0], shifted[i, 1], 1.0, shifted[i, 2])
        for k in range(6):
            el = np.linalg.norm(verts[(k+1)%6] - verts[k])
            if abs(el - 1.0) > 1e-8:
                return False
    
    # Check separation
    if not _check_separation(shifted):
        return False
    
    # Check all inner vertices inside outer hexagon
    all_verts = _all_inner_vertices(shifted)
    ov = _hex_vertices(0, 0, s, outer_hex_data[2])
    
    for v in all_verts:
        for k in range(6):
            p1 = ov[k]
            p2 = ov[(k+1) % 6]
            edge = p2 - p1
            to_v = v - p1
            cross_val = edge[0] * to_v[1] - edge[1] * to_v[0]
            if cross_val < -1e-6:
                return False
    
    return True


def _objective(params, verts_func):
    """Objective function for optimization: minimize s."""
    # params: [angle_0..angle_11, outer_cx, outer_cy, outer_angle]
    angles = params[:12]
    outer_cx = params[12]
    outer_cy = params[13]
    outer_angle = params[14]
    
    inner_data = np.column_stack([np.zeros(12), np.zeros(12), angles])
    # But we need to keep the centers fixed, only angles change
    # This is handled by the caller passing appropriate data
    
    verts = verts_func(angles)
    return _min_s_given(verts, [outer_cx, outer_cy], outer_angle)


def _make_verts_func(centers):
    """Create a function that computes all vertices given angles."""
    def func(angles):
        verts_list = []
        for i in range(12):
            cx, cy = centers[i]
            verts_list.append(_hex_vertices(cx, cy, 1.0, angles[i]))
        return np.vstack(verts_list)
    return func


def _generate_hex_cluster():
    """Generate a hexagonal cluster of 12 unit hexagons."""
    sqrt3 = np.sqrt(3.0)
    
    # Axial coordinates for a hexagonal arrangement
    # Center + 6 neighbors + 5 more from second ring
    # For unit hexagons with pointy-top (angle 0), adjacent centers are:
    # distance = 2 * (sqrt(3)/2) * 1 = sqrt(3) apart
    
    # Actually for regular hexagons with side 1, the distance between centers
    # of adjacent hexagons (sharing an edge) is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    
    # Let's use a proper hexagonal grid. For pointy-top hexagons:
    # horizontal spacing = sqrt(3), vertical spacing = 1.5
    # offset rows shifted by sqrt(3)/2
    
    # Better: use axial coordinates
    # For a hexagon with side 1, the six neighbors in axial coords (q,r) are:
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    
    axial_coords = [
        (0, 0),       # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (1, 1), (0, 2), (-2, 0), (-1, -1), (2, -1),  # ring 2 (6 of 12)
    ]
    
    centers = []
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append([x, y])
    
    return np.array(centers)


def _generate_row_arrangement():
    """Generate a 4-4-4 row arrangement."""
    sqrt3 = np.sqrt(3.0)
    centers = []
    # 3 rows of 4 hexagons
    for row in range(3):
        y = row * 1.5
        for col in range(4):
            x = (col - 1.5) * sqrt3
            if row % 2 == 1:
                x += sqrt3 / 2
            centers.append([x, y])
    return np.array(centers)


def _generate_compact_arrangement():
    """Generate a compact 3-4-3-2 arrangement."""
    sqrt3 = np.sqrt(3.0)
    centers = []
    rows = [(0, 3), (1.5, 4), (3.0, 3), (4.5, 2)]
    for y, count in rows:
        x_start = -(count - 1) * sqrt3 / 2.0
        for j in range(count):
            x = x_start + j * sqrt3
            centers.append([x, y])
    return np.array(centers)


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 570.0  # 30s safety margin
    
    sqrt3 = np.sqrt(3.0)
    
    # Generate candidate center arrangements
    candidate_centers = [
        _generate_hex_cluster(),
        _generate_row_arrangement(),
        _generate_compact_arrangement(),
    ]
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    for centers in candidate_centers:
        if time.time() > deadline - 60.0:
            break
        
        # Initial angles: all 0 (pointy-top)
        initial_angles = np.zeros(12)
        
        # Precompute verts function
        verts_func = _make_verts_func(centers)
        
        # Compute initial s with outer hexagon centered at centroid, angle 30 (flat-top)
        centroid = np.mean(centers, axis=0)
        initial_verts = verts_func(initial_angles)
        initial_s = _min_s_given(initial_verts, centroid, 30.0)
        
        if initial_s < best_s:
            best_s = initial_s
            best_inner = np.column_stack([centers[:, 0], centers[:, 1], initial_angles])
            best_outer = np.array([centroid[0], centroid[1], 30.0])
        
        # Optimize: vary inner angles and outer parameters
        # Parameter vector: [angle_0..angle_11, outer_cx, outer_cy, outer_angle]
        def objective(params):
            angles = params[:12]
            ocx = params[12]
            ocy = params[13]
            oang = params[14]
            verts = verts_func(angles)
            return _min_s_given(verts, [ocx, ocy], oang)
        
        x0 = np.concatenate([initial_angles, [centroid[0], centroid[1], 30.0]])
        
        # Use Nelder-Mead for the optimization
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                      'adaptive': True})
            
            if result.fun < best_s:
                new_angles = result.x[:12]
                new_ocx = result.x[12]
                new_ocy = result.x[13]
                new_oang = result.x[14]
                
                # Check separation with new angles
                new_inner = np.column_stack([centers[:, 0], centers[:, 1], new_angles])
                if _check_separation(new_inner):
                    new_outer = np.array([new_ocx, new_ocy, new_oang])
                    best_s = result.fun
                    best_inner = new_inner
                    best_outer = new_outer
        except Exception:
            pass
        
        # Try a few different outer angle values with optimized inner angles
        if time.time() < deadline - 30.0:
            for test_oang in [0, 15, 30, 45, 60]:
                if time.time() > deadline - 30.0:
                    break
                
                def obj_fixed_angle(params):
                    angles = params[:12]
                    ocx = params[12]
                    ocy = params[13]
                    verts = verts_func(angles)
                    return _min_s_given(verts, [ocx, ocy], test_oang)
                
                x0_fixed = np.concatenate([initial_angles, [centroid[0], centroid[1]]])
                
                try:
                    result = minimize(obj_fixed_angle, x0_fixed, method='Nelder-Mead',
                                     options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8,
                                              'adaptive': True})
                    
                    if result.fun < best_s:
                        new_angles = result.x[:12]
                        new_ocx = result.x[12]
                        new_ocy = result.x[13]
                        
                        new_inner = np.column_stack([centers[:, 0], centers[:, 1], new_angles])
                        if _check_separation(new_inner):
                            new_outer = np.array([new_ocx, new_ocy, float(test_oang)])
                            best_s = result.fun
                            best_inner = new_inner
                            best_outer = new_outer
                except Exception:
                    pass
        
        # Local refinement around best found so far
        if best_inner is not None and time.time() < deadline - 15.0:
            # Fine-tune with smaller steps
            step_sizes = [0.5, 0.2, 0.1, 0.05]
            for step in step_sizes:
                if time.time() > deadline - 15.0:
                    break
                improved = True
                while improved and time.time() < deadline - 15.0:
                    improved = False
                    current_inner = best_inner.copy()
                    current_outer = best_outer.copy()
                    
                    # Try perturbing each inner angle
                    for i in range(12):
                        if time.time() > deadline - 15.0:
                            break
                        for delta in [step, -step]:
                            trial_inner = current_inner.copy()
                            trial_inner[i, 2] += delta
                            if _check_separation(trial_inner):
                                trial_verts = _all_inner_vertices(trial_inner)
                                trial_s = _min_s_given(trial_verts, current_outer[:2], current_outer[2])
                                if trial_s < best_s - 1e-10:
                                    best_s = trial_s
                                    best_inner = trial_inner
                                    improved = True
                    
                    # Try perturbing outer center and angle
                    for param_idx, delta in [(0, step), (0, -step), (1, step), (1, -step), (2, step), (2, -step)]:
                        if time.time() > deadline - 15.0:
                            break
                        trial_outer = current_outer.copy()
                        trial_outer[param_idx] += delta
                        trial_verts = _all_inner_vertices(best_inner)
                        trial_s = _min_s_given(trial_verts, trial_outer[:2], trial_outer[2])
                        if trial_s < best_s - 1e-10:
                            best_s = trial_s
                            best_outer = trial_outer
                            improved = True
    
    # Final validation
    if best_inner is not None and best_outer is not None:
        if not _validate(best_inner, best_outer, best_s):
            # Fallback: use a known valid configuration
            # Simple 4-4-4 arrangement with all angles 0
            centers = _generate_row_arrangement()
            angles = np.zeros(12)
            inner = np.column_stack([centers[:, 0], centers[:, 1], angles])
            centroid = np.mean(centers, axis=0)
            outer = np.array([centroid[0], centroid[1], 30.0])
            s = _min_s_given(_all_inner_vertices(inner), centroid, 30.0)
            
            if _validate(inner, outer, s):
                best_inner = inner
                best_outer = outer
                best_s = s
            else:
                # Last resort: very large hexagon
                inner = np.column_stack([centers[:, 0], centers[:, 1], angles])
                outer = np.array([0.0, 0.0, 0.0])
                best_s = 15.0
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
