import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + np.cos(theta)
        y = cy + np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def _get_outer_normals(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        theta = angle_rad + np.pi/6 + i * np.pi / 3
        normals.append((np.cos(theta), np.sin(theta)))
    return np.array(normals)

def _min_enclosing_hex_side(composite_verts, outer_angle_deg, outer_center):
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = composite_verts - outer_center
    dots = rel_verts @ normals.T
    max_proj = np.max(np.abs(dots))
    return 2 * max_proj / np.sqrt(3)

def _compute_composite_hull(centers, angles):
    all_verts = []
    for i, (cx, cy) in enumerate(centers):
        verts = _hex_vertices(cx, cy, angles[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    try:
        hull = ConvexHull(all_verts)
        hull_verts = all_verts[hull.vertices]
        return hull_verts
    except Exception:
        return all_verts

def _hexagon_distance_field(cx1, cy1, a1, cx2, cy2, a2):
    """
    Compute minimum distance between two unit hexagons (positive if disjoint, negative if overlapping).
    Uses vertex-to-edge and edge-to-edge distance checks.
    """
    v1 = _hex_vertices(cx1, cy1, a1)
    v2 = _hex_vertices(cx2, cy2, a2)
    
    min_dist = float('inf')
    
    # Check each vertex of hex1 against each edge of hex2
    for vi in v1:
        for j in range(6):
            vj = v2[j]
            vj1 = v2[(j+1)%6]
            d = np.linalg.norm(vj1 - vj)
            if d < 1e-12:
                dist = np.linalg.norm(vi - vj)
            else:
                t = np.clip(np.dot(vi - vj, vj1 - vj) / (d * d), 0, 1)
                closest = vj + t * (vj1 - vj)
                dist = np.linalg.norm(vi - closest)
            min_dist = min(min_dist, dist)
    
    # Check each vertex of hex2 against each edge of hex1
    for vi in v2:
        for j in range(6):
            vj = v1[j]
            vj1 = v1[(j+1)%6]
            d = np.linalg.norm(vj1 - vj)
            if d < 1e-12:
                dist = np.linalg.norm(vi - vj)
            else:
                t = np.clip(np.dot(vi - vj, vj1 - vj) / (d * d), 0, 1)
                closest = vj + t * (vj1 - vj)
                dist = np.linalg.norm(vi - closest)
            min_dist = min(min_dist, dist)
    
    # Check edge-edge distances
    for i1 in range(6):
        p1 = v1[i1]
        p2 = v1[(i1+1)%6]
        for i2 in range(6):
            q1 = v2[i2]
            q2 = v2[(i2+1)%6]
            r = p2 - p1
            s = q2 - q1
            r1 = p1 - q1
            a = np.dot(r, r)
            e = np.dot(s, s)
            f = np.dot(s, r1)
            
            if a < 1e-12 and e < 1e-12:
                dist = np.linalg.norm(r1)
            elif a < 1e-12:
                t = np.clip(f / e, 0, 1)
                dist = np.linalg.norm(p1 - (q1 + t * s))
            elif e < 1e-12:
                t = np.clip(-np.dot(r, r1) / a, 0, 1)
                dist = np.linalg.norm(p1 + t * r - q1)
            else:
                c = np.dot(r, r1)
                d = np.dot(r, s)
                denom = a * e - d * d
                if denom < 1e-12:
                    t = 0.0
                    u = np.clip(f / e, 0, 1)
                else:
                    t = np.clip((d * f - c * e) / denom, 0, 1)
                    u = np.clip((a * f - d * c) / denom, 0, 1)
                closest1 = p1 + t * r
                closest2 = q1 + u * s
                dist = np.linalg.norm(closest1 - closest2)
            min_dist = min(min_dist, dist)
    
    return min_dist

def _overlap_penalty(centers, angles, beta=50.0):
    """
    Smooth penalty for overlapping hexagons using distance field.
    Returns 0 if all disjoint, positive penalty if overlapping.
    """
    n = len(centers)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dist = _hexagon_distance_field(
                centers[i][0], centers[i][1], angles[i],
                centers[j][0], centers[j][1], angles[j]
            )
            if dist < 0:
                penalty += np.exp(-beta * dist) - 1.0
    return penalty

def _objective_inner_only(vars, angles):
    """
    Objective for inner positions only. The outer hexagon is centered at origin
    with angle 0. This reduces dimensionality and smoothness issues.
    vars: [cx1, cy1, cx2, cy2, ..., cx11, cy11]
    """
    n_inner = len(angles)
    centers = []
    for i in range(n_inner):
        cx = vars[2*i]
        cy = vars[2*i + 1]
        centers.append([cx, cy])
    centers = np.array(centers)
    
    composite = _compute_composite_hull(centers, angles)
    # Outer hexagon centered at origin, angle 0
    S = _min_enclosing_hex_side(composite, 0.0, np.array([0.0, 0.0]))
    
    # Overlap penalty
    pen = _overlap_penalty(centers, angles)
    
    return S + 100.0 * pen

def _golden_section_1d(f, a, b, tol=1e-10, max_iter=200):
    """
    Golden section search for minimum of f on [a, b].
    """
    gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate ~0.618
    
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    
    fc = f(c)
    fd = f(d)
    
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
    
    return (a + b) / 2, min(fc, fd)

def _objective_outer_only(outer_angle_deg, centers, angles):
    """
    Objective for outer hexagon angle only, with center at centroid of inner cluster.
    """
    outer_center = np.mean(centers, axis=0)
    composite = _compute_composite_hull(centers, angles)
    S = _min_enclosing_hex_side(composite, outer_angle_deg, outer_center)
    return S

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    eps = 1e-9
    centers_list = []
    angles_list = []
    
    # Center hexagon
    centers_list.append([0.0, 0.0])
    angles_list.append(30.0)
    
    # Inner ring: 6 hexagons at distance sqrt(3) from center
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(30.0)
    
    # Outer ring: 4 hexagons at distance 3 from center, at 30, 90, 150, 210 degrees
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        angles_list.append(30.0)
        
    centers = np.array(centers_list)
    angles = np.array(angles_list)
    n_inner = len(centers)
    
    best_S = None
    best_centers = centers.copy()
    best_outer_angle = 0.0
    best_outer_center = np.array([0.0, 0.0])
    
    # Phase 1: Optimize inner positions only (outer at origin, angle 0)
    # This is a 22-dimensional problem (11 hexagons * 2 coords)
    x0_inner = np.zeros(2 * n_inner)
    for i in range(n_inner):
        x0_inner[2*i] = centers[i][0]
        x0_inner[2*i + 1] = centers[i][1]
    
    inner_best_S = None
    inner_best_vars = x0_inner.copy()
    
    try:
        res = minimize(_objective_inner_only, x0_inner, args=(angles,), 
                       method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if time.time() < DEADLINE - 40.0:
            new_centers = np.array([[res.x[2*i], res.x[2*i+1]] for i in range(n_inner)])
            pen = _overlap_penalty(new_centers, angles)
            if pen < 1e-6:
                composite = _compute_composite_hull(new_centers, angles)
                S_trial = _min_enclosing_hex_side(composite, 0.0, np.array([0.0, 0.0]))
                if inner_best_S is None or S_trial < inner_best_S:
                    inner_best_S = S_trial
                    inner_best_vars = res.x.copy()
    except Exception:
        pass
    
    # Use best inner positions
    if inner_best_S is not None:
        best_centers = np.array([[inner_best_vars[2*i], inner_best_vars[2*i+1]] for i in range(n_inner)])
        best_S = inner_best_S
    else:
        best_S = _min_enclosing_hex_side(_compute_composite_hull(centers, angles), 0.0, np.array([0.0, 0.0]))
    
    # Phase 2: 1D golden-section search over outer rotation angle
    # The outer hexagon is centered at the centroid of the inner cluster
    outer_center = np.mean(best_centers, axis=0)
    
    # Define objective for golden section search
    def angle_obj(theta_deg):
        return _objective_outer_only(theta_deg, best_centers, angles)
    
    # Search over [0, 60) degrees (hexagon has 6-fold symmetry)
    best_angle, best_angle_S = _golden_section_1d(angle_obj, 0.0, 60.0, tol=1e-12, max_iter=100)
    
    if best_angle_S < best_S:
        best_S = best_angle_S
        best_outer_angle = best_angle
    
    # Phase 3: Fine-tune outer center position with Nelder-Mead
    # Optimize [outer_cx, outer_cy] with fixed angle
    def outer_center_obj(vars):
        ocx, ocy = vars
        composite = _compute_composite_hull(best_centers, angles)
        return _min_enclosing_hex_side(composite, best_outer_angle, np.array([ocx, ocy]))
    
    x0_outer_center = np.array([outer_center[0], outer_center[1]])
    
    try:
        res = minimize(outer_center_obj, x0_outer_center, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_S:
            best_S = res.fun
            best_outer_center = res.x
    except Exception:
        pass
    
    # Phase 4: Joint polish - optimize outer center and angle together
    def outer_joint_obj(vars):
        ocx, ocy, oangle = vars
        composite = _compute_composite_hull(best_centers, angles)
        return _min_enclosing_hex_side(composite, oangle, np.array([ocx, ocy]))
    
    x0_joint = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    try:
        res = minimize(outer_joint_obj, x0_joint, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_S:
            best_S = res.fun
            best_outer_center = res.x[:2]
            best_outer_angle = res.x[2]
    except Exception:
        pass
    
    # Final: Recompute exact side length
    composite = _compute_composite_hull(best_centers, angles)
    final_S = _min_enclosing_hex_side(composite, best_outer_angle, best_outer_center)
    if final_S < best_S:
        best_S = final_S
    
    # Build output
    inner = []
    for i in range(n_inner):
        inner.append([best_centers[i][0], best_centers[i][1], angles[i]])
    
    outer_center_out = (best_outer_center[0], best_outer_center[1])
    
    return inner, outer_center_out, best_S, best_outer_angle
