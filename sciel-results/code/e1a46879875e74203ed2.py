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
    hull = ConvexHull(all_verts)
    hull_verts = all_verts[hull.vertices]
    return hull_verts

def _min_enclosing_hex_side_smooth(composite_verts, outer_angle_deg, outer_center, beta=100.0):
    """Smooth approximation of min enclosing hex side using soft-max."""
    normals = _get_outer_normals(outer_angle_deg)
    rel_verts = composite_verts - outer_center
    dots = rel_verts @ normals.T
    abs_dots = np.abs(dots)
    # Soft max over all projections
    max_val = np.logaddexp.reduce(abs_dots.ravel()) / 1.0  # This isn't quite right
    # Better: use log-sum-exp for soft max
    flat = abs_dots.ravel()
    m = np.max(flat)
    soft_max = m + np.log(np.sum(np.exp(flat - m)))
    return 2 * soft_max / np.sqrt(3)

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
            # Distance from point vi to segment vj-vj1
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
            # Distance between two segments
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
                    if t == 0.0 and u == 0.0:
                        pass  # Already handled by vertex-edge checks
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
                # Overlapping - use smooth penalty
                penalty += np.exp(-beta * dist) - 1.0
    return penalty

def _joint_objective(vars, angles):
    """
    Joint objective: minimize outer hex side + overlap penalty.
    vars: [outer_cx, outer_cy, outer_angle, cx1, cy1, cx2, cy2, ..., cx11, cy11]
    """
    n_inner = len(angles)
    outer_cx = vars[0]
    outer_cy = vars[1]
    outer_angle = vars[2]
    
    # Extract inner centers
    centers = []
    for i in range(n_inner):
        cx = vars[3 + 2*i]
        cy = vars[4 + 2*i]
        centers.append([cx, cy])
    centers = np.array(centers)
    
    # Compute enclosing side
    composite = _compute_composite_hull(centers, angles)
    S = _min_enclosing_hex_side(composite, outer_angle, np.array([outer_cx, outer_cy]))
    
    # Compute overlap penalty
    pen = _overlap_penalty(centers, angles)
    
    return S + 100.0 * pen

def _objective(vars, centers, angles):
    outer_cx, outer_cy, outer_angle = vars
    outer_center = np.array([outer_cx, outer_cy])
    composite = _compute_composite_hull(centers, angles)
    S = _min_enclosing_hex_side(composite, outer_angle, outer_center)
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
    
    # Initial outer hexagon: centered at origin, angle 30
    x0_outer = np.array([0.0, 0.0, 30.0])
    
    best_S = None
    best_vars_outer = x0_outer
    best_centers = centers.copy()
    
    # Phase 1: Optimize outer hexagon with fixed inner positions (Nelder-Mead)
    try:
        res = minimize(_objective, x0_outer, args=(centers, angles), method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
        if time.time() < DEADLINE - 30.0:
            if res.fun < (best_S if best_S is not None else float('inf')):
                best_S = res.fun
                best_vars_outer = res.x
    except Exception:
        pass
    
    # Phase 2: Joint optimization using L-BFGS-B
    # Build initial variable vector: [outer_cx, outer_cy, outer_angle, cx1, cy1, ..., cx11, cy11]
    x0_joint = np.zeros(3 + 2 * n_inner)
    x0_joint[0] = best_vars_outer[0]
    x0_joint[1] = best_vars_outer[1]
    x0_joint[2] = best_vars_outer[2]
    for i in range(n_inner):
        x0_joint[3 + 2*i] = centers[i][0]
        x0_joint[4 + 2*i] = centers[i][1]
    
    # Try joint optimization with different outer angle initializations
    joint_best_S = best_S if best_S is not None else float('inf')
    joint_best_vars = x0_joint.copy()
    
    for init_angle in [0.0, 15.0, 30.0, 45.0, 60.0]:
        if time.time() > DEADLINE - 20.0:
            break
        
        x0_trial = x0_joint.copy()
        x0_trial[2] = init_angle
        
        try:
            res = minimize(_joint_objective, x0_trial, args=(angles,), 
                          method='L-BFGS-B', 
                          options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if time.time() < DEADLINE - 15.0:
                # Extract the actual S (without penalty) from the result
                outer_cx, outer_cy, outer_angle = res.x[0], res.x[1], res.x[2]
                new_centers = np.array([[res.x[3+2*i], res.x[4+2*i]] for i in range(n_inner)])
                composite = _compute_composite_hull(new_centers, angles)
                S_trial = _min_enclosing_hex_side(composite, outer_angle, np.array([outer_cx, outer_cy]))
                pen_trial = _overlap_penalty(new_centers, angles)
                
                # Only accept if no significant overlap
                if pen_trial < 1e-6 and S_trial < joint_best_S:
                    joint_best_S = S_trial
                    joint_best_vars = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Polish with Nelder-Mead on outer hexagon with optimized inner positions
    if joint_best_S < (best_S if best_S is not None else float('inf')):
        best_S = joint_best_S
        best_vars_outer = joint_best_vars[:3].copy()
        best_centers = np.array([[joint_best_vars[3+2*i], joint_best_vars[4+2*i]] for i in range(n_inner)])
    
    # Final polish: Nelder-Mead on outer hexagon with best inner positions
    if time.time() < DEADLINE - 5.0:
        x0_polish = best_vars_outer.copy()
        try:
            res = minimize(_objective, x0_polish, args=(best_centers, angles), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < (best_S if best_S is not None else float('inf')):
                best_S = res.fun
                best_vars_outer = res.x
        except Exception:
            pass
    
    # Final result
    outer_cx, outer_cy, outer_angle = best_vars_outer
    outer_center = (outer_cx, outer_cy)
    
    inner = []
    for i in range(n_inner):
        inner.append([best_centers[i][0], best_centers[i][1], angles[i]])
    
    if best_S is None:
        side = _min_enclosing_hex_side(
            _compute_composite_hull(best_centers, angles), 30.0, np.array([0.0, 0.0]))
    else:
        side = best_S
    
    return inner, outer_center, side, outer_angle
