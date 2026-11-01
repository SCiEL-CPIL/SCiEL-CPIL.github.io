import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def hex_vertices(cx, cy, side, angle_rad):
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def outer_normals(angle_deg):
    """Unit outward normals of a regular hexagon with given rotation (degrees)."""
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        ang = a + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    return normals

def required_side_exact(inner_hexes, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner_hexes: list of (x, y, angle_deg)
    Returns (side_length, D) where D is max projection distance.
    """
    ox, oy = outer_center
    normals = outer_normals(outer_angle_deg)
    max_proj = 0.0
    for (ix, iy, iang) in inner_hexes:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        for nx, ny in normals:
            projs = (verts[:, 0] - ox) * nx + (verts[:, 1] - oy) * ny
            mp = max(np.max(projs), -np.min(projs))
            if mp > max_proj:
                max_proj = mp
    S = (max_proj + APOTHEM) / APOTHEM
    return S, max_proj

def required_side_smooth(inner_hexes, outer_center, outer_angle_deg, temperature=0.1):
    """Smooth surrogate for required side length using log-sum-exp."""
    ox, oy = outer_center
    normals = outer_normals(outer_angle_deg)
    all_projs = []
    for (ix, iy, iang) in inner_hexes:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        for nx, ny in normals:
            projs = (verts[:, 0] - ox) * nx + (verts[:, 1] - oy) * ny
            all_projs.extend(projs.tolist())
            all_projs.extend((-projs).tolist())
    
    all_projs = np.array(all_projs)
    if len(all_projs) == 0:
        return 0.0
    
    max_val = np.max(all_projs)
    if max_val > 1000:
        return 1e6
    
    shifted = (all_projs - max_val) / temperature
    log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
    D = temperature * log_sum
    S = (D + APOTHEM) / APOTHEM
    return S

def sat_overlap(v1, v2, eps=1e-7):
    """SAT overlap check for two convex polygons (numpy arrays of shape (6,2))."""
    axes = []
    for poly in (v1, v2):
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            ln = np.linalg.norm(n)
            if ln > 1e-12:
                axes.append(n / ln)
    
    for ax in axes:
        proj1 = v1 @ ax
        proj2 = v2 @ ax
        if np.max(proj1) < np.min(proj2) + eps or np.max(proj2) < np.min(proj1) + eps:
            return False
    return True

def check_disjoint(inner_hexes, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    polys = [hex_vertices(x, y, 1.0, math.radians(a)) for (x, y, a) in inner_hexes]
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def overlap_penalty(inner_hexes, min_dist=SQRT3 - 0.001):
    """Quadratic penalty when center distances fall below min_dist."""
    pen = 0.0
    n = len(inner_hexes)
    for i in range(n):
        xi, yi, _ = inner_hexes[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner_hexes[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen

def axial_to_cartesian(axial_list, spacing=1.0):
    pts = []
    for q, r in axial_list:
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        pts.append((x, y))
    return np.array(pts)

def build_inner_from_axial(axial_list, spacing=1.0, rot_offsets=None):
    """Build list of (x, y, angle_deg) from axial coordinates."""
    pts = axial_to_cartesian(axial_list, spacing)
    inner = []
    for i, (x, y) in enumerate(pts):
        angle = 0.0
        if rot_offsets is not None and i < len(rot_offsets):
            angle = rot_offsets[i]
        inner.append([float(x), float(y), float(angle)])
    return inner

def candidate_topologies():
    """Generate candidate axial coordinate sets for 11 hexagons."""
    candidates = []
    
    # Hexagonal ring + center (7) + 4 extensions
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_1)
    
    # Compact cluster
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (1, 2), (-1, 3)
    ]
    candidates.append(axial_2)
    
    # Elongated
    axial_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_3)
    
    # Diamond
    axial_4 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_4)
    
    # T-shape
    axial_5 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_5)
    
    # L-shape
    axial_6 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_6)
    
    # Cross
    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)
    
    # 3x3 minus corners
    axial_8 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_8)
    
    # Staircase
    axial_9 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (2, 3), (3, 2),
        (3, 1)
    ]
    candidates.append(axial_9)
    
    # Wide flat
    axial_10 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (2, 2)
    ]
    candidates.append(axial_10)
    
    # Compact with offset
    axial_11 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_11)
    
    # Ring variant
    axial_12 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_12)
    
    return candidates

def optimize_rigid_cluster(axial_list, deadline):
    """
    Optimize a rigid cluster: all hexagons share the same orientation,
    optimize global rotation, translation, and outer hexagon parameters.
    Returns (best_S, best_inner, best_center, best_outer_angle).
    """
    base_pts = axial_to_cartesian(axial_list, 1.0)
    cx0 = float(np.mean(base_pts[:, 0]))
    cy0 = float(np.mean(base_pts[:, 1]))
    
    best_S = float('inf')
    best_inner = None
    best_center = None
    best_outer_angle = 0.0
    
    def objective(params, temp=0.05):
        """params = [comp_angle_deg, dx, dy, ox, oy, outer_angle_deg]"""
        if time.time() > deadline:
            return 1e6
        comp_angle = params[0]
        dx, dy = params[1], params[2]
        ox, oy = params[3], params[4]
        outer_angle = params[5]
        
        inner = []
        for (bx, by) in base_pts:
            x = bx + dx
            y = by + dy
            inner.append([float(x), float(y), float(comp_angle)])
        
        S = required_side_smooth(inner, (ox, oy), outer_angle, temperature=temp)
        return S
    
    def objective_exact(params):
        comp_angle = params[0]
        dx, dy = params[1], params[2]
        ox, oy = params[3], params[4]
        outer_angle = params[5]
        
        inner = []
        for (bx, by) in base_pts:
            x = bx + dx
            y = by + dy
            inner.append([float(x), float(y), float(comp_angle)])
        
        S, _ = required_side_exact(inner, (ox, oy), outer_angle)
        return S
    
    # Multi-start optimization
    starts = []
    for comp_a in [0.0, 15.0, 30.0, 45.0, 60.0]:
        for ox_off in [-0.5, 0.0, 0.5]:
            for oy_off in [-0.5, 0.0, 0.5]:
                starts.append([comp_a, 0.0, 0.0, cx0 + ox_off, cy0 + oy_off, comp_a])
    
    # Coarse-to-fine temperature schedule
    temperatures = [0.5, 0.2, 0.1, 0.05, 0.02]
    
    for temp in temperatures:
        if time.time() > deadline:
            break
        
        for start in starts[:8]:  # Limit starts per temperature
            if time.time() > deadline:
                break
            
            try:
                res = minimize(objective, start, method='Nelder-Mead',
                             args=(temp,),
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
                
                if res.fun < best_S and res.fun < 1e5:
                    # Verify with exact computation
                    S_exact, _ = required_side_exact(
                        build_inner_from_axial(axial_list, rot_offsets=[res.x[0]] * N),
                        (res.x[3], res.x[4]), res.x[5]
                    )
                    if S_exact < best_S:
                        best_S = S_exact
                        best_inner = []
                        for (bx, by) in base_pts:
                            x = bx + res.x[1]
                            y = by + res.x[2]
                            best_inner.append([float(x), float(y), float(res.x[0])])
                        best_center = (float(res.x[3]), float(res.x[4]))
                        best_outer_angle = float(res.x[5])
            except Exception:
                pass
    
    # Final polish with exact objective
    if best_inner is not None and time.time() < deadline:
        try:
            init = [best_inner[0][2], 0, 0, best_center[0], best_center[1], best_outer_angle]
            # Refine with exact objective
            res = minimize(objective_exact, init, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_inner = []
                for (bx, by) in base_pts:
                    x = bx + res.x[1]
                    y = by + res.x[2]
                    best_inner.append([float(x), float(y), float(res.x[0])])
                best_center = (float(res.x[3]), float(res.x[4]))
                best_outer_angle = float(res.x[5])
        except Exception:
            pass
    
    return best_S, best_inner, best_center, best_outer_angle

def optimize_with_rotations(axial_list, deadline):
    """
    Optimize with per-hexagon rotation offsets.
    This allows non-uniform orientations for tighter packing.
    """
    base_pts = axial_to_cartesian(axial_list, 1.0)
    cx0 = float(np.mean(base_pts[:, 0]))
    cy0 = float(np.mean(base_pts[:, 1]))
    
    best_S = float('inf')
    best_inner = None
    best_center = None
    best_outer_angle = 0.0
    
    # Start from best rigid solution
    S_rigid, inner_rigid, center_rigid, angle_rigid = optimize_rigid_cluster(axial_list, deadline)
    if inner_rigid is not None:
        best_S = S_rigid
        best_inner = inner_rigid
        best_center = center_rigid
        best_outer_angle = angle_rigid
    
    if time.time() > deadline:
        return best_S, best_inner, best_center, best_outer_angle
    
    # Now try to improve by adding rotation DOFs
    # Parameters: [dx, dy, ox, oy, outer_angle, rot_0, ..., rot_10]
    # Total: 5 + 11 = 16 DOF (too many for direct NM)
    
    # Instead, do coordinate descent on rotations
    current_inner = best_inner[:]
    current_center = best_center
    current_outer = best_outer_angle
    current_S = best_S
    
    # Try perturbing individual rotations
    for iter_round in range(3):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            best_rot = current_inner[i][2]
            best_S_i = current_S
            
            # Try a few rotation values
            for rot_trial in [best_rot, best_rot + 15, best_rot - 15, best_rot + 30, best_rot - 30,
                             0.0, 30.0, 60.0, 90.0, 120.0, 150.0]:
                trial_inner = current_inner[:]
                trial_inner[i] = [current_inner[i][0], current_inner[i][1], rot_trial]
                
                S_trial, _ = required_side_exact(trial_inner, current_center, current_outer)
                # Also check disjointness approximately
                if S_trial < best_S_i - 1e-9:
                    best_S_i = S_trial
                    best_rot = rot_trial
            
            if best_rot != current_inner[i][2]:
                current_inner[i][2] = best_rot
                current_S = best_S_i
                improved = True
        
        if not improved:
            break
    
    if current_S < best_S:
        best_S = current_S
        best_inner = current_inner
        best_center = current_center
        best_outer_angle = current_outer
    
    return best_S, best_inner, best_center, best_outer_angle

def construct():
    deadline = START_TIME + TIME_LIMIT - SAFETY_MARGIN
    
    if time.time() > deadline:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    candidates = candidate_topologies()
    
    best_global_S = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_outer_angle = 0.0
    
    # Phase 1: Fast screen with rigid optimization
    screened = []
    for axial in candidates:
        if time.time() > deadline:
            break
        
        # Quick estimate: bounding box of centers
        pts = axial_to_cartesian(axial, 1.0)
        extent_x = np.max(pts[:, 0]) - np.min(pts[:, 0])
        extent_y = np.max(pts[:, 1]) - np.min(pts[:, 1])
        rough_S = max(extent_x, extent_y) / SQRT3 + 1.0  # Very rough
        
        screened.append((rough_S, axial))
    
    # Sort by rough estimate
    screened.sort(key=lambda x: x[0])
    
    # Phase 2: Optimize top candidates
    for _, axial in screened[:8]:
        if time.time() > deadline:
            break
        
        # Try rigid optimization
        S_rigid, inner_rigid, center_rigid, angle_rigid = optimize_rigid_cluster(axial, deadline)
        
        if inner_rigid is not None and S_rigid < best_global_S:
            best_global_S = S_rigid
            best_global_inner = inner_rigid
            best_global_center = center_rigid
            best_global_outer_angle = angle_rigid
        
        # Try with rotations if time permits
        if time.time() < deadline - 10:
            S_rot, inner_rot, center_rot, angle_rot = optimize_with_rotations(axial, deadline)
            if inner_rot is not None and S_rot < best_global_S:
                best_global_S = S_rot
                best_global_inner = inner_rot
                best_global_center = center_rot
                best_global_outer_angle = angle_rot
    
    # Fallback
    if best_global_inner is None:
        axial = candidates[0]
        best_global_inner = build_inner_from_axial(axial)
        pts = axial_to_cartesian(axial)
        best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_global_outer_angle = 0.0
        best_global_S, _ = required_side_exact(best_global_inner, best_global_center, best_global_outer_angle)
    
    # Verify disjointness and adjust if needed
    if not check_disjoint(best_global_inner, eps=1e-5):
        # Slightly expand spacing to ensure disjointness
        for spacing in [1.001, 1.002, 1.005, 1.01]:
            axial = candidates[0]  # Use first candidate as fallback
            test_inner = build_inner_from_axial(axial, spacing=spacing)
            if check_disjoint(test_inner, eps=1e-5):
                best_global_inner = test_inner
                pts = axial_to_cartesian(axial, spacing)
                best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
                best_global_S, _ = required_side_exact(best_global_inner, best_global_center, best_global_outer_angle)
                break
    
    # Final exact computation
    S_final, D_final = required_side_exact(best_global_inner, best_global_center, best_global_outer_angle)
    
    # Small safety scale
    SCALE = 1.0 + 1e-10
    scaled_inner = []
    cx, cy = best_global_center
    for (x, y, a) in best_global_inner:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        scaled_inner.append([float(sx), float(sy), float(a)])
    
    S_scaled, _ = required_side_exact(scaled_inner, best_global_center, best_global_outer_angle)
    
    return scaled_inner, [float(cx), float(cy)], float(S_scaled), float(best_global_outer_angle)
