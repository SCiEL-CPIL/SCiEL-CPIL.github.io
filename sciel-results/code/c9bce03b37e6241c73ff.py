import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

# Precompute constants
SQRT3 = np.sqrt(3.0)
HEX_ROTATIONS = np.radians([0, 60, 120, 180, 240, 300])
HEX_NORMALS_BASE = np.radians([30, 90, 150, 210, 270, 330])

def _get_hex_vertices_rotations(rot_deg):
    """
    Returns a (12, 6, 2) array of vertices for 12 hexagons, 
    assuming they are centered at origin, with the specified rotations.
    """
    # rot_deg is shape (12,)
    angles = HEX_ROTATIONS[None, :] + rot_deg[:, None]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    # Vertices are (1,0) rotated by angle
    # v_i = (cos(a), sin(a))
    verts = np.stack([cos_a, sin_a], axis=-1) # (12, 6, 2)
    return verts

def _get_hex_normals_rotations(rot_deg):
    """
    Returns a (12, 6, 2) array of outward normals for 12 hexagons.
    """
    angles = HEX_NORMALS_BASE[None, :] + rot_deg[:, None]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    normals = np.stack([cos_a, sin_a], axis=-1) # (12, 6, 2)
    return normals

def _compute_pairwise_distances(pos, rots):
    """
    pos: (12, 2) centers
    rots: (12,) rotations in degrees
    
    Returns:
        min_dist: (66,) array of minimum separation distances.
                  Negative means overlap (penetration depth).
        min_dist_idx: (66,) indices of the separating axis that gave the min dist.
                      (Not strictly needed for distance, but useful for debugging)
    """
    n = len(pos)
    # Get vertices relative to local origin
    # We need vertices in global coords: v_global = v_local + center
    # v_local depends on rotation.
    
    # Efficiently compute all vertices in global space? 
    # verts_global[i] = verts_local[i] + pos[i]
    # verts_local[i] is (6,2)
    
    verts_local = _get_hex_vertices_rotations(rots) # (12, 6, 2)
    # Broadcast centers
    verts_global = verts_local + pos[:, None, :] # (12, 6, 2)
    
    normals = _get_hex_normals_rotations(rots) # (12, 6, 2)
    
    min_dists = np.full((n * (n - 1) // 2), np.inf)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Check all 12 axes (6 from i, 6 from j)
            # Axis from i: normals[i]
            # Axis from j: normals[j]
            
            # Project vertices of i and j onto normals[i]
            # proj_i = verts_global[i] @ normals[i].T -> (6,)
            # proj_j = verts_global[j] @ normals[i].T -> (6,)
            
            # To vectorize across pairs, we would need to expand dimensions.
            # Given n=12, 66 pairs. 
            # Let's stick to a semi-vectorized loop for clarity and moderate speed, 
            # but optimize the inner projection.
            
            d_min = np.inf
            
            # Axes from hexagon i
            for k in range(6):
                n_vec = normals[i, k]
                p_i = verts_global[i] @ n_vec
                p_j = verts_global[j] @ n_vec
                min_i, max_i = np.min(p_i), np.max(p_i)
                min_j, max_j = np.min(p_j), np.max(p_j)
                
                if max_i < min_j:
                    d = min_j - max_i
                elif max_j < min_i:
                    d = min_i - max_j
                else:
                    d = 0.0 # Overlap on this axis, but we need min overlap
                    # Actually, for SAT, if they overlap on this axis, 
                    # the "separation" is negative penetration.
                    # Penetration = min(max_i, max_j) - max(min_i, min_j)
                    pen = min(max_i, max_j) - max(min_i, min_j)
                    # We want the maximum separation (most positive) or minimum penetration (least negative)?
                    # SAT: If ANY axis has positive separation, they are disjoint.
                    # We want to know the "gap". If disjoint, gap > 0. If overlapping, gap < 0 (penetration).
                    # To minimize overlap, we want to maximize the gap (make it positive).
                    # The "distance" in SAT context for optimization is usually the minimum over axes of the separation.
                    # If separated: sep = min_j - max_i (positive).
                    # If overlapping: sep = -(min(max_i, max_j) - max(min_i, min_j)) (negative).
                    
                    if d_min > pen: # pen is negative
                         d_min = pen
                         # Wait, if d_min was inf, and pen is negative, we update.
                         # If d_min was positive (disjoint on previous axis), we should stop?
                         # NO. If they are disjoint on ANY axis, they are disjoint. 
                         # But we are calculating the "worst-case" axis for the distance metric?
                         # Standard SAT for distance: 
                         # If disjoint on axis k, distance is sep_k.
                         # If overlapping on axis k, distance is -pen_k.
                         # The true distance is the MAX over all axes of sep_k (where sep is positive if disjoint).
                         # Wait, no. 
                         # For convex polygons, the minimum translation vector to separate them is along the axis that gives the minimum penetration.
                         # For distance: 
                         # If disjoint, distance = min over axes of separation? No.
                         # If disjoint, distance is the minimum distance between any two points.
                         # SAT gives the minimum translation to separate.
                         # If they are disjoint, the "penetration" is negative.
                         # We define gap(d) such that d > 0 means disjoint, d < 0 means overlap.
                         # For each axis:
                         #   if disjoint: gap_axis = separation > 0
                         #   if overlap: gap_axis = -penetration < 0
                         # The "SAT distance" is the MAX of gap_axis over all axes?
                         # No. If they are disjoint on axis 1 (gap=1) and axis 2 (gap=2), they are disjoint.
                         # The actual geometric distance is not simply the max of these projections.
                         # However, for the purpose of a "soft" constraint in optimization, 
                         # we usually use the minimum penetration depth.
                         # If we define f = -min_gap (where min_gap is the most negative gap, i.e. deepest penetration),
                         # then minimizing f minimizes the deepest overlap.
                         
                         # Let's refine:
                         # For a specific axis, if disjoint, sep > 0. If overlap, pen < 0 (let's call it -pen).
                         # We want to find if they are disjoint. They are disjoint if ANY sep > 0.
                         # To measure "how much they overlap", we look at the axes where they overlap.
                         # The penetration depth is the MINIMUM of the penetrations over the overlapping axes.
                         # But if they are disjoint on some axis, the penetration is effectively -infinity? No.
                         
                         # Standard approach for "smooth" SAT distance:
                         # Calculate separation on all 12 axes.
                         # If any separation > 0, the polygons are disjoint. Distance is positive.
                         # To get a continuous value, we can take the log-sum-exp of the separations?
                         # Or simply: 
                         # If max_separation > 0, distance = max_separation (approx).
                         # Else, distance = min_separation (negative penetration).
                         
                         # Let's just track the max separation.
                         # If max_sep > 0, they are disjoint.
                         # If max_sep < 0, they are overlapping, and max_sep is the least negative (closest to disjoint).
                         
                         if max_i < min_j:
                             sep = min_j - max_i
                         elif max_j < min_i:
                             sep = min_i - max_j
                         else:
                             sep = min(max_i, max_j) - max(min_i, min_j) # Negative
                         
                         if sep > d_min:
                             d_min = sep
                             
            min_dists[idx] = d_min
            idx += 1
            
    return min_dists

def _outer_hex_side_fast(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    # Apothem = side * sqrt(3)/2 => side = 2/sqrt(3) * apothem
    return (2.0 / SQRT3) * max_proj

def _make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    pts = []
    spacing_x = SQRT3 + gap
    spacing_y = 1.5 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_ring(gap=0.0):
    pts = []
    d1 = SQRT3 + gap
    d2 = 2 * SQRT3 + gap
    
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _objective(p, t0, deadline, penalty_weight=100.0, softness=10.0):
    if time.time() - t0 > deadline:
        return 1e6
    
    inner = p[:36].reshape(12, 3)
    ox = p[36]
    oy = p[37]
    oang = p[38]
    
    pos = inner[:, :2]
    rots = inner[:, 2]
    
    # Compute pairwise distances
    dists = _compute_pairwise_distances(pos, rots)
    
    # Penalty for overlaps
    # We want to push dists > 0.
    # Penalty = sum( max(0, -dist)^2 ) or smooth version.
    neg_dists = -dists
    # Smooth max/sum of violations
    # LSE approximation of sum(max(0, x)^2)
    # Let's use a simple quadratic penalty for negative distances
    violations = np.maximum(0, -dists)
    penalty = np.sum(violations**2) * penalty_weight
    
    # Outer size
    verts_list = []
    for i in range(12):
        cx, cy, rot = inner[i]
        v = _get_hex_vertices_rotations(np.array([rot]))[0] + np.array([cx, cy])
        verts_list.append(v)
    all_pts = np.vstack(verts_list)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    
    L = _outer_hex_side_fast(all_pts, oang)
    
    return L + penalty

def _check_disjoint(inner, tol=1e-7):
    pos = inner[:, :2]
    rots = inner[:, 2]
    dists = _compute_pairwise_distances(pos, rots)
    return np.min(dists) > -tol

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    # Generate diverse initial configurations
    configs = []
    gap = 0.1
    configs.append(_make_hex_grid(4, 3, gap=gap))
    configs.append(_make_hex_grid(3, 4, gap=gap))
    configs.append(_make_hex_grid(2, 6, gap=gap))
    configs.append(_make_hex_grid(6, 2, gap=gap))
    configs.append(_make_ring(gap=gap))
    configs.append(_make_hex_grid(4, 3, rot_offset=30, gap=gap))
    configs.append(_make_hex_grid(3, 4, rot_offset=30, gap=gap))
    
    # Random starts
    np.random.seed(42)
    for _ in range(10):
        pts = np.random.randn(12, 3) * 3.0
        pts[:, 2] = np.random.uniform(0, 60, 12)
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        configs.append(pts)
        
    best_L = float('inf')
    best_params = None
    
    # Optimization Loop
    for cfg in configs:
        if time_left() < 20:
            break
            
        # Try a few outer angles
        for oang in [0.0, 30.0, 15.0, 45.0]:
            if time_left() < 15:
                break
                
            init_params = np.concatenate((cfg.flatten(), [0.0, 0.0, oang]))
            
            # 1. Stochastic Local Search to find a feasible region
            # We will perturb parameters and accept if penalty decreases or L decreases (if feasible)
            current_params = init_params.copy()
            current_cost = _objective(current_params, t0, DEADLINE)
            current_L = 0
            current_pen = 0
            
            # Evaluate components
            inner_c = current_params[:36].reshape(12, 3)
            pos_c = inner_c[:, :2]
            rots_c = inner_c[:, 2]
            dists_c = _compute_pairwise_distances(pos_c, rots_c)
            min_dist_c = np.min(dists_c)
            if min_dist_c > 0:
                verts_list_c = []
                for i in range(12):
                    cx, cy, rot = inner_c[i]
                    v = _get_hex_vertices_rotations(np.array([rot]))[0] + np.array([cx, cy])
                    verts_list_c.append(v)
                all_pts_c = np.vstack(verts_list_c)
                all_pts_c[:, 0] -= current_params[36]
                all_pts_c[:, 1] -= current_params[37]
                current_L = _outer_hex_side_fast(all_pts_c, current_params[38])
                current_pen = 0
            else:
                current_L = 0
                current_pen = np.sum(np.maximum(0, -dists_c)**2)
            
            # Local Search
            step_size = 0.5
            for _ in range(200):
                if time_left() < 10: break
                
                # Perturb
                noise = np.random.randn(39) * step_size
                new_params = current_params + noise
                
                new_cost = _objective(new_params, t0, DEADLINE)
                
                # Accept if cost is lower, or with probability exp(-delta/T)
                # Simple hill climbing first
                if new_cost < current_cost:
                    current_params = new_params
                    current_cost = new_cost
                    
                    # Update components
                    inner_c = current_params[:36].reshape(12, 3)
                    pos_c = inner_c[:, :2]
                    rots_c = inner_c[:, 2]
                    dists_c = _compute_pairwise_distances(pos_c, rots_c)
                    min_dist_c = np.min(dists_c)
                    if min_dist_c > 0:
                        verts_list_c = []
                        for i in range(12):
                            cx, cy, rot = inner_c[i]
                            v = _get_hex_vertices_rotations(np.array([rot]))[0] + np.array([cx, cy])
                            verts_list_c.append(v)
                        all_pts_c = np.vstack(verts_list_c)
                        all_pts_c[:, 0] -= current_params[36]
                        all_pts_c[:, 1] -= current_params[37]
                        current_L = _outer_hex_side_fast(all_pts_c, current_params[38])
                        current_pen = 0
                    else:
                        current_L = 0
                        current_pen = np.sum(np.maximum(0, -dists_c)**2)
                    
                    # If feasible, check if it's the best
                    if current_pen == 0 and current_L < best_L:
                        best_L = current_L
                        best_params = current_params.copy()
                        
                # Decrease step size
                step_size *= 0.95
                
            # 2. Polish with L-BFGS-B
            if time_left() < 10: break
            
            try:
                res = minimize(
                    _objective,
                    current_params,
                    args=(t0, DEADLINE - 5),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 100,
                        'ftol': 1e-8,
                        'gtol': 1e-6,
                        'maxfun': 500
                    }
                )
                
                # Check validity of result
                inner_r = res.x[:36].reshape(12, 3)
                if _check_disjoint(inner_r):
                    ox = res.x[36]
                    oy = res.x[37]
                    oang_val = res.x[38]
                    verts_list_r = []
                    for i in range(12):
                        cx, cy, rot = inner_r[i]
                        v = _get_hex_vertices_rotations(np.array([rot]))[0] + np.array([cx, cy])
                        verts_list_r.append(v)
                    all_pts_r = np.vstack(verts_list_r)
                    all_pts_r[:, 0] -= ox
                    all_pts_r[:, 1] -= oy
                    L_r = _outer_hex_side_fast(all_pts_r, oang_val)
                    
                    if L_r < best_L:
                        best_L = L_r
                        best_params = res.x.copy()
                        
            except Exception:
                pass
                
    # Final refinement
    if best_params is not None and time_left() > 10:
        try:
            res = minimize(
                _objective,
                best_params,
                args=(t0, DEADLINE - 2),
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxfun': 1000
                }
            )
            inner_f = res.x[:36].reshape(12, 3)
            if _check_disjoint(inner_f):
                ox = res.x[36]
                oy = res.x[37]
                oang_val = res.x[38]
                verts_list_f = []
                for i in range(12):
                    cx, cy, rot = inner_f[i]
                    v = _get_hex_vertices_rotations(np.array([rot]))[0] + np.array([cx, cy])
                    verts_list_f.append(v)
                all_pts_f = np.vstack(verts_list_f)
                all_pts_f[:, 0] -= ox
                all_pts_f[:, 1] -= oy
                L_f = _outer_hex_side_fast(all_pts_f, oang_val)
                
                if L_f < best_L:
                    best_L = L_f
                    best_params = res.x.copy()
        except Exception:
            pass

    # Fallback
    if best_params is None:
        cfg = _make_hex_grid(4, 3, gap=0.5)
        best_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_get_hex_vertices_rotations(np.array([p[2]]))[0] + np.array([p[0], p[1]]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle

    inner = []
    for i in range(12):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        r = best_params[i * 3 + 2]
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (float(best_params[36]), float(best_params[37]))
    outer_angle = float(best_params[38])
    
    # Final validity check
    if not _check_disjoint(inner):
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_get_hex_vertices_rotations(np.array([p[2]]))[0] + np.array([p[0], p[1]]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle

    all_pts = np.vstack([_get_hex_vertices_rotations(np.array([p[2]]))[0] + np.array([p[0], p[1]]) for p in inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side_fast(all_pts, outer_angle)
    outer_side *= 1.0001
    
    return inner, outer_center, outer_side, outer_angle
