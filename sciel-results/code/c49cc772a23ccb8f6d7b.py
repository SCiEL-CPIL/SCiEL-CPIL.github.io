import numpy as np
import math
import time
import random
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 140.0


def hex_sdf_dist(px, py, cx, cy, side, rot_deg):
    """
    Signed distance from point (px,py) to regular hexagon centered at (cx,cy).
    Positive = outside, negative = inside.
    """
    theta = math.radians(rot_deg)
    c, s = math.cos(theta), math.sin(theta)
    dx = px - cx
    dy = py - cy
    # Rotate into hexagon's local frame
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    # Hexagon with vertices at angles 0,60,...: local frame has vertices at (side*cos(k*pi/3), side*sin(k*pi/3))
    # The hexagon can be described as intersection of 6 half-planes: n_k . p <= apothem
    # where n_k = (cos(k*pi/3), sin(k*pi/3)), apothem = side * sqrt(3)/2
    apothem = side * math.sqrt(3.0) / 2.0
    # Max over 6 directions of (n_k . p - apothem) gives SDF approximation
    # Actually for a regular hexagon, the SDF is: max_k (n_k . p - apothem) clipped
    # For exact SDF of convex polygon: d = max_k (n_k . p - apothem) if outside
    # But we need to handle the "corner" regions properly
    # For a regular hexagon, the exact SDF is:
    #   d = max(0, max_k (n_k . p - apothem))  -- this is the distance to the supporting lines
    # But this overestimates near corners. The true SDF of a convex polygon is the distance to the boundary.
    # For a regular hexagon, a good approximation that's exact at most points:
    # Use the max of the 6 half-plane distances, but cap it properly.
    
    # Better approach: compute distance to each edge, take min
    # Edge k goes from vertex k to vertex k+1
    # For a regular hexagon, by symmetry we can compute in one sector and use symmetry
    
    # Let's just compute distance to each of the 6 edges
    min_d = float('inf')
    for k in range(6):
        a1 = math.radians(60.0 * k)
        a2 = math.radians(60.0 * (k + 1))
        v1 = (side * math.cos(a1), side * math.sin(a1))
        v2 = (side * math.cos(a2), side * math.sin(a2))
        # Distance from point (lx,ly) to segment v1-v2
        ex = v2[0] - v1[0]
        ey = v2[1] - v1[1]
        px_ = lx - v1[0]
        py_ = ly - v1[1]
        t = (px_ * ex + py_ * ey) / (ex * ex + ey * ey)
        t = max(0.0, min(1.0, t))
        qx = v1[0] + t * ex
        qy = v1[1] + t * ey
        d = math.sqrt((lx - qx)**2 + (ly - qy)**2)
        min_d = min(min_d, d)
    
    # Determine if inside: check if all half-plane constraints satisfied
    inside = True
    for k in range(6):
        a = math.radians(60.0 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        if nx * lx + ny * ly > apothem + 1e-12:
            inside = False
            break
    
    return -min_d if inside else min_d


def smooth_hex_overlap(v1_params, v2_params, beta=10.0):
    """
    Smooth overlap penalty between two hexagons using SDF sampling.
    v1_params, v2_params: (cx, cy, rot_deg)
    Returns smooth approximation of overlap depth.
    """
    cx1, cy1, r1 = v1_params
    cx2, cy2, r2 = v2_params
    
    # Quick distance check
    d_centers = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    if d_centers > 3.0:
        return 0.0
    
    # Sample points on boundary of hexagon 1
    # Use 12 sample points around hexagon 1
    sample_angles = np.array([math.radians(30.0 * k) for k in range(12)])
    points1 = np.column_stack((
        cx1 + math.cos(sample_angles),
        cy1 + math.sin(sample_angles)
    ))
    
    # Also sample some interior points of hexagon 1
    interior_angles = np.array([math.radians(30.0 * k) for k in range(6)])
    interior_points = np.column_stack((
        cx1 + 0.5 * math.cos(interior_angles),
        cy1 + 0.5 * math.sin(interior_angles)
    ))
    
    all_points1 = np.vstack([points1, interior_points])
    
    # Compute SDF distance from these points to hexagon 2
    dists = []
    for px, py in all_points1:
        d = hex_sdf_dist(px, py, cx2, cy2, 1.0, r2)
        if d < 0:
            dists.append(-d)  # penetration depth
    
    if not dists:
        return 0.0
    
    # Smooth max of penetrations
    d_arr = np.array(dists)
    max_d = np.max(d_arr)
    if max_d < 1e-8:
        return 0.0
    
    # Smooth sum of squares
    scaled = d_arr**2 / beta
    max_s = np.max(scaled)
    if max_s > 400:
        return np.sum(d_arr**2)
    lse = max_s + math.log(sum(math.exp(x - max_s) for x in scaled))
    return beta * lse


def pairwise_smooth_overlap(xys_rots, beta=10.0):
    """Compute total smooth overlap penalty for all pairs."""
    centers = xys_rots[:, :2]
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.8:
                total += smooth_hex_overlap(xys_rots[i], xys_rots[j], beta)
    return total


def outer_hex_side_smooth(points, outer_rot_deg, beta=5.0):
    """
    Smooth minimum side length of outer hexagon.
    Uses smooth max instead of hard max.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T  # shape (num_points, 6)
    abs_proj = np.abs(proj)
    
    # Smooth max over all points and 6 directions
    # We want smooth_max of abs_proj
    max_val = np.max(abs_proj)
    
    # For smooth max, use log-sum-exp
    scaled = abs_proj / beta
    max_s = np.max(scaled)
    if max_s > 400:
        smooth_max = np.max(abs_proj)
    else:
        lse = max_s + math.log(np.sum(np.exp(scaled - max_s)))
        smooth_max = beta * lse
    
    # Side = 2 * apothem / sqrt(3)
    return (2.0 / math.sqrt(3.0)) * smooth_max


def is_valid_config(xys_rots, tol=1e-6):
    """Check if configuration has no overlaps using SDF."""
    centers = xys_rots[:, :2]
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.5:
                # Check overlap using SDF sampling
                cx1, cy1, r1 = xys_rots[i]
                cx2, cy2, r2 = xys_rots[j]
                # Sample boundary of hex 1
                for k in range(12):
                    ang = math.radians(30.0 * k)
                    px = cx1 + math.cos(ang)
                    py = cy1 + math.sin(ang)
                    d_sdf = hex_sdf_dist(px, py, cx2, cy2, 1.0, r2)
                    if d_sdf < -tol:
                        return False
                # Sample boundary of hex 2
                for k in range(12):
                    ang = math.radians(30.0 * k)
                    px = cx2 + math.cos(ang)
                    py = cy2 + math.sin(ang)
                    d_sdf = hex_sdf_dist(px, py, cx1, cy1, 1.0, r1)
                    if d_sdf < -tol:
                        return False
    return True


def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with rotation rot_deg."""
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))


def get_L(xys_rots, outer_rot):
    """Compute outer side length for a valid configuration."""
    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
    return outer_hex_side_smooth(all_points, outer_rot, beta=1000.0)  # Very sharp max ≈ hard max


def make_initial_config(style=0, gap=0.05):
    """Generate diverse initial configurations."""
    sqrt3 = math.sqrt(3.0)
    
    if style == 0:
        # Honeycomb: 1 center + 6 ring + 5 outer
        pts = []
        d1 = sqrt3 + gap
        d2 = 2 * sqrt3 + gap
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = math.radians(60 * k)
            pts.append([d1 * math.cos(ang), d1 * math.sin(ang), 0.0])
        for k in range(5):
            ang = math.radians(60 * k + 30)
            pts.append([d2 * math.cos(ang), d2 * math.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    elif style == 1:
        # Ring of 12
        pts = []
        R = 2.5 + gap
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    elif style == 2:
        # 2x6 grid
        pts = []
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5
        for r in range(2):
            y = r * spacing_y
            for c in range(6):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    elif style == 3:
        # 3x4 grid
        pts = []
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5
        for r in range(3):
            y = r * spacing_y
            for c in range(4):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    elif style == 4:
        # Two rows of 6 with offset
        pts = []
        spacing_x = sqrt3 * 1.1 + gap
        spacing_y = 1.5 + gap
        for r in range(2):
            y = r * spacing_y
            for c in range(6):
                x = c * spacing_x + (spacing_x / 2 if r == 1 else 0)
                pts.append([x, y, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    elif style == 5:
        # Rotated honeycomb
        base = make_initial_config(0, gap)
        base[:, 2] = 30.0
        return base
    
    elif style == 6:
        # Compact cluster
        pts = []
        sqrt3 = math.sqrt(3.0)
        d1 = sqrt3 * 1.02
        d2 = 2 * sqrt3 * 1.02
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = math.radians(60 * k)
            pts.append([d1 * math.cos(ang), d1 * math.sin(ang), 0.0])
        for k in range(5):
            ang = math.radians(60 * k + 30)
            pts.append([d2 * math.cos(ang), d2 * math.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    else:
        # Fallback: random
        pts = np.random.randn(N, 3) * 2.0
        pts[:, 2] *= 30
        return pts


def objective(params, beta=15.0):
    """
    Smooth objective: outer side length + overlap penalty.
    params: 3*N + 1 values (x, y, rot for each hexagon, plus outer rotation)
    """
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    
    # Smooth overlap penalty
    penalty = pairwise_smooth_overlap(xys_rots, beta=beta)
    
    # Outer side length (smooth)
    all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
    L = outer_hex_side_smooth(all_points, outer_rot, beta=5.0)
    
    return L + 50.0 * penalty


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def check_time(margin=10.0):
        return time_left() > margin
    
    best_L = float('inf')
    best_params = None
    
    # Phase 1: Evaluate diverse initial configurations
    initial_configs = []
    for style in range(7):
        for gap in [0.05, 0.1, 0.15]:
            cfg = make_initial_config(style, gap)
            initial_configs.append(cfg)
    
    outer_rots = [0.0, 30.0, 15.0, 45.0, 10.0, 20.0]
    
    for cfg in initial_configs:
        if not check_time(50.0):
            break
        
        for orot in outer_rots:
            if not check_time(50.0):
                break
            
            params = np.concatenate((cfg.flatten(), [orot]))
            
            if is_valid_config(params):
                L_val = get_L(params.reshape(N, 3), orot)
                if L_val < best_L:
                    best_L = L_val
                    best_params = params.copy()
    
    # Phase 2: Differential evolution for global search
    if best_params is not None and check_time(30.0):
        try:
            bounds = [(-5.0, 5.0)] * (3 * N) + [(-90.0, 90.0)]
            
            def de_objective(x):
                if not check_time(5.0):
                    return 1e6
                xys = x[:3*N].reshape(N, 3)
                if not is_valid_config(xys, tol=1e-5):
                    return 1e6
                return get_L(xys, x[3*N])
            
            x0 = best_params[:3*N].reshape(N, 3)
            # Perturb initial points for DE
            x0_perturbed = x0 + np.random.randn(N, 3) * 0.3
            x0_perturbed[:, 2] %= 360.0
            
            de_bounds_list = [(-5.0, 5.0)] * (3 * N) + [(-90.0, 90.0)]
            
            result_de = differential_evolution(
                de_objective,
                de_bounds_list,
                x0=x0_perturbed.flatten(),
                maxiter=100,
                popsize=8,
                tol=1e-8,
                seed=42,
                polish=True,
                init='sobol'
            )
            
            if result_de.fun < best_L:
                xys_de = result_de.x[:3*N].reshape(N, 3)
                if is_valid_config(xys_de):
                    L_de = get_L(xys_de, result_de.x[3*N])
                    if L_de < best_L:
                        best_L = L_de
                        best_params = result_de.x.copy()
        except Exception:
            pass
    
    # Phase 3: Gradient-based refinement with smooth objective
    if best_params is not None and check_time(20.0):
        try:
            # Use smooth objective for better gradients
            res_lbgfs = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                args=(15.0,),
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxfun': 1000
                }
            )
            
            if is_valid_config(res_lbgfs.x[:3*N].reshape(N, 3)):
                L_val = get_L(res_lbgfs.x[:3*N].reshape(N, 3), res_lbgfs.x[3*N])
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_lbgfs.x.copy()
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead for final polish
    if best_params is not None and check_time(15.0):
        try:
            res_nm = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                args=(20.0,),
                options={
                    'maxiter': 5000,
                    'xatol': 1e-9,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            
            if is_valid_config(res_nm.x[:3*N].reshape(N, 3)):
                L_val = get_L(res_nm.x[:3*N].reshape(N, 3), res_nm.x[3*N])
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 5: High-precision final polish
    if best_params is not None and check_time(8.0):
        try:
            # Multiple restarts with tiny perturbations
            for attempt in range(3):
                if not check_time(5.0):
                    break
                
                perturbed = best_params.copy()
                noise_scale = 0.005 * (attempt + 1)
                perturbed[:3*N] += np.random.normal(0, noise_scale, size=3*N)
                
                res_final = minimize(
                    objective,
                    perturbed,
                    method='L-BFGS-B',
                    args=(25.0,),
                    options={
                        'maxiter': 200,
                        'ftol': 1e-14,
                        'gtol': 1e-12,
                        'maxfun': 500
                    }
                )
                
                if is_valid_config(res_final.x[:3*N].reshape(N, 3)):
                    L_val = get_L(res_final.x[:3*N].reshape(N, 3), res_final.x[3*N])
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res_final.x.copy()
        except Exception:
            pass
    
    # Phase 6: Final direct optimization
    if best_params is not None and check_time(4.0):
        try:
            res_direct = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                args=(30.0,),
                options={
                    'maxiter': 200,
                    'ftol': 1e-15,
                    'gtol': 1e-13,
                    'maxfun': 400
                }
            )
            
            if is_valid_config(res_direct.x[:3*N].reshape(N, 3)):
                L_val = get_L(res_direct.x[:3*N].reshape(N, 3), res_direct.x[3*N])
                if L_val < best_L:
                    best_L = L_val
                    best_params = res_direct.x.copy()
        except Exception:
            pass
    
    # Final validation and fallback
    if best_params is None or not is_valid_config(best_params[:3*N].reshape(N, 3)):
        # Use best from initial configs
        found_valid = False
        for cfg in initial_configs:
            for orot in outer_rots:
                params = np.concatenate((cfg.flatten(), [orot]))
                if is_valid_config(params):
                    L_val = get_L(params[:3*N].reshape(N, 3), orot)
                    if L_val < best_L:
                        best_L = L_val
                        best_params = params.copy()
                        found_valid = True
                    break
            if found_valid:
                break
        
        if best_params is None:
            # Absolute fallback
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = []
    for x, y, rot in xys_rots:
        inner.append([float(x), float(y), float(rot)])
    
    return inner, (0.0, 0.0), float(best_L), float(outer_rot)
