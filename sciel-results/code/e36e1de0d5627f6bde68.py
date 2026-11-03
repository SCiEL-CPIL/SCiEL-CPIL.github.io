import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap_fast(v1, v2):
    """Fast SAT overlap check for two hexagons. Returns separation (positive) or -penetration."""
    # Pre-compute edges and normals for both polygons
    axes = []
    for v in (v1, v2):
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap

def outer_hex_side_centered(points, outer_rot_deg, center_x=0.0, center_y=0.0):
    """Minimum side length of regular hexagon centered at (center_x, center_y) with given rotation."""
    rel_points = points - np.array([center_x, center_y])
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = rel_points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        center_x = params[3*N + 1]
        center_y = params[3*N + 2]
        
        # Quick center distance pre-filter
        centers = xys_rots[:, :2]
        min_center_dist = np.sqrt(3) - 0.1  # Conservative threshold
        
        penalty = 0.0
        vertices_list = []
        for i in range(N):
            cx, cy, rot = xys_rots[i]
            v = get_hex_vertices(cx, cy, 1.0, rot)
            vertices_list.append(v)
        
        # SAT overlap check with early exit for far-apart hexagons
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < min_center_dist:
                    sep = sat_overlap_fast(vertices_list[i], vertices_list[j])
                    if sep < 0:
                        penalty += (-sep) ** 2
        
        all_points = np.vstack(vertices_list)
        L = outer_hex_side_centered(all_points, outer_rot, center_x, center_y)
        
        return L + 50.0 * penalty
    
    def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
        """Create a hex grid of unit hexagons with optional gap."""
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5
        for r in range(rows):
            y = r * spacing_y
            for c in range(cols):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, rot_offset])
        pts = np.array(pts)
        if len(pts) > N:
            pts = pts[:N]
        elif len(pts) < N:
            # Pad with additional positions
            while len(pts) < N:
                last = pts[-1]
                pts.append([last[0] + sqrt3, last[1], rot_offset])
            pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_ring_config(gap=0.0):
        """1 center + 6 ring + 5 outer with gap."""
        pts = []
        sqrt3 = np.sqrt(3)
        d1 = sqrt3 + gap
        d2 = 2 * sqrt3 + gap
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_compact_cluster(gap=0.0):
        """Compact arrangement: 2 rows of 3 + 2 rows of 3 offset."""
        pts = []
        sqrt3 = np.sqrt(3)
        sx = sqrt3 + gap
        sy = 1.5 + gap * 0.5
        # Row 0: 3 hexes
        for c in range(3):
            pts.append([c * sx, 0, 0.0])
        # Row 1: 3 hexes offset
        for c in range(3):
            pts.append([c * sx + sx/2, sy, 0.0])
        # Row 2: 3 hexes
        for c in range(3):
            pts.append([c * sx, 2*sy, 0.0])
        # Row 3: 3 hexes offset
        for c in range(3):
            pts.append([c * sx + sx/2, 3*sy, 0.0])
        pts = np.array(pts)
        pts = pts[:N]
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_diagonal_strip(gap=0.0):
        """Diagonal strip arrangement."""
        pts = []
        sqrt3 = np.sqrt(3)
        for i in range(N):
            x = i * sqrt3 * 0.7 + gap * i * 0.5
            y = i * 1.2 + gap * i * 0.3
            pts.append([x, y, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations
    gap = 0.08
    configs = []
    configs.append(('grid_4x3', make_hex_grid(4, 3, gap=gap)))
    configs.append(('grid_3x4', make_hex_grid(3, 4, gap=gap)))
    configs.append(('grid_2x6', make_hex_grid(2, 6, gap=gap)))
    configs.append(('grid_6x2', make_hex_grid(6, 2, gap=gap)))
    configs.append(('ring', make_ring_config(gap=gap)))
    configs.append(('compact', make_compact_cluster(gap=gap)))
    configs.append(('diagonal', make_diagonal_strip(gap=gap)))
    
    # Add rotated variants
    for name, cfg in list(configs[:4]):
        rotated = cfg.copy()
        rotated[:, 2] = 30.0
        configs.append((name + '_rot30', rotated))
    
    outer_rots = [0.0, 30.0, 15.0, 45.0, -15.0]
    
    for cfg_name, cfg in configs:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 15:
                break
            
            # Initialize center at mean of points
            cx0 = np.mean(cfg[:, 0])
            cy0 = np.mean(cfg[:, 1])
            
            init_params = np.concatenate((cfg.flatten(), [orot, cx0, cy0]))
            
            # Check initial validity
            xys_rots_init = init_params[:3*N].reshape(N, 3)
            outer_rot_init = init_params[3*N]
            center_x_init = init_params[3*N + 1]
            center_y_init = init_params[3*N + 2]
            
            vertices_list_init = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots_init]
            
            has_overlap = False
            centers = xys_rots_init[:, :2]
            for i in range(N):
                for j in range(i + 1, N):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < np.sqrt(3) - 0.05:
                        sep = sat_overlap_fast(vertices_list_init[i], vertices_list_init[j])
                        if sep < -1e-6:
                            has_overlap = True
                            break
                if has_overlap:
                    break
            
            if not has_overlap:
                all_points = np.vstack(vertices_list_init)
                L_val = outer_hex_side_centered(all_points, outer_rot_init, center_x_init, center_y_init)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            try:
                res = minimize(
                    objective,
                    init_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 150,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'adaptive': True
                    }
                )
                
                xys_rots = res.x[:3*N].reshape(N, 3)
                outer_rot = res.x[3*N]
                center_x = res.x[3*N + 1]
                center_y = res.x[3*N + 2]
                
                vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
                
                has_overlap = False
                centers = xys_rots[:, :2]
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = centers[i, 0] - centers[j, 0]
                        dy = centers[i, 1] - centers[j, 1]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < np.sqrt(3) - 0.05:
                            sep = sat_overlap_fast(vertices_list[i], vertices_list[j])
                            if sep < -1e-6:
                                has_overlap = True
                                break
                    if has_overlap:
                        break
                
                if not has_overlap:
                    all_points = np.vstack(vertices_list)
                    L_val = outer_hex_side_centered(all_points, outer_rot, center_x, center_y)
                    
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res.x.copy()
                        
            except Exception:
                pass
    
    # Final refinement with tighter tolerances
    if best_params is not None and time.time() - start_time < DEADLINE - 25:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-7,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            
            xys_rots = res.x[:3*N].reshape(N, 3)
            outer_rot = res.x[3*N]
            center_x = res.x[3*N + 1]
            center_y = res.x[3*N + 2]
            
            vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
            
            has_overlap = False
            centers = xys_rots[:, :2]
            for i in range(N):
                for j in range(i + 1, N):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < np.sqrt(3) - 0.05:
                        sep = sat_overlap_fast(vertices_list[i], vertices_list[j])
                        if sep < -1e-6:
                            has_overlap = True
                            break
                    if has_overlap:
                        break
            
            if not has_overlap:
                all_points = np.vstack(vertices_list)
                L_val = outer_hex_side_centered(all_points, outer_rot, center_x, center_y)
                
                if L_val < best_L:
                    best_L = L_val
                    best_params = res.x.copy()
                    
        except Exception:
            pass
    
    # Fallback
    if best_params is None:
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side_centered(pts, 0.0, 0.0, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0
    
    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    center_x = best_params[3*N + 1]
    center_y = best_params[3*N + 2]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [float(center_x), float(center_y)]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle
