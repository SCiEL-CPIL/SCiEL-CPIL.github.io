import numpy as np
import time
import math

N = 12
DEADLINE = 115.0  # Safety margin below 120s

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hex_edge_normals(rot_deg):
    # Normals to the 6 edges of a unit hexagon with given rotation
    # Edges are perpendicular to directions 30, 90, 150, 210, 270, 330 + rot
    base = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))

def _sat_overlap_fast(v1, v2, n1, n2):
    """
    Fast SAT overlap check using precomputed normals.
    v1, v2: (6, 2) vertex arrays
    n1, n2: (6, 2) normal arrays for each hexagon
    Returns: negative if overlapping (penetration depth), positive if separated.
    """
    min_overlap = float('inf')
    # Check axes from hex1
    for k in range(6):
        n = n1[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    # Check axes from hex2
    for k in range(6):
        n = n2[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    return -min_overlap

def _outer_hex_side_fast(points, outer_rot_deg):
    """
    Compute the minimum side length of a regular hexagon (centered at origin, given rotation)
    that contains all points.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def _check_disjoint(inner):
    n = len(inner)
    verts = []
    norms = []
    for i in range(n):
        x, y, rot = inner[i]
        v = _hex_vertices(x, y, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts.append(v)
        norms.append(nm)
    for i in range(n):
        for j in range(i + 1, n):
            d = _sat_overlap_fast(verts[i], verts[j], norms[i], norms[j])
            if d < -1e-9:
                return False
    return True

def _make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = 2.0 + gap
    spacing_y = sqrt3 + gap * 0.5
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
    d1 = np.sqrt(3) + gap
    d2 = 2 * np.sqrt(3) + gap
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

def _make_packed_cluster():
    """
    Create a dense cluster using a more optimized initial layout.
    """
    pts = []
    sqrt3 = np.sqrt(3)
    # Center
    pts.append([0, 0, 0.0])
    # First ring (6 hexagons)
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang), 0.0])
    # Second ring: 5 more hexagons to make 12 total
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _evaluate_config(cfg, ox, oy, oang):
    """
    Evaluates a configuration.
    cfg: (12, 3) array of [x, y, rot]
    Returns (is_valid, outer_side)
    """
    verts_list = []
    norms_list = []
    for i in range(12):
        cx, cy, rot = cfg[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts_list.append(v)
        norms_list.append(nm)
    
    # Check disjoint
    for i in range(12):
        for j in range(i + 1, 12):
            d = _sat_overlap_fast(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
            if d < -1e-9:
                return False, float('inf')
    
    # Calculate outer side
    all_pts = np.vstack(verts_list)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    
    L = _outer_hex_side_fast(all_pts, oang)
    return True, L

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    # Generate diverse initial configurations
    configs = [
        _make_hex_grid(4, 3, gap=0.1),
        _make_hex_grid(3, 4, gap=0.1),
        _make_hex_grid(2, 6, gap=0.1),
        _make_hex_grid(6, 2, gap=0.1),
        _make_ring(gap=0.1),
        _make_hex_grid(4, 3, rot_offset=30, gap=0.1),
        _make_hex_grid(3, 4, rot_offset=30, gap=0.1),
        _make_packed_cluster(),
    ]
    
    # Select the best initial configuration
    best_init_L = float('inf')
    best_init_cfg = None
    best_init_ox = 0.0
    best_init_oy = 0.0
    best_init_oang = 0.0
    
    for cfg in configs:
        # Try a couple of outer rotations for the initial evaluation
        for oang in [0.0, 30.0]:
            valid, L = _evaluate_config(cfg, 0.0, 0.0, oang)
            if valid and L < best_init_L:
                best_init_L = L
                best_init_cfg = cfg.copy()
                best_init_ox = 0.0
                best_init_oy = 0.0
                best_init_oang = oang
    
    if best_init_cfg is None:
        # Fallback: spread out random points
        np.random.seed(42)
        best_init_cfg = np.random.randn(12, 3) * 5.0
        best_init_cfg[:, 2] = 0.0
        best_init_cfg[:, 0] -= np.mean(best_init_cfg[:, 0])
        best_init_cfg[:, 1] -= np.mean(best_init_cfg[:, 1])
        valid, L = _evaluate_config(best_init_cfg, 0.0, 0.0, 0.0)
        if not valid:
             # Force validity by expanding
             scale = 1.5
             while not valid and scale < 10.0:
                 best_init_cfg[:, :2] *= scale
                 scale += 0.5
                 valid, L = _evaluate_config(best_init_cfg, 0.0, 0.0, 0.0)
        best_init_L = L
        best_init_ox = 0.0
        best_init_oy = 0.0
        best_init_oang = 0.0

    # Simulated Annealing
    cfg = best_init_cfg
    ox = best_init_ox
    oy = best_init_oy
    oang = best_init_oang
    
    cur_L = best_init_L
    best_L = cur_L
    best_cfg = cfg.copy()
    best_ox = ox
    best_oy = oy
    best_oang = oang
    
    T = 1.0
    T_end = 0.001
    alpha = 0.9995
    
    # Move parameters
    pos_step = 0.5
    rot_step = 10.0
    center_step = 0.2
    oang_step = 5.0
    
    np.random.seed(42)
    
    while time_left() > 5.0:
        r = np.random.random()
        
        if r < 0.50:
            # Move one inner hexagon position
            idx = np.random.randint(12)
            old_x = cfg[idx, 0]
            old_y = cfg[idx, 1]
            cfg[idx, 0] += np.random.normal(0, pos_step)
            cfg[idx, 1] += np.random.normal(0, pos_step)
            
            valid, new_L = _evaluate_config(cfg, ox, oy, oang)
            
            if valid:
                if new_L < cur_L or (T > T_end and np.random.random() < math.exp(-(new_L - cur_L) / T)):
                    cur_L = new_L
                    if new_L < best_L:
                        best_L = new_L
                        best_cfg = cfg.copy()
                        best_ox = ox
                        best_oy = oy
                        best_oang = oang
                else:
                    cfg[idx, 0] = old_x
                    cfg[idx, 1] = old_y
            else:
                cfg[idx, 0] = old_x
                cfg[idx, 1] = old_y
                
        elif r < 0.70:
            # Rotate one inner hexagon
            idx = np.random.randint(12)
            old_rot = cfg[idx, 2]
            cfg[idx, 2] += np.random.normal(0, rot_step)
            
            valid, new_L = _evaluate_config(cfg, ox, oy, oang)
            
            if valid:
                if new_L < cur_L or (T > T_end and np.random.random() < math.exp(-(new_L - cur_L) / T)):
                    cur_L = new_L
                    if new_L < best_L:
                        best_L = new_L
                        best_cfg = cfg.copy()
                        best_ox = ox
                        best_oy = oy
                        best_oang = oang
                else:
                    cfg[idx, 2] = old_rot
            else:
                cfg[idx, 2] = old_rot
                
        elif r < 0.85:
            # Move outer center
            old_ox = ox
            old_oy = oy
            ox += np.random.normal(0, center_step)
            oy += np.random.normal(0, center_step)
            
            valid, new_L = _evaluate_config(cfg, ox, oy, oang)
            
            if valid:
                if new_L < cur_L or (T > T_end and np.random.random() < math.exp(-(new_L - cur_L) / T)):
                    cur_L = new_L
                    if new_L < best_L:
                        best_L = new_L
                        best_cfg = cfg.copy()
                        best_ox = ox
                        best_oy = oy
                        best_oang = oang
                else:
                    ox = old_ox
                    oy = old_oy
            else:
                ox = old_ox
                oy = old_oy
                
        else:
            # Rotate outer hexagon
            old_oang = oang
            oang += np.random.normal(0, oang_step)
            
            valid, new_L = _evaluate_config(cfg, ox, oy, oang)
            
            if valid:
                if new_L < cur_L or (T > T_end and np.random.random() < math.exp(-(new_L - cur_L) / T)):
                    cur_L = new_L
                    if new_L < best_L:
                        best_L = new_L
                        best_cfg = cfg.copy()
                        best_ox = ox
                        best_oy = oy
                        best_oang = oang
                else:
                    oang = old_oang
            else:
                oang = old_oang
                
        T = max(T_end, T * alpha)
        
        # Adaptive step size: shrink if we've been improving, expand if stuck?
        # For simplicity, keep fixed steps but maybe reduce slightly over time
        # pos_step *= 0.9999
        # rot_step *= 0.9999

    # Final validity check and extraction
    if not _check_disjoint(best_cfg):
        # Fallback to a known valid configuration if something went wrong
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    inner = []
    for i in range(12):
        x = best_cfg[i, 0]
        y = best_cfg[i, 1]
        r = best_cfg[i, 2]
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (float(best_ox), float(best_oy))
    outer_angle = float(best_oang)
    
    # Recalculate exact outer side for the best config to be sure
    all_pts = np.vstack([_hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(12)])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side_fast(all_pts, outer_angle)
    
    # Add a tiny safety margin to ensure validity if floating point errors occur
    outer_side *= 1.0001
    
    return inner, outer_center, outer_side, outer_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
