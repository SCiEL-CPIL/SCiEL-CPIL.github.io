import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s
INRADIUS = math.sqrt(3) / 2.0
MIN_DIST_CIRCLE = 2 * INRADIUS  # 2 * sqrt(3)/2 = sqrt(3) ~ 1.732

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    
    # Fast pre-check: Circle packing (vectorized)
    # If distance between centers < 2*inradius, they MUST overlap.
    min_dist_sq = (MIN_DIST_CIRCLE * MIN_DIST_CIRCLE) - 1e-9
    
    for i in range(n):
        dx = cx_arr[i] - cx_arr[i+1:]
        dy = cy_arr[i] - cy_arr[i+1:]
        dist_sq = dx*dx + dy*dy
        if np.any(dist_sq < min_dist_sq):
            return False

    # Exact check: SAT
    verts_list = []
    axes_list = []
    for i in range(n):
        v = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        verts_list.append(v)
        axes_list.append(_axes_from_polygon(v))
    
    for i in range(n):
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            vj = verts_list[j]
            aj = axes_list[j]
            
            overlap = True
            for axis in ai:
                minA, maxA = _project_polygon(vi, axis)
                minB, maxB = _project_polygon(vj, axis)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    overlap = False
                    break
            
            if overlap:
                for axis in aj:
                    minA, maxA = _project_polygon(vi, axis)
                    minB, maxB = _project_polygon(vj, axis)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        overlap = False
                        break
            
            if overlap:
                return False
                
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _generate_rigid_configs():
    """
    Generates several rigid honeycomb-like configurations of 12 unit hexagons.
    Returns a list of numpy arrays (N, 3) with [x, y, angle].
    """
    configs = []
    sqrt3 = math.sqrt(3)
    
    # Topology 1: 1 center, 6 ring, 5 attached to ring
    # Lattice vectors: v1=(sqrt3, 0), v2=(sqrt3/2, 3/2)
    # Points in lattice coords (i, j) -> x = i*sqrt3 + j*sqrt3/2, y = j*1.5
    # Center (0,0)
    # Neighbors: (1,0), (0,1), (-1,0), (0,-1), (-1,1), (1,-1)
    # 5 more: (2,0), (1,1), (-2,0), (-1,-1), (0,2)
    pts_lattice = [
        (0,0),
        (1,0), (0,1), (-1,0), (0,-1), (-1,1), (1,-1),
        (2,0), (1,1), (-2,0), (-1,-1), (0,2)
    ]
    
    cfg1 = []
    for i, j in pts_lattice:
        x = i * sqrt3 + j * sqrt3 / 2.0
        y = j * 1.5
        cfg1.append([x, y, 0.0])
    configs.append(np.array(cfg1))
    
    # Topology 2: Different attachment
    # 1 center, 6 ring.
    # Attach 5 to: (2,0), (2,1), (1,2), (-1,2), (-2,1) ? 
    # Let's just shift the last 5 points to different neighbors to create a different shape.
    pts_lattice2 = [
        (0,0),
        (1,0), (0,1), (-1,0), (0,-1), (-1,1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    cfg2 = []
    for i, j in pts_lattice2:
        x = i * sqrt3 + j * sqrt3 / 2.0
        y = j * 1.5
        cfg2.append([x, y, 0.0])
    configs.append(np.array(cfg2))
    
    # Topology 3: Rectangular-ish 3x4 block (distorted)
    # 3 rows of 4?
    # Row 0: 4 hexes
    # Row 1: 4 hexes
    # Row 2: 4 hexes
    # This is not a standard honeycomb, but we can place them.
    # Actually, a 2x6 rectangle of hexagons is also possible.
    # Let's create a 2x6 grid in lattice coords.
    # i in 0..5, j in 0..1
    pts_lattice3 = []
    for j in range(2):
        for i in range(6):
            pts_lattice3.append((i, j))
            
    cfg3 = []
    for i, j in pts_lattice3:
        x = i * sqrt3 + j * sqrt3 / 2.0
        y = j * 1.5
        cfg3.append([x, y, 0.0])
    configs.append(np.array(cfg3))
    
    return configs

def _optimize_rigid(config, time_limit=2.0):
    """
    Optimizes global X, Y, Angle for a rigid config.
    Returns best (side, cx, cy, angle, optimized_config).
    """
    t0 = time.time()
    best_side = float('inf')
    best_params = (0.0, 0.0, 0.0)
    
    # Initial guess: Center of mass at (0,0), angle 0
    cx_cm = np.mean(config[:, 0])
    cy_cm = np.mean(config[:, 1])
    
    # Try a few initial angles
    initial_angles = [0, 15, 30, 45, 60]
    
    for init_angle in initial_angles:
        if time.time() - t0 > time_limit:
            break
            
        x0 = np.array([cx_cm, cy_cm, init_angle])
        
        def objective(params):
            x, y, angle = params
            # Shift config
            a = math.radians(angle)
            ca, sa = math.cos(a), math.sin(a)
            
            cx_arr = np.zeros(N)
            cy_arr = np.zeros(N)
            angle_arr = np.zeros(N)
            
            for i in range(N):
                px, py, _ = config[i]
                # Rotate around origin then translate
                rx = ca * px - sa * py
                ry = sa * px + ca * py
                cx_arr[i] = rx + x
                cy_arr[i] = ry + y
                angle_arr[i] = angle
                
            # Check validity? Rigid config is valid internally by construction (honeycomb).
            # But we need to check if it fits in the bounding box? No, we minimize the box.
            # We don't need validity check for the rigid part if the topology is known valid.
            # The honeycomb lattice points are disjoint.
            
            side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
            return side
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception:
            pass

    # Finalize best params
    x, y, angle = best_params
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    cx_arr = np.zeros(N)
    cy_arr = np.zeros(N)
    angle_arr = np.zeros(N)
    for i in range(N):
        px, py, _ = config[i]
        rx = ca * px - sa * py
        ry = sa * px + ca * py
        cx_arr[i] = rx + x
        cy_arr[i] = ry + y
        angle_arr[i] = angle
        
    # Verify validity just in case (should be valid)
    if not _is_valid_fast(cx_arr, cy_arr, angle_arr):
        return None
        
    return best_side, cx_arr, cy_arr, angle_arr

def construct():
    t0 = time.time()
    random.seed(42)
    
    best_side = float('inf')
    best_cfg = None
    
    # Phase 1: Rigid Topology Optimization
    rigid_configs = _generate_rigid_configs()
    
    for cfg in rigid_configs:
        if time.time() - t0 > TIME_LIMIT - 60:
            break
        result = _optimize_rigid(cfg)
        if result:
            side, cx_arr, cy_arr, angle_arr = result
            if side < best_side:
                best_side = side
                best_cfg = np.column_stack((cx_arr, cy_arr, angle_arr))
    
    if best_cfg is None:
        # Fallback to a simple lattice if rigid failed
        cfg = rigid_configs[0]
        best_cfg = cfg
        cx_arr = cfg[:, 0]
        cy_arr = cfg[:, 1]
        angle_arr = cfg[:, 2]
        best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)

    # Phase 2: Local Relaxation (SA)
    # Allow individual hexagons to move slightly to improve packing.
    cx_arr = best_cfg[:, 0].copy()
    cy_arr = best_cfg[:, 1].copy()
    angle_arr = best_cfg[:, 2].copy()
    
    cur_score = best_side
    T = 0.1
    T_end = 0.001
    alpha = 0.99
    
    apothem_unit = math.sqrt(3) / 2.0
    normals = [(math.cos(math.radians(90 + 60 * k)), math.sin(math.radians(90 + 60 * k))) for k in range(6)]
    
    def _quick_score(cx_arr, cy_arr, angle_arr):
        max_need = 0.0
        for i in range(N):
            a = math.radians(angle_arr[i])
            ca, sa = math.cos(a), math.sin(a)
            for k in range(6):
                theta = math.radians(60 * k)
                vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
                vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
                for (nx, ny) in normals:
                    val = (nx * vx + ny * vy) / apothem_unit
                    if val > max_need:
                        max_need = val
        return max_need

    steps = 5000
    for step in range(steps):
        if time.time() - t0 > TIME_LIMIT - 10:
            break
            
        idx = random.randrange(N)
        old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
        
        # Small perturbation
        cx_arr[idx] += random.gauss(0, 0.05)
        cy_arr[idx] += random.gauss(0, 0.05)
        angle_arr[idx] += random.gauss(0, 2.0)
        
        valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
        
        if valid:
            new_score = _quick_score(cx_arr, cy_arr, angle_arr)
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = np.column_stack((cx_arr, cy_arr, angle_arr))
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
        else:
            cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
            
        T = max(T_end, T * alpha)

    # Final verification and optimization of outer hex center/angle
    cx_final = best_cfg[:, 0]
    cy_final = best_cfg[:, 1]
    angle_final = best_cfg[:, 2]
    
    # Optimize outer hex center and angle
    best_final_side = best_side
    best_final_center = (0.0, 0.0)
    best_final_angle = 0.0
    
    verts_best = [_hex_vertices(cx_final[i], cy_final[i], angle_final[i]) for i in range(N)]
    
    for outer_angle in range(0, 30, 2): # Coarse grid
        if time.time() - t0 > TIME_LIMIT - 2:
            break
            
        a = math.radians(outer_angle)
        normals_out = [(math.cos(a + math.radians(90 + 60 * k)), math.sin(a + math.radians(90 + 60 * k))) for k in range(6)]
        
        # Simple local search for center
        cx_c, cy_c = 0.0, 0.0
        for _iter in range(10):
            max_proj = 0.0
            for i in range(N):
                for (vx, vy) in verts_best[i]:
                    for (nx, ny) in normals_out:
                        val = (nx * (vx - cx_c) + ny * (vy - cy_c)) / apothem_unit
                        if val > max_proj:
                            max_proj = val
            
            best_val = max_proj
            best_cx, best_cy = cx_c, cy_c
            for dx in [-0.1, 0, 0.1]:
                for dy in [-0.1, 0, 0.1]:
                    ncx, ncy = cx_c + dx, cy_c + dy
                    val = 0.0
                    for i in range(N):
                        for (vx, vy) in verts_best[i]:
                            for (nx, ny) in normals_out:
                                v = (nx * (vx - ncx) + ny * (vy - ncy)) / apothem_unit
                                if v > val:
                                    val = v
                    if val < best_val:
                        best_val = val
                        best_cx, best_cy = ncx, ncy
            if abs(best_cx - cx_c) < 1e-5 and abs(best_cy - cy_c) < 1e-5:
                break
            cx_c, cy_c = best_cx, best_cy
            
        if max_proj < best_final_side:
            best_final_side = max_proj
            best_final_center = (cx_c, cy_c)
            best_final_angle = float(outer_angle)

    # Final check
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        # Fallback to initial rigid config if it got corrupted
        cfg = rigid_configs[0]
        cx_final = cfg[:, 0]
        cy_final = cfg[:, 1]
        angle_final = cfg[:, 2]
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
        best_final_center = (0.0, 0.0)
        best_final_angle = 0.0
    else:
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, best_final_center, best_final_angle)

    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in zip(cx_final, cy_final, angle_final)]
    return inner, best_final_center, best_final_side, best_final_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
