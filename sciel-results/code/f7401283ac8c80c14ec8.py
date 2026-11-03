"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s

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
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            vj = verts_list[j]
            aj = axes_list[j]
            
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx * dx + dy * dy
            
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            
            if dist_sq < 4.0 - 1e-9:
                return False
            
            for axis in ai:
                ax, ay = axis
                minA = min(vx * ax + vy * ay for vx, vy in vi)
                maxA = max(vx * ax + vy * ay for vx, vy in vi)
                minB = min(vx * ax + vy * ay for vx, vy in vj)
                maxB = max(vx * ax + vy * ay for vx, vy in vj)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    break
            else:
                for axis in aj:
                    ax, ay = axis
                    minA = min(vx * ax + vy * ay for vx, vy in vi)
                    maxA = max(vx * ax + vy * ay for vx, vy in vi)
                    minB = min(vx * ax + vy * ay for vx, vy in vj)
                    maxB = max(vx * ax + vy * ay for vx, vy in vj)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        break
                else:
                    return False
    
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    """Compute the minimum side length of an outer hexagon (at given center/rotation)
    that encloses all inner hexagons."""
    apothem_unit = math.sqrt(3) / 2.0
    a = math.radians(angle_deg)
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

def _optimize_outer(cx_arr, cy_arr, angle_arr, center, angle_deg, n_center_steps=5, n_angle_steps=6):
    """Grid-search over outer hexagon center and rotation to minimize side length."""
    best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg)
    best_cx, best_cy = center
    best_angle = angle_deg
    
    # Compute centroid of all inner hexagon vertices for initial center guess
    all_verts = []
    for i in range(len(cx_arr)):
        all_verts.extend(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    centroid_x = sum(v[0] for v in all_verts) / len(all_verts)
    centroid_y = sum(v[1] for v in all_verts) / len(all_verts)
    
    # Grid search over center offsets around centroid
    cx0, cy0 = center
    dx_range = 1.5
    dy_range = 1.5
    
    for di in range(n_center_steps):
        for dj in range(n_center_steps):
            tcx = centroid_x + (dx_range * (di - n_center_steps // 2) / max(n_center_steps - 1, 1))
            tcy = centroid_y + (dy_range * (dj - n_center_steps // 2) / max(n_center_steps - 1, 1))
            for angle_idx in range(n_angle_steps):
                tangle = 360.0 * angle_idx / n_angle_steps
                s = _outer_hex_side(cx_arr, cy_arr, angle_arr, (tcx, tcy), tangle)
                if s < best_side:
                    best_side = s
                    best_cx, best_cy = tcx, tcy
                    best_angle = tangle
    
    # Also try original center with different angles
    for angle_idx in range(n_angle_steps):
        tangle = 360.0 * angle_idx / n_angle_steps
        s = _outer_hex_side(cx_arr, cy_arr, angle_arr, (cx0, cy0), tangle)
        if s < best_side:
            best_side = s
            best_cx, best_cy = cx0, cy0
            best_angle = tangle
    
    return best_side, (best_cx, best_cy), best_angle

def construct():
    t0 = time.time()
    random.seed(42)
    
    def make_flower_initial():
        pts = []
        pts.append([0.0, 0.0, 0.0])
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        r2 = r1 * 1.732
        angles = [30, 90, 150, 210, 270]
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y, 0.0])
        return pts[:12]
    
    def make_lattice_initial():
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        for c in range(3):
            x = c * col_width - col_width
            y = -row_height
            pts.append([x, y, 0.0])
        for c in range(4):
            x = c * col_width - col_width - col_width / 2
            y = 0.0
            pts.append([x, y, 0.0])
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        pts.append([-col_width * 2 - col_width / 2 - col_width, 0.0, 0.0])
        pts.append([col_width * 2 - col_width / 2 + col_width, 0.0, 0.0])
        return pts
    
    def make_random_initial():
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts
    
    def make_ring_initial():
        pts = [[0.0, 0.0, 0.0]]
        r = 2 * math.sqrt(3)
        for i in range(11):
            angle = math.radians(360 * i / 11 + 15)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            pts.append([x, y, 0.0])
        return pts
    
    def make_compact_hex():
        """Tight hexagonal cluster: center + 6 at distance 2*sqrt(3) + 5 more."""
        pts = [[0.0, 0.0, 0.0]]
        r = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            pts.append([r * math.cos(angle), r * math.sin(angle), 0.0])
        # 5 more in second ring, spaced
        r2 = r * 1.8
        for i in range(5):
            angle = math.radians(60 * i + 30)
            pts.append([r2 * math.cos(angle), r2 * math.sin(angle), 0.0])
        return pts
    
    initializers = [
        make_flower_initial,
        make_lattice_initial,
        make_ring_initial,
        make_compact_hex,
        make_random_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 60:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side, center, oangle = _optimize_outer(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
                best_center = center
                best_angle = oangle
    
    if best_cfg is None:
        cfg = make_lattice_initial()
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        best_side, best_center, best_angle = _optimize_outer(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)

    # --- Phase 1: Simulated Annealing on inner positions ---
    cfg = [row[:] for row in best_cfg]
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    cur_score = best_side
    T = 0.3
    T_end = 0.001
    alpha = 0.995
    
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    def _quick_score(cx_arr, cy_arr, angle_arr):
        """Fast score using current best outer hexagon params."""
        ocx, ocy = best_center
        oa = math.radians(best_angle)
        onormals = []
        for k in range(6):
            na = oa + math.radians(90 + 60 * k)
            onormals.append((math.cos(na), math.sin(na)))
        
        max_need = 0.0
        for i in range(N):
            a = math.radians(angle_arr[i])
            ca, sa = math.cos(a), math.sin(a)
            for k in range(6):
                theta = math.radians(60 * k)
                vx = cx_arr[i] + ca * math.cos(theta) - sa * math.sin(theta)
                vy = cy_arr[i] + sa * math.cos(theta) + ca * math.sin(theta)
                for (nx, ny) in onormals:
                    val = (nx * (vx - ocx) + ny * (vy - ocy)) / apothem_unit
                    if val > max_need:
                        max_need = val
        return max_need

    while time.time() - t0 < TIME_LIMIT - 35:
        r = random.random()
        idx = random.randrange(N)
        old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
        
        # Perturbation
        if r < 0.5:
            cx_arr[idx] += random.gauss(0, T * 0.5)
            cy_arr[idx] += random.gauss(0, T * 0.5)
        elif r < 0.8:
            angle_arr[idx] += random.gauss(0, T * 15.0)
            angle_arr[idx] %= 60.0
        else:
            # Move two hexagons
            idx2 = random.randrange(N)
            if idx2 == idx:
                idx2 = (idx + 1) % N
            old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
            cx_arr[idx] += random.gauss(0, T * 0.3)
            cy_arr[idx] += random.gauss(0, T * 0.3)
            cx_arr[idx2] += random.gauss(0, T * 0.3)
            cy_arr[idx2] += random.gauss(0, T * 0.3)
        
        # Check validity
        if not _is_valid_fast(cx_arr, cy_arr, angle_arr):
            # Revert
            cx_arr[idx] = old_x
            cy_arr[idx] = old_y
            angle_arr[idx] = old_a
            if r >= 0.8:
                cx_arr[idx2] = old_x2
                cy_arr[idx2] = old_y2
                angle_arr[idx2] = old_a2
            T *= alpha
            continue
        
        # Compute new score
        new_score = _quick_score(cx_arr, cy_arr, angle_arr)
        
        if new_score < cur_score or random.random() < math.exp((cur_score - new_score) / max(T, 1e-9)):
            cur_score = new_score
            # Periodically re-optimize outer hexagon
            if int((cur_score * 1000) % 50) == 0 and time.time() - t0 < TIME_LIMIT - 40:
                side, center, oangle = _optimize_outer(cx_arr, cy_arr, angle_arr, best_center, best_angle)
                if side < best_side:
                    best_side = side
                    best_center = center
                    best_angle = oangle
                    cur_score = side
                    cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
        else:
            # Revert
            cx_arr[idx] = old_x
            cy_arr[idx] = old_y
            angle_arr[idx] = old_a
            if r >= 0.8:
                cx_arr[idx2] = old_x2
                cy_arr[idx2] = old_y2
                angle_arr[idx2] = old_a2
        
        T *= alpha
        if T < T_end:
            T = T_end

    # Final re-optimization of outer hexagon
    if time.time() - t0 < TIME_LIMIT - 10:
        final_side, final_center, final_angle = _optimize_outer(cx_arr, cy_arr, angle_arr, best_center, best_angle)
        if final_side < best_side:
            best_side = final_side
            best_center = final_center
            best_angle = final_angle
            cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]

    return cfg, best_center, best_side, best_angle
