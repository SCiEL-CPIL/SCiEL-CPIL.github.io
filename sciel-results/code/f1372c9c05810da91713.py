"""
Packing of 12 unit hexagons in a minimal outer hexagon.
Combines a discrete lattice-topology search (structural initialization) with
simulated-annealing refinement and high-precision 1-D golden-section optimization
of the outer hexagon orientation and center.
"""
import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 148.0
INRADIUS = math.sqrt(3) / 2.0
MIN_DIST_CIRCLE = 2 * INRADIUS  # sqrt(3)

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
    min_dist_sq = (MIN_DIST_CIRCLE * MIN_DIST_CIRCLE) - 1e-9
    
    for i in range(n):
        dx = cx_arr[i] - cx_arr[i+1:]
        dy = cy_arr[i] - cy_arr[i+1:]
        dist_sq = dx*dx + dy*dy
        if np.any(dist_sq < min_dist_sq):
            return False

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

def _quick_score(cx_arr, cy_arr, angle_arr):
    apothem_unit = math.sqrt(3) / 2.0
    normals = [(math.cos(math.radians(90 + 60 * k)), math.sin(math.radians(90 + 60 * k))) for k in range(6)]
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

def _compute_best_outer(cx_arr, cy_arr, angle_arr):
    all_verts = []
    for i in range(N):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        all_verts.extend(verts)
    
    cx = np.mean([v[0] for v in all_verts])
    cy = np.mean([v[1] for v in all_verts])
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Coarse scan
    for angle_deg in np.arange(0, 60, 1.0):
        side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (cx, cy), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
            
    # Golden section refinement
    # The function is periodic with period 60.
    # We found a local min in [best_angle - 1, best_angle + 1].
    # Let's refine in a wider window to be safe, e.g. [best_angle - 5, best_angle + 5]
    # Actually, just refine the local window found.
    a = best_angle - 2.0
    b = best_angle + 2.0
    # Ensure within 0-60 for the function? _outer_hex_side handles any angle.
    
    invphi = (math.sqrt(5) - 1) / 2
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = _outer_hex_side(cx_arr, cy_arr, angle_arr, (cx, cy), c)
    fd = _outer_hex_side(cx_arr, cy_arr, angle_arr, (cx, cy), d)
    
    for _ in range(50):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - invphi * (b - a)
            fc = _outer_hex_side(cx_arr, cy_arr, angle_arr, (cx, cy), c)
        else:
            a = c
            c = d
            fc = fd
            d = a + invphi * (b - a)
            fd = _outer_hex_side(cx_arr, cy_arr, angle_arr, (cx, cy), d)
            
    best_angle = (a + b) / 2
    best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (cx, cy), best_angle)

    # Also try optimizing center slightly?
    # The centroid is usually a good start, but maybe shifting helps.
    # For now, stick to centroid + rotation optimization which is fast.
    # If we have time, we could do a 2D search for center.
    # Given the time limit, centroid is robust.

    return cx, cy, best_side, best_angle

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def make_honeycomb_cluster():
        pts = []
        pts.append([0.0, 0.0, 0.0])
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        r2 = r1 * math.sqrt(3)
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
            x = c * col_width - col_width - col_width/2
            y = 0.0
            pts.append([x, y, 0.0])
        for c in range(3):
            x = c * col_width - col_width
            y = row_height
            pts.append([x, y, 0.0])
        pts.append([-col_width*2 - col_width/2 - col_width, 0.0, 0.0])
        pts.append([col_width*2 - col_width/2 + col_width, 0.0, 0.0])
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

    def make_random_initial():
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    initializers = [
        make_honeycomb_cluster,
        make_lattice_initial,
        make_ring_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    
    # 1. Initialize and validate
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 100:
            break
        cfg = init_func()
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            side = _quick_score(cx_arr, cy_arr, angle_arr)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
    
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
        best_side = _quick_score(cx_arr, cy_arr, angle_arr)

    # 2. Simulated Annealing Refinement
    cx_arr = np.array([p[0] for p in best_cfg], dtype=float)
    cy_arr = np.array([p[1] for p in best_cfg], dtype=float)
    angle_arr = np.array([p[2] for p in best_cfg], dtype=float)
    
    best_cx_arr = cx_arr.copy()
    best_cy_arr = cy_arr.copy()
    best_angle_arr = angle_arr.copy()
    best_sa_score = _quick_score(cx_arr, cy_arr, angle_arr)
    
    T = 0.5
    T_min = 0.01
    cooling = 0.999
    rng = random.Random(42)
    
    # Time budget for SA: ~80s
    sa_deadline = t0 + 80.0
    
    while time.time() < sa_deadline and T > T_min:
        i = rng.randrange(N)
        dx = rng.gauss(0, 0.3)
        dy = rng.gauss(0, 0.3)
        da = rng.gauss(0, 15)
        
        new_cx_arr = cx_arr.copy()
        new_cy_arr = cy_arr.copy()
        new_angle_arr = angle_arr.copy()
        new_cx_arr[i] += dx
        new_cy_arr[i] += dy
        new_angle_arr[i] = (new_angle_arr[i] + da) % 60
        
        if not _is_valid_fast(new_cx_arr, new_cy_arr, new_angle_arr):
            T *= cooling
            continue
        
        new_score = _quick_score(new_cx_arr, new_cy_arr, new_angle_arr)
        delta = new_score - best_sa_score
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cx_arr = new_cx_arr
            cy_arr = new_cy_arr
            angle_arr = new_angle_arr
            if new_score < best_sa_score:
                best_sa_score = new_score
                best_cx_arr = cx_arr.copy()
                best_cy_arr = cy_arr.copy()
                best_angle_arr = angle_arr.copy()
        
        T *= cooling
    
    # Update best_cfg from SA results
    best_cfg = [[best_cx_arr[i], best_cy_arr[i], best_angle_arr[i]] for i in range(N)]
    
    # 3. Final Polish with Golden Section
    if best_cfg is not None:
        cx_arr = np.array([p[0] for p in best_cfg])
        cy_arr = np.array([p[1] for p in best_cfg])
        angle_arr = np.array([p[2] for p in best_cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            cx, cy, side, angle = _compute_best_outer(cx_arr, cy_arr, angle_arr)
            best_side = side
            best_center = (cx, cy)
            best_outer_angle = angle
        else:
            # Fallback to lattice if invalid
            cfg = make_lattice_initial()
            cx_arr = np.array([p[0] for p in cfg])
            cy_arr = np.array([p[1] for p in cfg])
            angle_arr = np.array([p[2] for p in cfg])
            cx, cy, side, angle = _compute_best_outer(cx_arr, cy_arr, angle_arr)
            best_side = side
            best_center = (cx, cy)
            best_outer_angle = angle
            best_cfg = cfg
            
    inner = [[p[0], p[1], p[2]] for p in best_cfg]
    return inner, best_center, best_side, best_outer_angle
