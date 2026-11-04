import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s
INRADIUS = math.sqrt(3) / 2.0
MIN_DIST_CIRCLE = 2 * INRADIUS  # sqrt(3) ~ 1.732
APOTHEM = math.sqrt(3) / 2.0

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
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
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
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
    """
    Fast validity check using distance-based pre-filter + SAT fallback.
    cx_arr, cy_arr, angle_arr: numpy arrays of shape (N,)
    Returns True if all hexagons are pairwise disjoint.
    """
    n = len(cx_arr)
    
    # Fast pre-check: Circle packing (vectorized)
    # If distance between centers < 2*inradius, they MUST overlap.
    min_dist_sq = (MIN_DIST_CIRCLE * MIN_DIST_CIRCLE) - 1e-9
    
    # Vectorized distance check
    for i in range(n):
        dx = cx_arr[i] - cx_arr[i+1:]
        dy = cy_arr[i] - cy_arr[i+1:]
        dist_sq = dx*dx + dy*dy
        if np.any(dist_sq < min_dist_sq):
            return False

    # Exact check: SAT (optimized)
    # Precompute vertices for all hexagons
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    # Precompute axes for all hexagons
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    # SAT check for each pair
    for i in range(n):
        xi, yi = cx_arr[i], cy_arr[i]
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            xj, yj = cx_arr[j], cy_arr[j]
            vj = verts_list[j]
            aj = axes_list[j]
            
            # Quick distance check to skip SAT if far apart
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            
            # SAT check
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
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
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
    """Approximate score assuming outer hex is centered at (0,0) and aligned 0."""
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

def _sdf_hexagon_point(px, py, cx, cy, angle_deg):
    """
    Signed distance from point (px,py) to a unit regular hexagon centered at (cx,cy)
    with rotation angle_deg. Negative if inside, positive if outside.
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx = px - cx
    dy = py - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    
    max_dist = 0.0
    for k in range(6):
        na = math.radians(60 * k + 30)
        nx, ny = math.cos(na), math.sin(na)
        dist = nx * lx + ny * ly - APOTHEM
        if dist > max_dist:
            max_dist = dist
    
    if max_dist < 0:
        min_dist_to_edge = float('inf')
        for k in range(6):
            na = math.radians(60 * k + 30)
            nx, ny = math.cos(na), math.sin(na)
            dist = APOTHEM - (nx * lx + ny * ly)
            if dist > 0 and dist < min_dist_to_edge:
                min_dist_to_edge = dist
        return -min_dist_to_edge
    else:
        return max_dist

def _sdf_hexagon_hexagon(cx1, cy1, a1, cx2, cy2, a2):
    """
    Approximate signed distance between two unit hexagons.
    Uses the minimum SDF from vertices of one hex to the other hexagon.
    Negative if overlapping, positive if disjoint.
    """
    d_center = math.hypot(cx1 - cx2, cy1 - cy2)
    if d_center > 2 * math.sqrt(3):
        return d_center - 2 * math.sqrt(3)
    
    verts1 = _hex_vertices(cx1, cy1, a1)
    verts2 = _hex_vertices(cx2, cy2, a2)
    
    min_sdf = float('inf')
    for (vx, vy) in verts1:
        d = _sdf_hexagon_point(vx, vy, cx2, cy2, a2)
        if d < min_sdf:
            min_sdf = d
    for (vx, vy) in verts2:
        d = _sdf_hexagon_point(vx, vy, cx1, cy1, a1)
        if d < min_sdf:
            min_sdf = d
    
    return min_sdf

def _sdf_outer_hexagon(px, py, cx, cy, side, angle_deg):
    """
    Signed distance from point (px,py) to the interior of an outer regular hexagon.
    Negative if inside, positive if outside.
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx = px - cx
    dy = py - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    
    apothem_outer = side * APOTHEM
    
    max_dist = 0.0
    for k in range(6):
        na = math.radians(60 * k + 30)
        nx, ny = math.cos(na), math.sin(na)
        dist = nx * lx + ny * ly - apothem_outer
        if dist > max_dist:
            max_dist = dist
    
    return max_dist

def _pbd_step(cx_arr, cy_arr, angle_arr, outer_side, outer_center, outer_angle, dt=0.05, iterations=10):
    """
    Position-Based Dynamics step: iteratively project onto constraint manifold.
    Constraints: (1) inner hexagons don't overlap, (2) inner hexagons inside outer hexagon.
    """
    cx_c, cy_c = outer_center
    
    for _iter in range(iterations):
        # Constraint 1: Pairwise non-overlap of inner hexagons
        for i in range(N):
            for j in range(i + 1, N):
                d = _sdf_hexagon_hexagon(cx_arr[i], cy_arr[i], angle_arr[i], 
                                         cx_arr[j], cy_arr[j], angle_arr[j])
                if d < 0:
                    # Overlap: push apart along center-to-center direction
                    dx = cx_arr[i] - cx_arr[j]
                    dy = cy_arr[i] - cy_arr[j]
                    dist = math.hypot(dx, dy)
                    if dist < 1e-9:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    correction = -d / 2.0  # Split correction equally
                    cx_arr[i] += dx * correction
                    cy_arr[i] += dy * correction
                    cx_arr[j] -= dx * correction
                    cy_arr[j] -= dy * correction
        
        # Constraint 2: Inner hexagons must be inside outer hexagon
        for i in range(N):
            verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
            max_violation = 0.0
            push_x, push_y = 0.0, 0.0
            for (vx, vy) in verts:
                sdf = _sdf_outer_hexagon(vx, vy, cx_c, cy_c, outer_side, outer_angle)
                if sdf > max_violation:
                    max_violation = sdf
                    # Compute push direction (from outer hex center toward vertex)
                    px = vx - cx_c
                    py = vy - cy_c
                    pd = math.hypot(px, py)
                    if pd > 1e-9:
                        push_x = px / pd
                        push_y = py / pd
            
            if max_violation > 0:
                # Push the hexagon center away from the outer hex boundary
                cx_arr[i] -= push_x * max_violation
                cy_arr[i] -= push_y * max_violation
    
    return cx_arr, cy_arr, angle_arr

def _pbd_optimize(cx_init, cy_init, angle_init, outer_center, outer_angle, 
                  time_budget, max_side=8.0):
    """
    PBD-based optimization: start with a large outer hexagon and shrink it,
    using PBD to resolve collisions at each step.
    """
    t0 = time.time()
    cx_arr = np.array(cx_init, dtype=float)
    cy_arr = np.array(cy_init, dtype=float)
    angle_arr = np.array(angle_init, dtype=float)
    
    cx_c, cy_c = outer_center
    
    # Start with a large outer hexagon and shrink
    current_side = max_side
    min_side = float('inf')
    best_cfg = None
    shrink_step = 0.15
    
    while time.time() - t0 < time_budget:
        # Try to shrink the outer hexagon
        new_side = current_side - shrink_step
        
        # If we can't shrink further, try a smaller step
        if new_side < 3.0:
            shrink_step = 0.05
            new_side = current_side - shrink_step
            if new_side < 2.5:
                break
        
        # Run PBD with the new (smaller) outer hexagon
        cx_test = cx_arr.copy()
        cy_test = cy_arr.copy()
        ang_test = angle_arr.copy()
        
        cx_test, cy_test, ang_test = _pbd_step(
            cx_test, cy_test, ang_test, 
            new_side, outer_center, outer_angle,
            dt=0.05, iterations=15
        )
        
        # Check if configuration is valid (no overlaps)
        if _is_valid_fast(cx_test, cy_test, ang_test):
            # Compute actual required side length
            actual_side = _outer_hex_side(cx_test, cy_test, ang_test, outer_center, outer_angle)
            
            if actual_side <= new_side + 0.01:
                # Success: accept this configuration
                current_side = new_side
                cx_arr, cy_arr, angle_arr = cx_test, cy_test, ang_test
                min_side = min(min_side, actual_side)
                best_cfg = [cx_arr[i], cy_arr[i], angle_arr[i]]
            else:
                # Partial success: keep positions but use actual side
                current_side = max(new_side, actual_side - 0.1)
                cx_arr, cy_arr, angle_arr = cx_test, cy_test, ang_test
                min_side = min(min_side, actual_side)
                best_cfg = [cx_arr[i], cy_arr[i], angle_arr[i]]
        else:
            # Failed: revert and try smaller shrink
            shrink_step *= 0.7
            if shrink_step < 0.01:
                break
    
    return min_side, best_cfg

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
        make_flower_initial,
        make_lattice_initial,
        make_ring_initial,
        make_random_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Try each initializer
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
            side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
                best_center = (0.0, 0.0)
                best_angle = 0.0
    
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
        best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, (0.0, 0.0), 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0

    # --- Phase 1: Simulated Annealing ---
    cfg = [row[:] for row in best_cfg]
    cx_arr = np.array([p[0] for p in cfg])
    cy_arr = np.array([p[1] for p in cfg])
    angle_arr = np.array([p[2] for p in cfg])
    
    cur_score = best_side
    T = 0.2
    T_end = 0.003
    alpha = 0.996
    
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    def _quick_score_local(cx_arr, cy_arr, angle_arr):
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

    while time.time() - t0 < TIME_LIMIT - 30:
        r = random.random()
        idx = random.randrange(N)
        old_x, old_y, old_a = cx_arr[idx], cy_arr[idx], angle_arr[idx]
        idx2 = -1
        old_x2, old_y2, old_a2 = 0, 0, 0
        
        if r < 0.7:
            cx_arr[idx] += random.gauss(0, 0.12)
            cy_arr[idx] += random.gauss(0, 0.12)
            angle_arr[idx] += random.gauss(0, 2.5)
        elif r < 0.85:
            angle_arr[idx] += random.gauss(0, 6.0)
        else:
            idx2 = random.randrange(N)
            if idx2 == idx:
                idx2 = (idx + 1) % N
            old_x2, old_y2, old_a2 = cx_arr[idx2], cy_arr[idx2], angle_arr[idx2]
            cx_arr[idx] += random.gauss(0, 0.08)
            cy_arr[idx] += random.gauss(0, 0.08)
            cx_arr[idx2] += random.gauss(0, 0.08)
            cy_arr[idx2] += random.gauss(0, 0.08)
            angle_arr[idx] += random.gauss(0, 1.5)
            angle_arr[idx2] += random.gauss(0, 1.5)
        
        valid = _is_valid_fast(cx_arr, cy_arr, angle_arr)
        
        if valid:
            new_score = _quick_score_local(cx_arr, cy_arr, angle_arr)
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_side:
                    best_side = new_score
                    best_cfg = [[cx_arr[i], cy_arr[i], angle_arr[i]] for i in range(N)]
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
            else:
                cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
                if idx2 != -1:
                    cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        else:
            cx_arr[idx], cy_arr[idx], angle_arr[idx] = old_x, old_y, old_a
            if idx2 != -1:
                cx_arr[idx2], cy_arr[idx2], angle_arr[idx2] = old_x2, old_y2, old_a2
        
        T = max(T_end, T * alpha)
    
    # --- Phase 2: Nelder-Mead optimization ---
    cx_best = np.array([p[0] for p in best_cfg])
    cy_best = np.array([p[1] for p in best_cfg])
    angle_best = np.array([p[2] for p in best_cfg])
    
    best_final_side = best_side
    best_final_cfg = best_cfg[:]
    best_final_center = best_center
    best_final_angle = best_angle
    
    # Build initial parameter vector
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = cx_best[i]
        x0[3*i+1] = cy_best[i]
        x0[3*i+2] = angle_best[i]
    
    # Try multiple outer hexagon angles
    for outer_angle_deg in [0, 5, 10, 15, 20, 25]:
        if time.time() - t0 > TIME_LIMIT - 20:
            break
        
        outer_center = (0.0, 0.0)
        
        # Run Nelder-Mead optimization
        try:
            from scipy.optimize import minimize
            
            result = minimize(
                _nm_objective,
                x0,
                args=(outer_center, float(outer_angle_deg)),
                method='Nelder-Mead',
                options={'maxiter': 200, 'fatol': 1e-8, 'xtol': 1e-6}
            )
            
            if result.fun < best_final_side:
                # Extract optimized configuration
                opt_params = result.x
                opt_cfg = []
                for i in range(N):
                    opt_cfg.append([opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]])
                
                # Verify validity
                opt_cx = np.array([p[0] for p in opt_cfg])
                opt_cy = np.array([p[1] for p in opt_cfg])
                opt_ang = np.array([p[2] for p in opt_cfg])
                
                if _is_valid_fast(opt_cx, opt_cy, opt_ang):
                    # Compute actual side length
                    side = _outer_hex_side(opt_cx, opt_cy, opt_ang, outer_center, float(outer_angle_deg))
                    if side < best_final_side:
                        best_final_side = side
                        best_final_cfg = opt_cfg[:]
                        best_final_center = outer_center
                        best_final_angle = float(outer_angle_deg)
                        
                        # Update x0 for next iteration
                        x0 = opt_params[:]
        except Exception:
            pass
    
    # Phase 3: Refine outer hexagon center for the best configuration found
    if best_final_cfg is not None:
        cx_ref = np.array([p[0] for p in best_final_cfg])
        cy_ref = np.array([p[1] for p in best_final_cfg])
        ang_ref = np.array([p[2] for p in best_final_cfg])
        
        verts_ref = [_hex_vertices(cx_ref[i], cy_ref[i], ang_ref[i]) for i in range(N)]
        
        # Optimize center for the current best angle
        outer_a = math.radians(best_final_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals_ref = []
        for k in range(6):
            na = outer_a + math.radians(90 + 60 * k)
            normals_ref.append((math.cos(na), math.sin(na)))
        
        cx_c, cy_c = best_final_center
        
        for _iter in range(30):
            if time.time() - t0 > TIME_LIMIT - 5:
                break
            
            best_cx, best_cy, best_val = cx_c, cy_c, float('inf')
            
            for dx in np.linspace(-0.3, 0.3, 7):
                for dy in np.linspace(-0.3, 0.3, 7):
                    ncx, ncy = cx_c + dx, cy_c + dy
                    val = 0.0
                    for i in range(N):
                        for (vx, vy) in verts_ref[i]:
                            for (nx, ny) in normals_ref:
                                v = (nx * (vx - ncx) + ny * (vy - ncy)) / apothem_unit
                                if v > val:
                                    val = v
                    if val < best_val:
                        best_val = val
                        best_cx, best_cy = ncx, ncy
            
            if abs(best_cx - cx_c) < 1e-4 and abs(best_cy - cy_c) < 1e-4:
                break
            cx_c, cy_c = best_cx, best_cy
        
        if best_val < best_final_side:
            best_final_side = best_val
            best_final_center = (cx_c, cy_c)
    
    # Final verification
    cx_final = np.array([p[0] for p in best_final_cfg])
    cy_final = np.array([p[1] for p in best_final_cfg])
    angle_final = np.array([p[2] for p in best_final_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        cfg = make_lattice_initial()
        scale = 1.3
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_final_cfg = cfg
        cx_final = np.array([p[0] for p in cfg])
        cy_final = np.array([p[1] for p in cfg])
        angle_final = np.array([p[2] for p in cfg])
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, (0.0, 0.0), 0.0)
        best_final_center = (0.0, 0.0)
        best_final_angle = 0.0
    else:
        # Recompute side with final center/angle to be sure
        best_final_side = _outer_hex_side(cx_final, cy_final, angle_final, best_final_center, best_final_angle)
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_final_cfg]
    return inner, best_final_center, best_final_side, best_final_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
