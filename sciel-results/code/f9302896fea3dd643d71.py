import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 140.0  # Safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """
    Calculate the 6 vertices of a regular hexagon.
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        # Rotate and translate
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _axes_from_polygon(verts):
    """
    Get the normal vectors for each edge of the polygon.
    """
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        # Normal vector
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB):
    """
    Check if two convex polygons overlap using Separating Axis Theorem (SAT).
    Returns True if they overlap, False if they are disjoint.
    """
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        ax, ay = axis
        # Project vertices of A onto axis
        minA = min(vx * ax + vy * ay for vx, vy in vertsA)
        maxA = max(vx * ax + vy * ay for vx, vy in vertsA)
        # Project vertices of B onto axis
        minB = min(vx * ax + vy * ay for vx, vy in vertsB)
        maxB = max(vx * ax + vy * ay for vx, vy in vertsB)
        
        # Check for separation
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _calculate_outer_side(cx_arr, cy_arr, angle_arr, outer_cx, outer_cy, outer_angle):
    """
    Calculate the minimum side length of the outer hexagon required to contain
    all inner hexagons.
    """
    a = math.radians(outer_angle)
    apothem_unit = math.sqrt(3) / 2.0
    
    # Normals to the sides of the outer hexagon
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    max_need = 0.0
    for i in range(N):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                # Distance from outer center to vertex projected onto normal
                val = (nx * (vx - outer_cx) + ny * (vy - outer_cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    """
    Check if all hexagons are disjoint.
    """
    n = len(cx_arr)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    for i in range(n):
        vi = verts_list[i]
        xi, yi = cx_arr[i], cy_arr[i]
        for j in range(i + 1, n):
            vj = verts_list[j]
            xj, yj = cx_arr[j], cy_arr[j]
            
            # Quick bounding box / distance check
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            
            # Max distance between centers of non-overlapping unit hexagons is 2*sqrt(3)
            # If dist > 2*sqrt(3), they definitely don't overlap.
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            
            # If dist < 2 (touching distance for flat-to-flat is 2*sqrt(3)? No, 
            # distance between centers of touching hexagons is 2*apothem = sqrt(3) if flat-to-flat?
            # Actually, for unit hexagons, the distance between centers when touching is 2*sqrt(3)/2 * 2? 
            # Apothem is sqrt(3)/2. Distance between parallel sides is sqrt(3).
            # If two hexagons touch side-to-side, center distance is sqrt(3).
            # If they touch vertex-to-vertex, center distance is 2.
            # So if dist < sqrt(3), they MUST overlap? No, if rotated, min distance is smaller?
            # Min distance between centers of disjoint hexagons is > sqrt(3) if aligned, but can be less if rotated?
            # Actually, the inradius is sqrt(3)/2. The sum of inradii is sqrt(3).
            # So if center distance < sqrt(3), they overlap.
            if dist_sq < (math.sqrt(3) - 1e-6) ** 2:
                return False
            
            # Perform SAT check
            if _separating_axis_check(vi, vj):
                return False
    
    return True

def construct():
    t0 = time.time()
    random.seed(42)
    
    # --- Initializers ---
    
    def make_lattice_initial():
        """3x4 hexagonal lattice."""
        pts = []
        row_height = math.sqrt(3)
        col_width = 2.0
        # 3 rows of 4
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts

    def make_ring_initial():
        """12 hexagons in a ring."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts

    def make_flower_initial():
        """Central hex + ring of 6 + outer 5."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        r1 = math.sqrt(3)  # Touching distance for flat-to-flat? No, 2*apothem = sqrt(3)
        # Wait, apothem of unit hex is sqrt(3)/2. Distance between centers of touching hexagons is 2*sqrt(3)/2 = sqrt(3).
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y, 0.0])
        
        r2 = r1 * 1.732  # Approx
        # Add 5 more in a larger ring
        for i in range(5):
            angle = math.radians(60 * i + 30)
            x = r2 * math.cos(angle)
            y = r2 * math.sin(angle)
            pts.append([x, y, 0.0])
        return pts[:12]

    def make_random_initial():
        pts = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            pts.append([x, y, a])
        return pts

    initializers = [
        make_lattice_initial,
        make_ring_initial,
        make_flower_initial,
        make_random_initial,
    ]
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Evaluate initial configurations
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
            side = _calculate_outer_side(cx_arr, cy_arr, angle_arr, 0.0, 0.0, 0.0)
            if side < best_side:
                best_side = side
                best_cfg = cfg[:]
                best_center = (0.0, 0.0)
                best_angle = 0.0

    if best_cfg is None:
        # Fallback: scaled lattice
        cfg = make_lattice_initial()
        for p in cfg:
            p[0] *= 1.5
            p[1] *= 1.5
        best_cfg = cfg
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        best_side = _calculate_outer_side(cx_arr, cy_arr, angle_arr, 0.0, 0.0, 0.0)

    # --- Nelder-Mead Optimization ---
    
    def nelder_mead(cfg_init, center_init, angle_init, time_budget):
        """
        Run Nelder-Mead optimization on the flattened parameter vector.
        """
        def flatten(cfg, center, angle):
            v = []
            for p in cfg:
                v.extend([p[0], p[1], p[2]])
            v.extend([center[0], center[1], angle])
            return np.array(v)
        
        def unflatten(v):
            cfg = []
            for i in range(N):
                cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
            center = (v[3*N], v[3*N+1])
            angle = v[3*N+2]
            return cfg, center, angle
        
        def objective(v):
            cfg, center, angle = unflatten(v)
            cx_arr = np.array([p[0] for p in cfg])
            cy_arr = np.array([p[1] for p in cfg])
            angle_arr = np.array([p[2] for p in cfg])
            
            # Check validity
            if not _is_valid_fast(cx_arr, cy_arr, angle_arr):
                # Calculate penalty based on overlap depth
                # For simplicity, return a large penalty
                # We can refine this by estimating overlap
                return 100.0
            
            side = _calculate_outer_side(cx_arr, cy_arr, angle_arr, center[0], center[1], angle)
            return side

        v0 = flatten(cfg_init, center_init, angle_init)
        n_params = len(v0)
        
        # Initialize simplex
        simplex = [v0.copy()]
        step = 0.1
        for i in range(n_params):
            vi = v0.copy()
            vi[i] += step
            simplex.append(vi)
        
        # NM Parameters
        alpha_nm = 1.0
        gamma_nm = 2.0
        rho_nm = 0.5
        sigma_nm = 0.5
        
        best_f = min(objective(v) for v in simplex)
        best_v = min(simplex, key=objective)
        
        max_iters = 2000
        iter_count = 0
        
        while iter_count < max_iters and time.time() - t0 < time_budget:
            # Sort simplex
            simplex.sort(key=lambda v: objective(v))
            f_best = objective(simplex[0])
            
            if f_best < best_f:
                best_f = f_best
                best_v = simplex[0].copy()
            
            f_worst = objective(simplex[-1])
            
            # Convergence check
            if abs(f_worst - f_best) < 1e-6:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha_nm * (centroid - simplex[-1])
            fr = objective(xr)
            
            if fr < f_best:
                # Expansion
                xe = centroid + gamma_nm * (centroid - simplex[-1])
                fe = objective(xe)
                if fe < fr:
                    simplex[-1] = xe
                else:
                    simplex[-1] = xr
            elif fr < objective(simplex[-2]):
                simplex[-1] = xr
            else:
                # Contraction
                if fr < f_worst:
                    # Outside contraction
                    xc = centroid + rho_nm * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    xc = centroid + rho_nm * (centroid - simplex[-1])
                fc = objective(xc)
                
                if fc < min(fr, f_worst):
                    simplex[-1] = xc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
            
            iter_count += 1
        
        return best_v, best_f

    # Run NM from the best initial configuration
    if time.time() - t0 < TIME_LIMIT - 10:
        cfg_start = [row[:] for row in best_cfg]
        center_start = best_center
        angle_start = best_angle
        
        # Run NM
        best_v, best_f = nelder_mead(cfg_start, center_start, angle_start, TIME_LIMIT - 10)
        
        if best_f < best_side:
            cfg_opt, center_opt, angle_opt = unflatten(best_v)
            cx_arr = np.array([p[0] for p in cfg_opt])
            cy_arr = np.array([p[1] for p in cfg_opt])
            angle_arr = np.array([p[2] for p in cfg_opt])
            
            if _is_valid_fast(cx_arr, cy_arr, angle_arr):
                side_opt = _calculate_outer_side(cx_arr, cy_arr, angle_arr, center_opt[0], center_opt[1], angle_opt)
                if side_opt < best_side:
                    best_side = side_opt
                    best_cfg = cfg_opt
                    best_center = center_opt
                    best_angle = angle_opt

    # Final verification and return
    cx_final = np.array([p[0] for p in best_cfg])
    cy_final = np.array([p[1] for p in best_cfg])
    angle_final = np.array([p[2] for p in best_cfg])
    
    if not _is_valid_fast(cx_final, cy_final, angle_final):
        # Fallback to initial valid config if NM broke it
        cfg = make_lattice_initial()
        for p in cfg:
            p[0] *= 1.3
            p[1] *= 1.3
        best_cfg = cfg
        cx_final = np.array([p[0] for p in cfg])
        cy_final = np.array([p[1] for p in cfg])
        angle_final = np.array([p[2] for p in cfg])
        best_side = _calculate_outer_side(cx_final, cy_final, angle_final, 0.0, 0.0, 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0
    else:
        best_side = _calculate_outer_side(cx_final, cy_final, angle_final, best_center[0], best_center[1], best_angle)
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, best_center, best_side, best_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
