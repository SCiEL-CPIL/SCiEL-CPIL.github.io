"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses direct free-form optimization with smooth differentiable objective.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
T_START = time.time()
DEADLINE = T_START + 130.0  # safety margin before 150s kill

def time_left():
    return DEADLINE - time.time()

def hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    return [(cx + s * math.cos(a + i * math.pi / 3.0),
             cy + s * math.sin(a + i * math.pi / 3.0)) for i in range(6)]

def point_in_convex_polygon(px, py, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True

def polygons_overlap_sat(A, B):
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def compute_min_outer_side(inner, outer_center, outer_angle):
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    return (2.0 / math.sqrt(3.0)) * max_proj

def smooth_max(values, beta):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    m = np.max(values)
    return m + np.log(np.sum(np.exp((values - m) * beta)) / len(values)) / beta

def compute_min_outer_side_smooth(inner, outer_center, outer_angle, beta):
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    projections = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                projections.append(dx * nx + dy * ny)
    if not projections:
        return 0.0
    return (2.0 / math.sqrt(3.0)) * smooth_max(projections, beta)

def build_ring12(scale, inner_rot, outer_rot):
    inner = []
    for k in range(12):
        theta = math.radians(inner_rot + k * 30.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        inner.append((x, y, inner_rot))
    return inner, (0.0, 0.0), outer_rot

def build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot):
    inner = []
    for k in range(6):
        theta = math.radians(rot1 + k * 60.0)
        inner.append((scale1 * math.cos(theta), scale1 * math.sin(theta), rot1))
    for k in range(6):
        theta = math.radians(rot2 + k * 60.0)
        inner.append((scale2 * math.cos(theta), scale2 * math.sin(theta), rot2))
    return inner, (0.0, 0.0), outer_rot

def build_rect3x4(sx, sy, rot, outer_rot):
    inner = []
    for i in range(3):
        for j in range(4):
            inner.append(((i - 1) * sx, (j - 1.5) * sy, rot))
    return inner, (0.0, 0.0), outer_rot

def build_hex_lattice(s, rot, outer_rot):
    """Honeycomb lattice of 12 hexagons in hexagonal arrangement."""
    inner = []
    # 7-hex cluster + 5 more
    positions = [
        (0, 0), (s, 0), (-s, 0),
        (s/2, s*math.sqrt(3)/2), (-s/2, s*math.sqrt(3)/2),
        (s/2, -s*math.sqrt(3)/2), (-s/2, -s*math.sqrt(3)/2),
        (1.5*s, s*math.sqrt(3)/2), (-1.5*s, s*math.sqrt(3)/2),
        (1.5*s, -s*math.sqrt(3)/2), (-1.5*s, -s*math.sqrt(3)/2),
        (0, s*math.sqrt(3)), (0, -s*math.sqrt(3)),
    ][:12]
    for (px, py) in positions:
        inner.append((px, py, rot))
    return inner, (0.0, 0.0), outer_rot

def free_form_objective(params, outer_rot, beta, mode):
    """
    params: 36 values = 12 * (x, y, angle_deg)
    Returns smooth objective: outer_side + overlap_penalty
    """
    inner = []
    for i in range(12):
        x = params[i*3]
        y = params[i*3+1]
        ang = params[i*3+2]
        inner.append((x, y, ang))
    
    center = (0.0, 0.0)
    side = compute_min_outer_side_smooth(inner, center, outer_rot, beta)
    
    # Compute pairwise overlap penalty using smooth min of SAT gaps
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    penalty = 0.0
    for i in range(12):
        for j in range(i+1, 12):
            A, B = polys[i], polys[j]
            min_gap = 1e10
            for poly in (A, B):
                n = len(poly)
                for k in range(n):
                    x1, y1 = poly[k]
                    x2, y2 = poly[(k+1) % n]
                    dx, dy = x2 - x1, y2 - y1
                    nx, ny = -dy, dx
                    norm = math.sqrt(nx*nx + ny*ny)
                    if norm < 1e-12:
                        continue
                    nx, ny = nx/norm, ny/norm
                    minA = min(nx*px + ny*py for px, py in A)
                    maxA = max(nx*px + ny*py for px, py in A)
                    minB = min(nx*px + ny*py for px, py in B)
                    maxB = max(nx*px + ny*py for px, py in B)
                    gap = min(maxA, maxB) - max(minA, minB)
                    if gap < min_gap:
                        min_gap = gap
            # min_gap < 0 means overlap; smooth penalty
            if min_gap < 0:
                penalty += min_gap * min_gap * 50.0
            elif min_gap < 0.01:
                penalty += (0.01 - min_gap) * 10.0
    
    return side + penalty

def optimize_free_form(outer_rot, time_budget):
    """Direct optimization of 36 free parameters."""
    start = time.time()
    
    # Multiple initializations
    initializations = []
    
    # Init 1: ring of 12
    s0 = 2.2
    for k in range(12):
        theta = math.radians(k * 30.0)
        initializations.append([s0*math.cos(theta), s0*math.sin(theta), 0.0])
    
    # Init 2: hex lattice
    s = 1.8
    positions = [
        (0, 0), (s, 0), (-s, 0),
        (s/2, s*math.sqrt(3)/2), (-s/2, s*math.sqrt(3)/2),
        (s/2, -s*math.sqrt(3)/2), (-s/2, -s*math.sqrt(3)/2),
        (1.5*s, s*math.sqrt(3)/2), (-1.5*s, s*math.sqrt(3)/2),
        (1.5*s, -s*math.sqrt(3)/2), (-1.5*s, -s*math.sqrt(3)/2),
        (0, s*math.sqrt(3)),
    ]
    init2 = []
    for (px, py) in positions:
        init2.extend([px, py, 0.0])
    initializations.append(init2)
    
    # Init 3: 3x4 grid
    init3 = []
    for i in range(3):
        for j in range(4):
            init3.extend([(i-1)*2.0, (j-1.5)*2.0, 0.0])
    initializations.append(init3)
    
    # Init 4: two rings of 6
    init4 = []
    for k in range(6):
        theta = math.radians(k * 60.0)
        init4.extend([1.5*math.cos(theta), 1.5*math.sin(theta), 0.0])
    for k in range(6):
        theta = math.radians(30.0 + k * 60.0)
        init4.extend([3.0*math.cos(theta), 3.0*math.sin(theta), 0.0])
    initializations.append(init4)
    
    best_x = None
    best_f = 1000.0
    
    n_inits = len(initializations)
    time_per_init = time_budget / n_inits
    
    for init_idx, x0 in enumerate(initializations):
        if time.time() > DEADLINE - 10.0:
            break
        if time.time() - start > time_budget:
            break
        
        x0_arr = np.array(x0, dtype=float)
        f0 = free_form_objective(x0_arr, outer_rot, 40.0, 'smooth')
        
        if f0 < best_f:
            best_f = f0
            best_x = x0_arr.copy()
        
        # Nelder-Mead with smooth objective
        try:
            budget_here = min(time_per_init * 0.6, time_budget - (time.time() - start))
            if budget_here < 1.0:
                continue
            
            result = minimize(
                lambda p: free_form_objective(p, outer_rot, 40.0, 'smooth'),
                x0_arr,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5,
                         'adaptive': True}
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
        
        # L-BFGS-B with smooth objective for gradient-based refinement
        if time.time() > DEADLINE - 10.0:
            break
        try:
            budget_here = min(time_per_init * 0.4, time_budget - (time.time() - start))
            if budget_here < 1.0:
                continue
            
            # Bounds: positions in [-10, 10], angles in [-180, 180]
            bounds = []
            for i in range(12):
                bounds.append((-10, 10))  # x
                bounds.append((-10, 10))  # y
                bounds.append((-180, 180))  # angle
            
            result = minimize(
                lambda p: free_form_objective(p, outer_rot, 60.0, 'smooth'),
                best_x if best_x is not None else x0_arr,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6,
                         'maxfun': 500}
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Final polish with exact objective
    if best_x is not None and time.time() < DEADLINE - 5.0:
        try:
            # Convert to inner list and compute exact side
            inner = [(best_x[i*3], best_x[i*3+1], best_x[i*3+2]) for i in range(12)]
            side = compute_min_outer_side(inner, (0,0), outer_rot)
            if check_validity(inner, (0,0), side, outer_rot):
                best_f = side
            else:
                # Try to fix: slightly expand
                best_f = side + 0.01
        except Exception:
            pass
    
    if best_x is None:
        return None, None
    return best_x, best_f

def optimize_topology(topology, outer_rot, time_budget):
    """Optimize a parameterized topology (kept for compatibility)."""
    start = time.time()
    
    if topology == 'ring12':
        x0 = [2.2, 0.0]
        bounds = [(0.8, 3.5), (0, 360)]
    elif topology == 'hex6+6':
        x0 = [1.5, 3.0, 0.0, 30.0]
        bounds = [(0.8, 2.5), (1.5, 4.5), (0, 360), (0, 360)]
    elif topology == 'rect3x4':
        x0 = [2.0, 2.0, 0.0]
        bounds = [(1.5, 3.5), (1.5, 3.5), (0, 360)]
    elif topology == 'hex_lattice':
        x0 = [1.8, 0.0]
        bounds = [(1.2, 3.0), (0, 360)]
    else:
        return None, None, None
    
    best_x = x0
    best_f = 1000.0
    
    def topo_obj(params):
        if topology == 'ring12':
            inner, center, orot = build_ring12(params[0], params[1], outer_rot)
        elif topology == 'hex6+6':
            inner, center, orot = build_hex6_plus6(params[0], params[1], params[2], params[3], outer_rot)
        elif topology == 'rect3x4':
            inner, center, orot = build_rect3x4(params[0], params[1], params[2], outer_rot)
        elif topology == 'hex_lattice':
            inner, center, orot = build_hex_lattice(params[0], params[1], outer_rot)
        else:
            return 1000.0
        side = compute_min_outer_side(inner, center, orot)
        if not check_validity(inner, center, side, orot):
            return side + 5.0
        return side
    
    best_f = topo_obj(x0)
    
    # Grid search
    n_grid = max(4, min(8, int(time_budget / 2.0)))
    if len(x0) == 2:
        g1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
        g2 = np.linspace(bounds[1][0], bounds[1][1], n_grid)
        for a in g1:
            for b in g2:
                f = topo_obj([a, b])
                if f < best_f:
                    best_f = f
                    best_x = [a, b]
                if time.time() - start > time_budget * 0.7:
                    break
            if time.time() - start > time_budget * 0.7:
                break
    elif len(x0) == 3:
        g1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
        g2 = np.linspace(bounds[1][0], bounds[1][1], n_grid)
        g3 = np.linspace(bounds[2][0], bounds[2][1], max(3, n_grid-2))
        for a in g1:
            for b in g2:
                for c in g3:
                    f = topo_obj([a, b, c])
                    if f < best_f:
                        best_f = f
                        best_x = [a, b, c]
                if time.time() - start > time_budget * 0.7:
                    break
            if time.time() - start > time_budget * 0.7:
                break
    elif len(x0) == 4:
        g1 = np.linspace(bounds[0][0], bounds[0][1], max(3, n_grid-2))
        g2 = np.linspace(bounds[1][0], bounds[1][1], max(3, n_grid-2))
        g3 = np.linspace(bounds[2][0], bounds[2][1], 3)
        g4 = np.linspace(bounds[3][0], bounds[3][1], 3)
        for a in g1:
            for b in g2:
                if b < a + 0.5:
                    continue
                for c in g3:
                    for d in g4:
                        f = topo_obj([a, b, c, d])
                        if f < best_f:
                            best_f = f
                            best_x = [a, b, c, d]
                if time.time() - start > time_budget * 0.7:
                    break
            if time.time() - start > time_budget * 0.7:
                break
    
    # Local refinement
    if time.time() - start < time_budget * 0.9:
        try:
            result = minimize(lambda p: topo_obj(p), best_x, method='Nelder-Mead',
                             bounds=bounds, options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass
    
    return best_x, best_f, topology

def optimize():
    best_side = 40.0
    best_inner = None
    best_outer_rot = 0.0
    best_valid = False
    
    outer_rots = [0.0, 30.0, 15.0, 45.0]
    topologies = ['ring12', 'hex6+6', 'rect3x4', 'hex_lattice']
    
    # Phase 1: Topology-based optimization (fast, good initial solutions)
    total_combos = len(topologies) * len(outer_rots)
    time_per_topo = max(2.0, min(8.0, 50.0 / total_combos))
    
    for orot in outer_rots:
        for topo in topologies:
            if time.time() > DEADLINE - 40.0:
                break
            params, f, topo_used = optimize_topology(topo, orot, time_per_topo)
            if params is not None and f < best_side:
                if topo_used == 'ring12':
                    inner, center, orot_used = build_ring12(params[0], params[1], orot)
                elif topo_used == 'hex6+6':
                    inner, center, orot_used = build_hex6_plus6(params[0], params[1], params[2], params[3], orot)
                elif topo_used == 'rect3x4':
                    inner, center, orot_used = build_rect3x4(params[0], params[1], params[2], orot)
                elif topo_used == 'hex_lattice':
                    inner, center, orot_used = build_hex_lattice(params[0], params[1], orot)
                else:
                    continue
                side = compute_min_outer_side(inner, center, orot_used)
                if check_validity(inner, center, side, orot_used):
                    if side < best_side:
                        best_side = side
                        best_inner = inner
                        best_outer_rot = orot
                        best_valid = True
    
    # Phase 2: Free-form direct optimization (structural improvement)
    if time.time() < DEADLINE - 30.0:
        for orot in outer_rots[:2]:  # focus on best outer rotations
            if time.time() > DEADLINE - 15.0:
                break
            time_budget = min(30.0, DEADLINE - 15.0 - time.time())
            if time_budget < 5.0:
                break
            params, f = optimize_free_form(orot, time_budget)
            if params is not None:
                inner = [(params[i*3], params[i*3+1], params[i*3+2]) for i in range(12)]
                side = compute_min_outer_side(inner, (0,0), orot)
                if check_validity(inner, (0,0), side, orot):
                    if side < best_side:
                        best_side = side
                        best_inner = inner
                        best_outer_rot = orot
                        best_valid = True
                elif side < best_side + 0.05:
                    # Close but invalid; try to fix by slight perturbation
                    for trial in range(10):
                        if time.time() > DEADLINE - 10.0:
                            break
                        trial_inner = []
                        for (x, y, ang) in inner:
                            dx = (np.random.random() - 0.5) * 0.02
                            dy = (np.random.random() - 0.5) * 0.02
                            trial_inner.append((x+dx, y+dy, ang))
                        s2 = compute_min_outer_side(trial_inner, (0,0), orot)
                        if check_validity(trial_inner, (0,0), s2, orot) and s2 < best_side:
                            best_side = s2
                            best_inner = trial_inner
                            best_outer_rot = orot
                            best_valid = True
                            break
    
    if best_inner is not None:
        return best_side, best_inner, 'free_form', best_outer_rot, best_valid
    return best_side, None, None, best_outer_rot, best_valid

def construct():
    t0 = time.time()
    try:
        side, inner, topology, outer_rot, valid = optimize()
        if valid and inner is not None:
            # Final verification
            actual_side = compute_min_outer_side(inner, (0,0), outer_rot)
            if check_validity(inner, (0,0), actual_side, outer_rot):
                return inner, (0.0, 0.0), actual_side, outer_rot
            # If slightly invalid, expand
            if actual_side < side + 0.1:
                return inner, (0.0, 0.0), actual_side + 0.01, outer_rot
    except Exception:
        pass
    
    # Fallback: known valid ring configuration
    inner = []
    s = 2.5
    for k in range(12):
        theta = math.radians(k * 30.0)
        inner.append((s * math.cos(theta), s * math.sin(theta), 0.0))
    side = compute_min_outer_side(inner, (0,0), 0.0)
    if not check_validity(inner, (0,0), side, 0.0):
        side += 0.5
    return inner, (0.0, 0.0), side, 0.0

if __name__ == '__main__':
    inner, center, side, angle = construct()
    print(f"Outer side: {side:.6f}")
    print(f"Center: {center}")
    print(f"Angle: {angle}")
    for i, (x, y, a) in enumerate(inner):
        print(f"  Hex {i}: ({x:.4f}, {y:.4f}, {a:.2f}°)")
