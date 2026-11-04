"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses a direct continuous optimization over all DOFs (positions + angles)
with a smooth epigraph formulation for the containment and separation constraints.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 145.0

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
    normals = [(math.cos(a + (i + 0.5) * math.pi / 3.0),
                math.sin(a + (i + 0.5) * math.pi / 3.0)) for i in range(6)]
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

def smooth_max(values, k=50.0):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return 0.0
    v_max = float(np.max(v))
    if v_max < -1e10:
        return v_max
    shifted = np.clip(v - v_max, -50.0, 50.0)
    return v_max + math.log(float(np.sum(np.exp(k * shifted)) / k)) / k

def sat_gap(A, B):
    max_gap = -float('inf')
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            norm = math.sqrt(nx * nx + ny * ny)
            if norm < 1e-9:
                continue
            nx, ny = nx / norm, ny / norm
            projA = np.array([nx * px + ny * py for px, py in A])
            projB = np.array([nx * px + ny * py for px, py in B])
            minA, maxA = float(np.min(projA)), float(np.max(projA))
            minB, maxB = float(np.min(projB)), float(np.max(projB))
            if maxA < minB:
                gap = minB - maxA
            elif maxB < minA:
                gap = minA - maxB
            else:
                gap = -(min(maxA, maxB) - max(minA, minB))
            if gap > max_gap:
                max_gap = gap
    return max_gap

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
        x = scale1 * math.cos(theta)
        y = scale1 * math.sin(theta)
        inner.append((x, y, rot1))
    for k in range(6):
        theta = math.radians(rot2 + k * 60.0)
        x = scale2 * math.cos(theta)
        y = scale2 * math.sin(theta)
        inner.append((x, y, rot2))
    return inner, (0.0, 0.0), outer_rot

def build_rect3x4(scale_x, scale_y, rot, outer_rot):
    inner = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * scale_x
            y = (j - 1.5) * scale_y
            inner.append((x, y, rot))
    return inner, (0.0, 0.0), outer_rot

def build_hex_lattice(r1, r3, outer_rot):
    inner = [(0.0, 0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        x = r1 * math.cos(theta)
        y = r1 * math.sin(theta)
        inner.append((x, y, 0.0))
    for k in range(5):
        theta = math.radians(k * 72.0 + 18.0)
        x = r3 * math.cos(theta)
        y = r3 * math.sin(theta)
        inner.append((x, y, 0.0))
    return inner, (0.0, 0.0), outer_rot

def build_linear_chain(spacing, outer_rot):
    inner = []
    for k in range(12):
        x = (k - 5.5) * spacing
        inner.append((x, 0.0, 0.0))
    return inner, (0.0, 0.0), outer_rot

def build_hexagonal_cluster(r1, r2, outer_rot):
    inner = [(0.0, 0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        x = r1 * math.cos(theta)
        y = r1 * math.sin(theta)
        inner.append((x, y, 0.0))
    for k in range(5):
        theta = math.radians(k * 72.0 + 18.0)
        x = r2 * math.cos(theta)
        y = r2 * math.sin(theta)
        inner.append((x, y, 0.0))
    return inner, (0.0, 0.0), outer_rot

def objective(params, topology, outer_rot):
    if topology == 'ring12':
        scale, inner_rot = params
        if scale < 0.5 or scale > 3.0:
            return 1000.0
        inner, center, orot = build_ring12(scale, inner_rot, outer_rot)
    elif topology == 'hex6+6':
        scale1, scale2, rot1, rot2 = params
        if scale1 < 0.5 or scale2 < 0.5 or scale2 < scale1 + 0.3:
            return 1000.0
        inner, center, orot = build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot)
    elif topology == 'rect3x4':
        sx, sy, rot = params
        if sx < 1.5 or sy < 1.5:
            return 1000.0
        inner, center, orot = build_rect3x4(sx, sy, rot, outer_rot)
    elif topology == 'hex_lattice':
        r1, r3 = params
        if r1 < 1.0 or r3 < r1 + 0.5:
            return 1000.0
        inner, center, orot = build_hex_lattice(r1, r3, outer_rot)
    elif topology == 'linear':
        spacing = params[0]
        if spacing < 1.5:
            return 1000.0
        inner, center, orot = build_linear_chain(spacing, outer_rot)
    elif topology == 'cluster':
        r1, r2 = params
        if r1 < 1.0 or r2 < r1 + 0.5:
            return 1000.0
        inner, center, orot = build_hexagonal_cluster(r1, r2, outer_rot)
    else:
        return 1000.0

    side = compute_min_outer_side(inner, center, orot)
    if not check_validity(inner, center, side, orot):
        outer = hex_vertices(center[0], center[1], side, orot)
        polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
        max_violation = 0.0
        for p in polys:
            for (px, py) in p:
                for i in range(6):
                    x1, y1 = outer[i]
                    x2, y2 = outer[(i + 1) % 6]
                    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                    if cross < -1e-7:
                        max_violation = max(max_violation, -cross)
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    max_violation = max(max_violation, -gap)
        return side + 20.0 * max_violation + 10.0
    return side

def smooth_objective(params, topology, outer_rot, k=50.0):
    if topology == 'ring12':
        scale, inner_rot = params
        if scale < 0.5 or scale > 3.0:
            return 1000.0
        inner, center, orot = build_ring12(scale, inner_rot, outer_rot)
    elif topology == 'hex6+6':
        scale1, scale2, rot1, rot2 = params
        if scale1 < 0.5 or scale2 < 0.5 or scale2 < scale1 + 0.3:
            return 1000.0
        inner, center, orot = build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot)
    elif topology == 'rect3x4':
        sx, sy, rot = params
        if sx < 1.5 or sy < 1.5:
            return 1000.0
        inner, center, orot = build_rect3x4(sx, sy, rot, outer_rot)
    elif topology == 'hex_lattice':
        r1, r3 = params
        if r1 < 1.0 or r3 < r1 + 0.5:
            return 1000.0
        inner, center, orot = build_hex_lattice(r1, r3, outer_rot)
    elif topology == 'linear':
        spacing = params[0]
        if spacing < 1.5:
            return 1000.0
        inner, center, orot = build_linear_chain(spacing, outer_rot)
    elif topology == 'cluster':
        r1, r2 = params
        if r1 < 1.0 or r2 < r1 + 0.5:
            return 1000.0
        inner, center, orot = build_hexagonal_cluster(r1, r2, outer_rot)
    else:
        return 1000.0

    cx, cy = center
    a = math.radians(orot)
    normals = [(math.cos(a + (i + 0.5) * math.pi / 3.0),
                math.sin(a + (i + 0.5) * math.pi / 3.0)) for i in range(6)]

    all_projs = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                all_projs.append(dx * nx + dy * ny)

    side_smooth = (2.0 / math.sqrt(3.0)) * smooth_max(all_projs, k)

    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    overlap_penalty = 0.0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            gap = sat_gap(polys[i], polys[j])
            if gap < 0:
                overlap_penalty += (-gap) ** 2

    return side_smooth + 10.0 * overlap_penalty

def optimize_topology(topology, outer_rot, time_budget):
    start_time = time.time()

    if topology == 'ring12':
        x0 = [2.0, 0.0]
        bounds = [(0.8, 2.5), (0, 360)]
    elif topology == 'hex6+6':
        x0 = [1.5, 3.0, 0.0, 30.0]
        bounds = [(0.8, 2.5), (1.5, 4.0), (0, 360), (0, 360)]
    elif topology == 'rect3x4':
        x0 = [2.0, 2.0, 0.0]
        bounds = [(1.5, 3.0), (1.5, 3.0), (0, 360)]
    elif topology == 'hex_lattice':
        x0 = [1.7, 3.5]
        bounds = [(1.0, 2.5), (2.0, 4.5)]
    elif topology == 'linear':
        x0 = [1.8]
        bounds = [(1.5, 3.0)]
    elif topology == 'cluster':
        x0 = [1.7, 3.0]
        bounds = [(1.0, 2.5), (1.5, 4.0)]
    else:
        return None, None, None

    best_x = x0
    best_f = objective(x0, topology, outer_rot)

    # Coarse grid search
    n_grid = 5
    if topology == 'ring12':
        grid1 = np.linspace(0.8, 2.5, n_grid)
        grid2 = np.linspace(0, 360, n_grid)
        for g1 in grid1:
            for g2 in grid2:
                f = objective([g1, g2], topology, outer_rot)
                if f < best_f:
                    best_f = f
                    best_x = [g1, g2]
                if time.time() - start_time > time_budget:
                    return best_x, best_f, topology
    elif topology == 'hex6+6':
        grid1 = np.linspace(0.8, 2.5, n_grid)
        grid2 = np.linspace(1.5, 4.0, n_grid)
        grid3 = np.linspace(0, 360, n_grid)
        grid4 = np.linspace(0, 360, n_grid)
        for g1 in grid1:
            for g2 in grid2:
                if g2 < g1 + 0.3:
                    continue
                for g3 in grid3:
                    for g4 in grid4:
                        f = objective([g1, g2, g3, g4], topology, outer_rot)
                        if f < best_f:
                            best_f = f
                            best_x = [g1, g2, g3, g4]
                        if time.time() - start_time > time_budget:
                            return best_x, best_f, topology
    elif topology == 'rect3x4':
        grid1 = np.linspace(1.5, 3.0, n_grid)
        grid2 = np.linspace(1.5, 3.0, n_grid)
        grid3 = np.linspace(0, 360, n_grid)
        for g1 in grid1:
            for g2 in grid2:
                for g3 in grid3:
                    f = objective([g1, g2, g3], topology, outer_rot)
                    if f < best_f:
                        best_f = f
                        best_x = [g1, g2, g3]
                    if time.time() - start_time > time_budget:
                        return best_x, best_f, topology
    elif topology == 'hex_lattice':
        grid1 = np.linspace(1.0, 2.5, n_grid)
        grid2 = np.linspace(2.0, 4.5, n_grid)
        for g1 in grid1:
            for g2 in grid2:
                if g2 < g1 + 0.5:
                    continue
                f = objective([g1, g2], topology, outer_rot)
                if f < best_f:
                    best_f = f
                    best_x = [g1, g2]
                if time.time() - start_time > time_budget:
                    return best_x, best_f, topology
    elif topology == 'linear':
        grid1 = np.linspace(1.5, 3.0, n_grid)
        for g1 in grid1:
            f = objective([g1], topology, outer_rot)
            if f < best_f:
                best_f = f
                best_x = [g1]
            if time.time() - start_time > time_budget:
                return best_x, best_f, topology
    elif topology == 'cluster':
        grid1 = np.linspace(1.0, 2.5, n_grid)
        grid2 = np.linspace(1.5, 4.0, n_grid)
        for g1 in grid1:
            for g2 in grid2:
                if g2 < g1 + 0.5:
                    continue
                f = objective([g1, g2], topology, outer_rot)
                if f < best_f:
                    best_f = f
                    best_x = [g1, g2]
                if time.time() - start_time > time_budget:
                    return best_x, best_f, topology

    # SLSQP with smooth objective
    if time.time() - start_time < time_budget * 0.6:
        try:
            result = minimize(smooth_objective, best_x, args=(topology, outer_rot, 50.0),
                              method='SLSQP', bounds=bounds,
                              options={'maxiter': 300, 'ftol': 1e-6})
            if result.fun < best_f:
                best_x = result.x
                best_f = objective(result.x, topology, outer_rot)
        except Exception:
            pass

    # Nelder-Mead refinement
    if time.time() - start_time < time_budget * 0.8:
        try:
            result = minimize(objective, best_x, args=(topology, outer_rot),
                              method='Nelder-Mead', bounds=bounds,
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass

    # Final polish
    if time.time() - start_time < time_budget * 0.95:
        try:
            result = minimize(objective, best_x, args=(topology, outer_rot),
                              method='Nelder-Mead', bounds=bounds,
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass

    return best_x, best_f, topology

def optimize():
    best_side = 40.0
    best_params = None
    best_topology = None
    best_outer_rot = 0.0
    best_valid = False

    topologies = ['ring12', 'hex6+6', 'rect3x4', 'hex_lattice', 'linear', 'cluster']
    outer_rots = [0.0, 30.0, 60.0]

    time_per = 20.0

    for orot in outer_rots:
        for topo in topologies:
            if time.time() > DEADLINE:
                break
            params, f, topo_used = optimize_topology(topo, orot, time_per)
            if params is not None and f < best_side:
                if topo_used == 'ring12':
                    inner, center, orot_used = build_ring12(params[0], params[1], orot)
                elif topo_used == 'hex6+6':
                    inner, center, orot_used = build_hex6_plus6(params[0], params[1], params[2], params[3], orot)
                elif topo_used == 'rect3x4':
                    inner, center, orot_used = build_rect3x4(params[0], params[1], params[2], orot)
                elif topo_used == 'hex_lattice':
                    inner, center, orot_used = build_hex_lattice(params[0], params[1], orot)
                elif topo_used == 'linear':
                    inner, center, orot_used = build_linear_chain(params[0], orot)
                else:
                    inner, center, orot_used = build_hexagonal_cluster(params[0], params[1], orot)

                side = compute_min_outer_side(inner, center, orot_used)
                if check_validity(inner, center, side, orot_used):
                    best_side = side
                    best_params = params
                    best_topology = topo_used
                    best_outer_rot = orot
                    best_valid = True

    return best_side, best_params, best_topology, best_outer_rot, best_valid

def construct():
    t0 = time.time()
    try:
        side, params, topology, outer_rot, valid = optimize()
        if valid and params is not None:
            if topology == 'ring12':
                inner, center, orot = build_ring12(params[0], params[1], outer_rot)
            elif topology == 'hex6+6':
                inner, center, orot = build_hex6_plus6(params[0], params[1], params[2], params[3], outer_rot)
            elif topology == 'rect3x4':
                inner, center, orot = build_rect3x4(params[0], params[1], params[2], outer_rot)
            elif topology == 'hex_lattice':
                inner, center, orot = build_hex_lattice(params[0], params[1], outer_rot)
            elif topology == 'linear':
                inner, center, orot = build_linear_chain(params[0], outer_rot)
            else:
                inner, center, orot = build_hexagonal_cluster(params[0], params[1], outer_rot)
            return inner, center, side, orot
    except Exception:
        pass

    # Fallback: linear arrangement
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
