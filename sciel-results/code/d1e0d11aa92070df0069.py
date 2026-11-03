"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses multi-topology lattice packing with topology search and local refinement.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

# --- Geometry helpers ---

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon (CCW or CW)."""
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
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # normal to edge
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            # project A
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            # project B
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False  # separating axis found
    return True

def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping.
    inner: list of (x, y, angle_deg) with unit side."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    # inside outer
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    # pairwise non-overlap (allow touching)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside.
    For a regular hexagon of side s, apothem = s * sqrt(3)/2.
    A point is inside iff for each of the 6 outward normals n_k, (p - c)·n_k <= apothem.
    So S >= (2/sqrt(3)) * max over all vertices and all k of (p-c)·n_k.
    """
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

# --- Multi-topology lattice packing ---

def build_ring12(scale, inner_rot, outer_rot):
    """Ring of 12 hexagons around center."""
    inner = []
    for k in range(12):
        theta = math.radians(inner_rot + k * 30.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        inner.append((x, y, inner_rot))
    return inner, (0.0, 0.0), outer_rot

def build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot):
    """Two concentric rings of 6, no center."""
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
    """3x4 rectangular grid."""
    inner = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * scale_x
            y = (j - 1.5) * scale_y
            inner.append((x, y, rot))
    return inner, (0.0, 0.0), outer_rot

def objective(params, topology, outer_rot):
    """Objective function for optimization."""
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
    else:
        return 1000.0
    
    side = compute_min_outer_side(inner, center, orot)
    if not check_validity(inner, center, side, orot):
        return side + 10.0  # penalty for invalid
    return side

def optimize_topology(topology, outer_rot, time_budget):
    """Optimize a single topology."""
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
    else:
        return None, None, None
    
    best_x = x0
    best_f = objective(x0, topology, outer_rot)
    
    # Coarse grid search
    n_grid = 8
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
    
    # Local refinement with scipy
    if time.time() - start_time < time_budget * 0.7:
        try:
            result = minimize(objective, best_x, args=(topology, outer_rot),
                             method='Nelder-Mead', bounds=bounds,
                             options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass
    
    return best_x, best_f, topology

def optimize():
    """Try all topologies and outer rotations."""
    best_side = 40.0
    best_params = None
    best_topology = None
    best_outer_rot = 0.0
    best_valid = False
    
    topologies = ['ring12', 'hex6+6', 'rect3x4']
    outer_rots = [0.0, 30.0, 60.0]
    
    time_per = 30.0  # seconds per topology-outer_rot combination
    
    for orot in outer_rots:
        for topo in topologies:
            if time.time() > DEADLINE:
                break
            params, f, topo_used = optimize_topology(topo, orot, time_per)
            if params is not None and f < best_side:
                # Verify validity
                if topo_used == 'ring12':
                    inner, center, orot_used = build_ring12(params[0], params[1], orot)
                elif topo_used == 'hex6+6':
                    inner, center, orot_used = build_hex6_plus6(params[0], params[1], params[2], params[3], orot)
                else:
                    inner, center, orot_used = build_rect3x4(params[0], params[1], params[2], orot)
                
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
            else:
                inner, center, orot = build_rect3x4(params[0], params[1], params[2], outer_rot)
            return inner, center, side, orot
    except Exception:
        pass

    # Fallback: linear arrangement (always valid, side ~ 40)
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
