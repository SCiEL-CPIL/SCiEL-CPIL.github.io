"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses Differential Evolution for global search and Nelder-Mead for local polish.
Includes robust validity checks with tolerance.
"""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
DEADLINE = time.time() + 130.0  # safety margin before 150s kill
EPS = 1e-7  # Tolerance for geometric checks


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def hex_vertices_array(cx, cy, s, angle_deg):
    """Vectorized version of hex_vertices returning numpy array."""
    a = math.radians(angle_deg)
    thetas = a + np.arange(6) * math.pi / 3.0
    return np.array([[cx + s * math.cos(t), cy + s * math.sin(t)] for t in thetas])


def sat_overlap(A, B):
    """SAT: True if two convex polygons intersect (strictly, not just touching).
    A, B are lists of (x,y) tuples."""
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
            # If separated by more than EPS, they don't overlap
            if maxA < minB - EPS or maxB < minA - EPS:
                return False
    return True


def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < EPS:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside."""
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


def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if sat_overlap(polys[i], polys[j]):
                return False
    return True


def build_initial_honeycomb():
    """
    Build a honeycomb-like initial configuration of 12 unit hexagons.
    Uses a compact arrangement: 1 center + 6 ring1 + 5 ring2.
    """
    a = 2.0  # standard spacing for touching hexagons
    positions = []
    positions.append((0.0, 0.0))
    for k in range(6):
        theta = math.radians(k * 60.0)
        positions.append((a * math.cos(theta), a * math.sin(theta)))
    for k in range(5):
        theta = math.radians(k * 60.0)
        positions.append((2.0 * a * math.cos(theta), 2.0 * a * math.sin(theta)))
    
    inner = [(x, y, 0.0) for x, y in positions]
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
    return inner, outer_center, outer_side, outer_angle


def build_initial_rectangular():
    """
    Build a 3x4 rectangular grid of hexagons.
    """
    dx = 2.0  # horizontal spacing
    dy = math.sqrt(3)  # vertical spacing for hex packing
    
    positions = []
    for row in range(2):
        for col in range(6):
            x = col * dx
            y = row * 2 * dy
            positions.append((x, y))
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    positions = [(x - cx, y - cy) for x, y in positions]
    
    inner = [(x, y, 0.0) for x, y in positions]
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
    return inner, outer_center, outer_side, outer_angle


def build_initial_compact():
    """
    Build a more compact initial arrangement using staggered rows.
    4-4-4 pattern with offset.
    """
    dx = 2.0
    dy = math.sqrt(3)
    
    positions = []
    # Row 0: 4 hexagons
    for col in range(4):
        positions.append((col * dx, 0.0))
    # Row 1: 4 hexagons, offset by dx/2
    for col in range(4):
        positions.append((col * dx + dx/2, dy))
    # Row 2: 4 hexagons
    for col in range(4):
        positions.append((col * dx, 2*dy))
    
    # Center
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    positions = [(x - cx, y - cy) for x, y in positions]
    
    inner = [(x, y, 0.0) for x, y in positions]
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
    return inner, outer_center, outer_side, outer_angle


def objective_de(params):
    """
    Objective for Differential Evolution.
    params: [x0,y0,theta0, ..., x11,y11,theta11, ocx, ocy, oangle]
    Note: oside is derived from the configuration to minimize it directly.
    We optimize positions and outer center/angle.
    Total: 12*3 + 3 = 39 parameters.
    """
    if time.time() > DEADLINE:
        return 1e6
    
    # Extract parameters
    inner_params = params[:36]
    ocx, ocy, oangle = params[36], params[37], params[38]
    
    # Build inner hexagons
    inner = []
    for i in range(N):
        x, y, theta = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        inner.append((x, y, theta))
    
    # Compute required outer side (this is the value we want to minimize)
    required_side = compute_min_outer_side(inner, (ocx, ocy), oangle)
    
    # Check non-overlap with SAT
    overlap_penalty = 0.0
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    
    for i in range(N):
        for j in range(i+1, N):
            if sat_overlap(polys[i], polys[j]):
                overlap_penalty += 1.0
    
    # Check containment
    outer = hex_vertices(ocx, ocy, required_side, oangle)
    contain_penalty = 0.0
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                contain_penalty += 1.0
    
    # Strong penalty for invalidity
    penalty = overlap_penalty * 100.0 + contain_penalty * 100.0
    return required_side + penalty


def optimize_free_placement(initial_inner, initial_center, initial_side, initial_angle, max_iter=500):
    """
    Optimize free placement using Differential Evolution + Nelder-Mead polish.
    """
    if time.time() > DEADLINE:
        return initial_inner, initial_center, initial_side, initial_angle
    
    # Build initial parameter vector for DE
    # We optimize x, y, angle for 12 hexes, and ox, oy, oangle for outer.
    # Outer side is derived.
    params_init = []
    for (x, y, ang) in initial_inner:
        params_init.extend([x, y, ang])
    params_init.extend([initial_center[0], initial_center[1], initial_angle])
    
    # Bounds for DE
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))  # x
        bounds.append((-10, 10))  # y
        bounds.append((-180, 180))  # angle
    bounds.append((-5, 5))  # ox
    bounds.append((-5, 5))  # oy
    bounds.append((-180, 180))  # oangle
    
    best_params = np.array(params_init)
    best_val = objective_de(best_params)
    
    # Run Differential Evolution
    try:
        # Use a subset of the population and iterations to stay within time
        # max_iter for DE is number of generations
        de_result = differential_evolution(
            objective_de,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=True  # Let DE do its own polish first
        )
        if de_result.fun < best_val:
            best_params = de_result.x
            best_val = de_result.fun
    except Exception:
        pass
    
    # Local Polish with Nelder-Mead
    try:
        nm_result = minimize(
            objective_de,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        if nm_result.fun < best_val:
            best_params = nm_result.x
            best_val = nm_result.fun
    except Exception:
        pass
    
    # Extract best solution
    opt_inner = []
    for i in range(N):
        opt_inner.append((best_params[3*i], best_params[3*i+1], best_params[3*i+2]))
    opt_center = (best_params[36], best_params[37])
    opt_angle = best_params[38]
    opt_side = compute_min_outer_side(opt_inner, opt_center, opt_angle)
    
    # Verify validity
    if check_validity(opt_inner, opt_center, opt_side, opt_angle):
        return opt_inner, opt_center, opt_side, opt_angle
    
    # If invalid, return initial
    return initial_inner, initial_center, initial_side, initial_angle


def polish_outer(inner, center, angle, max_iter=100):
    """
    Polish the outer hexagon parameters given fixed inner hexagons.
    """
    if time.time() > DEADLINE:
        return center, compute_min_outer_side(inner, center, angle), angle
    
    def outer_obj(params):
        ocx, ocy, oside, oangle = params
        required = compute_min_outer_side(inner, (ocx, ocy), oangle)
        if oside < required:
            return required + (required - oside) * 10
        # Check containment
        outer = hex_vertices(ocx, ocy, oside, oangle)
        for (x, y, ang) in inner:
            p = hex_vertices(x, y, 1.0, ang)
            for (px, py) in p:
                if not point_in_convex_polygon(px, py, outer):
                    return oside + 10
        return oside
    
    try:
        result = minimize(
            outer_obj,
            [center[0], center[1], compute_min_outer_side(inner, center, angle), angle],
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        if result.fun < 1e5:
            ocx, ocy, oside, oangle = result.x
            exact_side = compute_min_outer_side(inner, (ocx, ocy), oangle)
            return (ocx, ocy), exact_side, oangle
            
    except Exception:
        pass
    
    return center, compute_min_outer_side(inner, center, angle), angle


def construct():
    t0 = time.time()
    
    best_inner = None
    best_center = None
    best_side = 1e6
    best_angle = None
    
    # Try multiple initial configurations
    initial_configs = [
        build_initial_honeycomb(),
        build_initial_rectangular(),
        build_initial_compact(),
    ]
    
    for init_inner, init_center, init_side, init_angle in initial_configs:
        if time.time() > DEADLINE - 30:
            break
        
        # Optimize free placement
        opt_inner, opt_center, opt_side, opt_angle = optimize_free_placement(
            init_inner, init_center, init_side, init_angle
        )
        
        # Verify validity of the optimized result
        if check_validity(opt_inner, opt_center, opt_side, opt_angle):
            if opt_side < best_side:
                best_inner = opt_inner
                best_center = opt_center
                best_side = opt_side
                best_angle = opt_angle
                # If we found a very good solution, we might stop early
                if best_side < 4.0: 
                     # Theoretical lower bound is around 3.7-4.0 for 12 hexes? 
                     # Actually, 12 hexes in a honeycomb pattern of side 3 (radius 3) is 19 hexes.
                     # Side 2.5 fits 7. Side 3 fits 19.
                     # So 12 should fit in side ~2.8-3.0.
                     pass
    
    # If we found a valid solution, return it
    if best_inner is not None:
        return best_inner, best_center, best_side, best_angle
    
    # Fallback: simple honeycomb (ensure it's valid by adjusting side)
    inner, center, side, angle = build_initial_honeycomb()
    # Recompute side to ensure containment
    side = compute_min_outer_side(inner, center, angle)
    return inner, center, side, angle
