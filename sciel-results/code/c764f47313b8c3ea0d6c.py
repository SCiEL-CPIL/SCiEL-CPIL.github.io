import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def polygons_overlap_sat(A, B, tol=1e-9):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if abs(nx) < 1e-12 and abs(ny) < 1e-12:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - tol or maxB < minA - tol:
                return False
    return True


def point_in_convex_polygon(px, py, poly, tol=1e-9):
    """True if (px,py) is inside or on boundary of convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < tol:
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


def check_validity(inner, outer_center, outer_side, outer_angle):
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
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True


def build_initial_positions():
    """Generate initial positions for 12 hexagons using a ring topology."""
    inner = []
    # Center
    inner.append((0.0, 0.0, 0.0))
    # Ring 1: 6 hexagons at distance sqrt(3)
    r1 = math.sqrt(3.0)
    for k in range(6):
        theta = math.radians(k * 60.0)
        inner.append((r1 * math.cos(theta), r1 * math.sin(theta), 0.0))
    # Ring 2: 5 hexagons at larger distance
    r2 = 3.2
    for k in range(5):
        theta = math.radians(30.0 + k * 72.0)
        inner.append((r2 * math.cos(theta), r2 * math.sin(theta), 0.0))
    return inner


def objective_and_constraints(params, outer_angle):
    """
    Objective: minimize outer side length.
    params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11] (36 params)
    Returns (objective_value, constraint_values_list)
    """
    inner = []
    for i in range(N):
        x, y, a_deg = params[3*i], params[3*i+1], params[3*i+2]
        inner.append((x, y, a_deg))
    
    outer_center = (0.0, 0.0)
    side = compute_min_outer_side(inner, outer_center, outer_angle)
    
    # Constraints: pairwise non-overlap (using SAT)
    polys = [hex_vertices(x, y, 1.0, a) for (x, y, a) in inner]
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            # Constraint: polygons must not overlap
            # We use a distance-based proxy: ensure SAT says they don't overlap
            # For SLSQP, we need a scalar constraint >= 0
            # Use: 1 if no overlap, -1 if overlap -> but this is non-differentiable
            # Instead, use a smooth proxy based on minimum distance
            # For now, use a simple approach: check if they overlap
            overlap = polygons_overlap_sat(polys[i], polys[j], tol=-1e-6)
            # If overlapping, constraint violated (negative value)
            # We want constraint >= 0, so: -1 if overlap, 0 if no overlap
            # But this is discontinuous. Let's use a different approach.
            constraints.append(0.0 if not overlap else -1.0)
    
    return side, constraints


def optimize_direct(initial_inner, outer_angle, max_iter=50):
    """Optimize positions directly using SLSQP."""
    params = []
    for (x, y, a) in initial_inner:
        params.extend([x, y, a])
    params = np.array(params)
    
    def obj(p):
        if time.time() > DEADLINE - 2:
            return 1e6
        val, _ = objective_and_constraints(p, outer_angle)
        return val
    
    # For SLSQP, we need differentiable constraints
    # This is tricky with SAT. Let's use a penalty approach instead.
    
    best_side = 1e6
    best_params = params.copy()
    
    # Try multiple random perturbations
    random.seed(42)
    
    for attempt in range(3):
        if time.time() > DEADLINE - 5:
            break
        
        # Perturb initial positions
        p = params.copy()
        for i in range(len(p)):
            p[i] += random.gauss(0, 0.3)
        
        try:
            res = minimize(obj, p, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
            
            # Check validity
            inner_test = []
            for i in range(N):
                inner_test.append((res.x[3*i], res.x[3*i+1], res.x[3*i+2]))
            
            side_test = compute_min_outer_side(inner_test, (0,0), outer_angle)
            if check_validity(inner_test, (0,0), side_test, outer_angle):
                if side_test < best_side:
                    best_side = side_test
                    best_params = res.x.copy()
        except Exception:
            pass
    
    return best_side, best_params


def construct():
    t0 = time.time()
    
    best_side = 1e6
    best_result = None
    
    # Initial topology
    initial_inner = build_initial_positions()
    
    # Try different outer angles
    for outer_angle in [0.0, 15.0, 30.0]:
        if time.time() > DEADLINE - 10:
            break
        
        side, params = optimize_direct(initial_inner, outer_angle, max_iter=80)
        
        if side < best_side:
            inner = []
            for i in range(N):
                inner.append((params[3*i], params[3*i+1], params[3*i+2]))
            
            outer_center = (0.0, 0.0)
            exact_side = compute_min_outer_side(inner, outer_center, outer_angle)
            
            if check_validity(inner, outer_center, exact_side, outer_angle):
                best_side = exact_side
                best_result = (inner, outer_center, exact_side, outer_angle)
    
    # If optimization didn't find valid, use fallback
    if best_result is None:
        inner = build_initial_positions()
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        side = compute_min_outer_side(inner, outer_center, outer_angle)
        
        if check_validity(inner, outer_center, side, outer_angle):
            best_result = (inner, outer_center, side, outer_angle)
        else:
            # Very large fallback
            inner = [(i*4, 0, 0) for i in range(12)]
            outer_center = (22, 0)
            outer_angle = 0
            side = 10.0
            best_result = (inner, outer_center, side, outer_angle)
    
    return best_result
