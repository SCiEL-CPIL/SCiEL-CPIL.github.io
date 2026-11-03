import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 140.0

def hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def polygons_overlap_sat(A, B, tol=1e-9):
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

def get_inner_hexes_from_params(params):
    """
    params: [r1, r2, phi1, phi2, global_angle]
    r1: radius of ring 1 (6 hexes)
    r2: radius of ring 2 (5 hexes)
    phi1: angular offset of ring 1
    phi2: angular offset of ring 2
    global_angle: rotation of the entire assembly
    """
    r1, r2, phi1, phi2, g_ang = params
    g_rad = math.radians(g_ang)
    
    inner = []
    # Center
    inner.append((0.0, 0.0, g_ang))
    
    # Ring 1: 6 hexagons
    for k in range(6):
        theta = math.radians(k * 60.0 + phi1) + g_rad
        x = r1 * math.cos(theta)
        y = r1 * math.sin(theta)
        inner.append((x, y, g_ang))
        
    # Ring 2: 5 hexagons
    # Ideally placed in the gaps.
    # Gaps are at 30, 90, 150, 210, 270, 330 relative to ring 1 vertices if phi1=0.
    # Let's just use a uniform distribution with offset phi2.
    for k in range(5):
        theta = math.radians(k * 72.0 + phi2) + g_rad
        x = r2 * math.cos(theta)
        y = r2 * math.sin(theta)
        inner.append((x, y, g_ang))
        
    return inner

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

def check_validity(inner, outer_center, outer_side, outer_angle):
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

def objective(params):
    if time.time() > DEADLINE - 2:
        return 1e6
    inner = get_inner_hexes_from_params(params)
    outer_center = (0.0, 0.0)
    outer_angle = 0.0 # We optimize global_angle in params, but outer is fixed at 0 for calculation?
    # Wait, if we rotate the inner assembly, the outer hexagon can stay at 0.
    # The "global_angle" in params rotates the inner hexagons.
    # The outer hexagon angle is fixed to 0 for the purpose of finding the min side length.
    # The returned outer_angle should be 0.
    
    side = compute_min_outer_side(inner, outer_center, 0.0)
    
    # Check validity
    if not check_validity(inner, outer_center, side, 0.0):
        return 1e6 + side # Penalty
    
    return side

def construct():
    t0 = time.time()
    best_side = 1e6
    best_params = None
    
    # Initial guesses for r1, r2, phi1, phi2, g_ang
    # r1 approx sqrt(3) ~ 1.732
    # r2 approx 3.0 - 3.5
    # phi1, phi2 small
    # g_ang 0-60
    
    random.seed(42)
    
    # Try a few random starting points
    for _ in range(20):
        if time.time() > DEADLINE - 10:
            break
            
        r1 = random.uniform(1.5, 2.0)
        r2 = random.uniform(2.8, 3.8)
        phi1 = random.uniform(0, 60)
        phi2 = random.uniform(0, 72)
        g_ang = random.uniform(0, 60)
        
        p0 = np.array([r1, r2, phi1, phi2, g_ang])
        
        try:
            res = minimize(objective, p0, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_side and res.fun < 1e5:
                # Verify final result
                inner = get_inner_hexes_from_params(res.x)
                outer_center = (0.0, 0.0)
                outer_angle = 0.0
                exact_side = compute_min_outer_side(inner, outer_center, outer_angle)
                
                if check_validity(inner, outer_center, exact_side, outer_angle):
                    best_side = exact_side
                    best_params = res.x.copy()
                    
        except Exception:
            pass

    if best_params is None:
        # Fallback to a known valid configuration
        # 1-6-5 ring with standard spacing
        r1 = math.sqrt(3)
        r2 = 3.2
        phi1 = 0
        phi2 = 30
        g_ang = 0
        best_params = np.array([r1, r2, phi1, phi2, g_ang])
        
        inner = get_inner_hexes_from_params(best_params)
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        exact_side = compute_min_outer_side(inner, outer_center, outer_angle)
        
        if not check_validity(inner, outer_center, exact_side, outer_angle):
            # If even fallback fails, expand r2
            best_params[1] = 4.0
            inner = get_inner_hexes_from_params(best_params)
            exact_side = compute_min_outer_side(inner, outer_center, outer_angle)

    inner = get_inner_hexes_from_params(best_params)
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    exact_side = compute_min_outer_side(inner, outer_center, outer_angle)
    
    return (inner, outer_center, exact_side, outer_angle)
