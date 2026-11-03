import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin within 150 s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(vertices, outer_angle_deg):
    """
    Find the minimum side length of a regular hexagon with given angle
    that contains all given vertices.
    Returns (center_x, center_y, side_length).
    """
    # Normals of the outer hexagon sides
    # If vertices are at angles theta + 60k, normals are at theta + 30 + 60k
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    # LP Variables: [cx, cy, R] (R is inradius)
    # Minimize R
    # Constraints:
    # For each vertex v and normal n:
    # (v - c) . n <= R  =>  -c . n - R <= -v . n
    # (c - v) . n <= R  =>   c . n - R <=  v . n
    
    A = []
    b = []
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            
            # Constraint 1: -nx*cx - ny*cy - R <= -v_dot_n
            A.append([-nx, -ny, -1.0])
            b.append(-v_dot_n)
            
            # Constraint 2: nx*cx + ny*cy - R <= v_dot_n
            A.append([nx, ny, -1.0])
            b.append(v_dot_n)
            
    # R >= 0
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3) / 2)
            return cx, cy, side
        else:
            return None
    except:
        return None

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Emergency fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    sqrt3 = math.sqrt(3)
    # Use a very small scale factor to ensure strict disjointness while minimizing gap
    scale = 1.000001
    
    # Generate 11 points on triangular lattice
    # Center (0,0), 6 neighbors, 4 from next ring
    # Axial coordinates:
    # (0,0)
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # (2,0), (2,-1), (1,-2), (-1,2), (-2,1), (-2,2) -> Pick 4 closest?
    # Actually, just generate a grid and pick 11 closest to origin.
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * sqrt3 * scale + j * (sqrt3 / 2) * scale
            y = j * 1.5 * scale
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    seen = set()
    for _, x, y in candidates:
        # Deduplicate just in case, though lattice points are unique
        key = (round(x, 6), round(y, 6))
        if key not in seen:
            seen.add(key)
            positions.append((x, y))
            if len(positions) == N:
                break
            
    if len(positions) < N:
        # Fallback if something went wrong
        positions = [(k * sqrt3 * scale, 0.0) for k in range(N)]

    # Inner hexagons are all aligned (angle 0 or 30 doesn't matter for the shape, 
    # but we must pick one. Let's use 0.0 (flat top) or 30.0 (pointy top).
    # The lattice generated above assumes flat-top hexagons?
    # Distance between centers is sqrt(3) * scale.
    # For flat-top hexagons, the distance between centers of adjacent hexagons is sqrt(3) * side.
    # So this lattice corresponds to flat-top hexagons (angle 0).
    inner_angle = 0.0
    best_inner = [[px, py, inner_angle] for px, py in positions]
    
    # Get all vertices for the LP
    all_vertices = []
    for px, py in positions:
        verts = _get_hex_vertices(px, py, inner_angle)
        all_vertices.extend(verts)
        
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Coarse sweep of outer angle (0 to 30 degrees)
    # Due to symmetry, 0-30 covers all unique orientations.
    coarse_angles = np.arange(0, 30.1, 0.5)
    
    for ang in coarse_angles:
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(all_vertices, ang)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_outer_angle = ang
                
    # Fine sweep around the best coarse angle
    if best_outer_angle > 0:
        # Search +/- 0.5 degrees with 0.01 step
        fine_angles = np.arange(best_outer_angle - 0.5, best_outer_angle + 0.5 + 0.01, 0.01)
        for ang in fine_angles:
            if ang < 0 or ang > 30:
                continue
            if time.time() > _DEADLINE - 5.0:
                break
            res = _min_enclosing_hex_side_lp(all_vertices, ang)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_outer_angle = ang
                    
    # Very fine sweep around the best fine angle
    if best_outer_angle > 0:
        very_fine_angles = np.arange(best_outer_angle - 0.05, best_outer_angle + 0.05 + 0.001, 0.001)
        for ang in very_fine_angles:
            if ang < 0 or ang > 30:
                continue
            if time.time() > _DEADLINE - 2.0:
                break
            res = _min_enclosing_hex_side_lp(all_vertices, ang)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_outer_angle = ang

    # If we have time, try to optimize the positions slightly?
    # Given the lattice is optimal for density, and the LP finds the exact container,
    # moving points might not help unless we can "tilt" the lattice, which we already did via outer angle.
    # So we return the lattice solution.
    
    return best_inner, best_center, best_side, best_outer_angle
