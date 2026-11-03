"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon as (6,2) array."""
    angles = np.radians(angle_deg) + np.linspace(0, 2*np.pi, 7)[:-1]
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _convex_poly_overlap(v1, v2):
    """Check if two convex polygons (as (n,2) arrays) overlap using SAT."""
    for verts in [v1, v2]:
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i+1) % n]
            # Edge normal
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-12:
                continue
            normal = normal / np.linalg.norm(normal)
            
            # Project both polygons onto this normal
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return False
    return True

def _point_in_convex_poly(px, py, verts):
    """Check if point is inside convex polygon using cross products."""
    n = len(verts)
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i+1) % n]
        cross = (p2[0] - p1[0]) * (py - p1[1]) - (p2[1] - p1[1]) * (px - p1[0])
        if cross < 0:  # Assuming counter-clockwise
            return False
    return True

def _hexagon_fits_in_hexagon(inner_verts, outer_verts):
    """Check if inner hexagon fits inside outer hexagon."""
    for vx, vy in inner_verts:
        if not _point_in_convex_poly(vx, vy, outer_verts):
            return False
    return True

def _check_packing(inner_positions, outer_cx, outer_cy, outer_side, outer_angle):
    """Check if all inner hexagons fit inside outer hexagon and don't overlap."""
    outer_verts = _hexagon_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    # Check all inner hexagons fit in outer
    inner_verts_list = []
    for (cx, cy, ang) in inner_positions:
        iv = _hexagon_vertices(cx, cy, 1.0, ang)
        inner_verts_list.append(iv)
        if not _hexagon_fits_in_hexagon(iv, outer_verts):
            return False
    
    # Check no overlap between inner hexagons
    n = len(inner_positions)
    for i in range(n):
        for j in range(i+1, n):
            if _convex_poly_overlap(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def _find_min_outer_hexagon(inner_positions, cx0=0.0, cy0=0.0):
    """Find minimum side length outer hexagon that contains all inner hexagons."""
    best_side = float('inf')
    best_angle = 0.0
    
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        lo, hi = 1.0, 20.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if _check_packing(inner_positions, cx0, cy0, mid, outer_angle):
                hi = mid
            else:
                lo = mid
        if hi < best_side:
            best_side = hi
            best_angle = outer_angle
    
    return best_side, best_angle

def construct():
    sqrt3 = np.sqrt(3)
    
    def axial_to_cart(q, r, spacing=2.0):
        x = spacing * (q + r/2)
        y = spacing * r * sqrt3 / 2
        return (x, y)
    
    # Try several configurations
    configs = []
    
    # Config 1: Hex of radius 1 (7) + 4 extensions
    hex7 = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    configs.append(hex7 + [(2,0), (-2,0), (0,2), (0,-2)])
    
    # Config 2: Hex of radius 1 + 4 in different positions
    configs.append(hex7 + [(2,-1), (-2,1), (1,1), (-1,-1)])
    
    # Config 3: 3x4 grid-like
    configs.append([(q, r) for q in range(-2, 3) for r in range(-2, 3) 
                   if abs(q) + abs(r) + abs(q+r) <= 4][:11])
    
    # Config 4: Compact cluster
    configs.append([(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
                    (2,0), (-2,0), (1,-2), (-1,2)])
    
    best_result = None
    best_side = float('inf')
    
    for config in configs:
        inner = []
        for (q, r) in config:
            x, y = axial_to_cart(q, r)
            inner.append([x, y, 0.0])
        
        side, angle = _find_min_outer_hexagon(inner)
        if side < best_side:
            best_side = side
            best_result = (inner, (0.0, 0.0), side, angle)
    
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
