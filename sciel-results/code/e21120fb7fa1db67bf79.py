import numpy as np
import time
from itertools import combinations

N = 11

def hex_vertices(cx, cy, angle_deg=0.0, side=1.0):
    """Return 6 vertices of a regular hexagon with given center, angle, side."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def min_enclosing_hex_side(inner_hexes, outer_angle_deg=0.0):
    """Compute minimum side length of a regular hexagon (at given angle) centered at origin 
    that contains all inner hexagons. Returns the side length."""
    all_verts = []
    for (cx, cy, ang) in inner_hexes:
        verts = hex_vertices(cx, cy, ang, 1.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    outer_angle = np.radians(outer_angle_deg)
    normals = []
    for i in range(6):
        theta = outer_angle + np.pi / 6 + i * np.pi / 3
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    normals = np.array(normals)
    
    projections = all_verts @ normals.T
    max_proj = projections.max(axis=0)
    
    s = max_proj.max() * 2.0 / np.sqrt(3.0)
    return s

def try_arrangement(inner_hexes, outer_angles=None):
    """Try multiple outer angles and return best side length and angle."""
    if outer_angles is None:
        outer_angles = np.linspace(0, 60, 25)
    best_s = float('inf')
    best_ang = 0.0
    for ang in outer_angles:
        s = min_enclosing_hex_side(inner_hexes, ang)
        if s < best_s:
            best_s = s
            best_ang = ang
    return best_s, best_ang

def construct():
    start_time = time.time()
    deadline = 145.0
    
    def axial_to_cart(q, r, scale=2.0):
        x = scale * (q + 0.5 * r)
        y = scale * (np.sqrt(3) / 2) * r
        return (x, y)
    
    # Center + ring1 (6) + 4 from ring2 (12)
    center = (0, 0)
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), 
             (-2,0), (-1,-1), (0,-2), (1,-2), (2,-2), (2,-1)]
    
    base_positions = [axial_to_cart(q, r) for (q, r) in [center] + ring1]
    ring2_cart = [axial_to_cart(q, r) for (q, r) in ring2]
    
    best_overall = float('inf')
    best_inner = None
    best_outer_side = float('inf')
    best_outer_angle = 0.0
    
    # Try all combinations of 4 from ring2
    for combo in combinations(range(12), 4):
        if time.time() > deadline:
            break
        all_pos = base_positions + [ring2_cart[i] for i in combo]
        inner = [(x, y, 0.0) for (x, y) in all_pos]
        s, ang = try_arrangement(inner)
        if s < best_overall:
            best_overall = s
            best_inner = inner
            best_outer_side = s
            best_outer_angle = ang
    
    # Optimize with small translations
    if best_inner is not None:
        for dx in np.linspace(-0.5, 0.5, 11):
            for dy in np.linspace(-0.5, 0.5, 11):
                if time.time() > deadline:
                    break
                shifted = [(x + dx, y + dy, ang) for (x, y, ang) in best_inner]
                s, ang = try_arrangement(shifted)
                if s < best_overall:
                    best_overall = s
                    best_inner = shifted
                    best_outer_side = s
                    best_outer_angle = ang
    
    if best_inner is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, (0.0, 0.0), best_outer_side, best_outer_angle

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
