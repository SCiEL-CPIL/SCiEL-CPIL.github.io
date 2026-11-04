import numpy as np
import math

N = 11

def hexagon_support(angle, center, rot=0):
    # Support function of a unit hexagon
    # Vertices at distance 1
    # The support in direction n is max dot product with vertices
    # For a regular hexagon, the support is 1 / cos((angle - rot) mod 60 - 30) ?
    # No, simpler: the distance from center to side is sqrt(3)/2.
    # The support function h(n) for a regular hexagon with inradius r=sqrt(3)/2 is r / cos(theta) where theta is angle from normal to side.
    # But we can just compute max dot product with 6 vertices.
    vertices = []
    for i in range(6):
        a = rot + i * math.pi / 3
        vertices.append((center[0] + math.cos(a), center[1] + math.sin(a)))
    max_d = -1e9
    for v in vertices:
        d = v[0] * math.cos(angle) + v[1] * math.sin(angle)
        if d > max_d:
            max_d = d
    return max_d

def min_container_side(centers, inner_rots, container_rot):
    # Container normals
    max_proj = 0
    for k in range(6):
        n_angle = container_rot + k * math.pi / 3
        max_in = 0
        for i, c in enumerate(centers):
            h = hexagon_support(n_angle, c, inner_rots[i])
            if h > max_in:
                max_in = h
        # Container side S satisfies S * sqrt(3)/2 >= max_in
        # So S >= 2 * max_in / sqrt(3)
        s_required = 2 * max_in / math.sqrt(3)
        if s_required > max_proj:
            max_proj = s_required
    return max_proj

def construct():
    # Generate candidate centers
    # Hex grid basis
    u = (math.sqrt(3), 0)
    v = (math.sqrt(3)/2, 1.5)
    
    # Generate grid points
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            points.append((x, y))
            
    # Sort by distance from origin
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take first 11? This gives the compact cluster.
    centers = points[:11]
    
    # Optimize container rotation
    best_s = 1e9
    best_rot = 0
    for rot_deg in range(0, 30, 5):
        rot_rad = rot_deg * math.pi / 180
        s = min_container_side(centers, [0]*11, rot_rad)
        if s < best_s:
            best_s = s
            best_rot = rot_rad
            
    # Return in required format
    inner = [[c[0], c[1], 0.0] for c in centers]
    return inner, (0.0, 0.0), best_s, best_rot * 180 / math.pi
