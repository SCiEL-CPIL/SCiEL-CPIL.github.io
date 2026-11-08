"""Packing unit hexagons into minimal outer hexagon."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin
    
    # For hexagonal packing, use a hexagonal grid arrangement
    # Hexagonal numbers: 1, 7, 19, 37, 61, ...
    # Find minimal radius r such that hexagonal_number(r) >= N
    # hex_number(r) = 3*r*(r+1) + 1
    
    def hex_number(r):
        return 3 * r * (r + 1) + 1
    
    # Find minimal radius
    r = 0
    while hex_number(r) < N:
        r += 1
    
    # Generate positions for hexagonal cluster of radius r
    # In axial coordinates, hex at (q, q+r, q+r+1) where q+r+q+r+1 = 0... 
    # Actually for a hex of radius r centered at origin:
    # all (q, r_coord) where max(|q|, |r_coord|, |q+r_coord|) <= r
    
    positions = []
    for q in range(-r, r + 1):
        for r_coord in range(-r, r + 1):
            s = -q - r_coord
            if max(abs(q), abs(r_coord), abs(s)) <= r:
                positions.append((q, r_coord))
    
    # We need exactly N hexagons, so take first N from the cluster
    # Sort by distance from center for better packing
    positions.sort(key=lambda p: (p[0]**2 + p[1]**2 + p[0]*p[1], p[0], p[1]))
    positions = positions[:N]
    
    # Convert axial coordinates to Cartesian
    # For pointy-top hexagons with side length 1:
    # x = sqrt(3) * (q + r_coord/2)
    # y = 1.5 * r_coord
    # But we want flat-top or pointy-top? Let's use flat-top for hexagonal symmetry
    # Actually, for a regular hexagon with side 1, the distance from center to vertex is 1
    # and center to edge is sqrt(3)/2
    
    # Let's use the standard hex grid:
    # For flat-top hexagons:
    # x = 1.5 * q
    # y = sqrt(3) * (r_coord + q/2)
    
    # But we want to minimize the bounding hexagon. Let me think about this differently.
    # 
    # A unit regular hexagon has inradius sqrt(3)/2 and circumradius 1.
    # When packing in a hexagonal lattice, the centers are spaced by sqrt(3) in one direction
    # and 1.5 in the other (for flat-top), or similar.
    #
    # Actually, for two adjacent unit hexagons sharing an edge, the distance between centers
    # is 2 * inradius = sqrt(3).
    
    # Let me use a coordinate system where hexagons are arranged in a honeycomb.
    # For a hexagonal arrangement, the minimal bounding hexagon for a cluster of radius r
    # has side length 2*r + 1 (in terms of hexagon diameters) but we need to be careful.
    
    # Let me just compute the bounding box and then find the minimal enclosing hexagon.
    
    sqrt3 = math.sqrt(3)
    
    # Place hexagons in a hexagonal lattice
    # Using axial coordinates (q, r), convert to pixel coordinates
    # For flat-top hexagons with side length 1:
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    
    inner = []
    for q, r_coord in positions:
        x = 1.5 * q
        y = sqrt3 * (r_coord + 0.5 * q)
        inner.append([x, y, 0.0])
    
    # Now find the minimal enclosing regular hexagon
    # The outer hexagon should be centered at the centroid of the inner hexagons
    # and rotated to minimize its side length
    
    # Centroid
    cx = sum(p[0] for p in inner) / len(inner)
    cy = sum(p[1] for p in inner) / len(inner)
    
    # For a regular hexagon with side length s and rotation angle theta,
    # a point (x, y) is inside if the max over 6 directions of the projection
    # is <= s * cos(30 degrees) = s * sqrt(3)/2
    # Actually, the inradius of a regular hexagon with side s is s * sqrt(3)/2
    
    # The condition for a point to be inside a regular hexagon centered at (cx, cy)
    # with inradius R and orientation theta:
    # For each of the 6 edge normals at angles theta + k*60 degrees,
    # the projection of (point - center) onto that normal must be <= R
    
    # So we need: max_k |dot(point - center, normal_k)| <= R for all points
    # where R = s * sqrt(3)/2, so s = 2*R/sqrt(3)
    
    # To minimize s, we minimize R, which means minimizing the max projection over all points and all 6 normals.
    
    # For a given rotation theta, the required R is:
    # R(theta) = max over all points p of max over k of |dot(p - center, n_k(theta))|
    # where n_k(theta) = (cos(theta + k*60°), sin(theta + k*60°))
    
    # We want to minimize R(theta) over theta.
    
    # Sample theta values
    best_theta = 0.0
    best_R = float('inf')
    
    # Try many angles
    for i in range(360):
        theta = i * math.pi / 180.0  # degrees to radians
        max_proj = 0.0
        for px, py, _ in inner:
            dx = px - cx
            dy = py - cy
            for k in range(6):
                angle = theta + k * math.pi / 3.0
                nx = math.cos(angle)
                ny = math.sin(angle)
                proj = abs(dx * nx + dy * ny)
                if proj > max_proj:
                    max_proj = proj
        if max_proj < best_R:
            best_R = max_proj
            best_theta = i  # store in degrees
    
    # Refine around best_theta
    for i in range(-5, 6):
        theta_deg = best_theta + i * 0.1
        theta = theta_deg * math.pi / 180.0
        max_proj = 0.0
        for px, py, _ in inner:
            dx = px - cx
            dy = py - cy
            for k in range(6):
                angle = theta + k * math.pi / 3.0
                nx = math.cos(angle)
                ny = math.sin(angle)
                proj = abs(dx * nx + dy * ny)
                if proj > max_proj:
                    max_proj = proj
        if max_proj < best_R:
            best_R = max_proj
            best_theta = theta_deg
    
    # The inradius R = best_R, so side length s = 2 * R / sqrt(3)
    side_length = 2.0 * best_R / sqrt3
    
    # But wait - we also need to account for the fact that each inner hexagon has
    # circumradius 1. The point we're checking is the center of each inner hexagon.
    # The actual extent of each inner hexagon extends by 1 (circumradius) from its center.
    # So we need to add 1 to the required inradius.
    
    # Actually, the farthest point of an inner hexagon from its center is at distance 1 (circumradius).
    # So the maximum projection should include this: for each inner hexagon center, the farthest
    # point in any direction is at distance 1 from the center.
    # So the effective max projection for direction n is:
    # max over inner hexagons of (|dot(center - outer_center, n)| + 1)
    # because the hexagon extends 1 unit in any direction from its center.
    
    # Wait, that's not quite right either. The circumradius is 1, so in any direction,
    # the hexagon extends at most 1 unit from its center. So yes, we add 1.
    
    # Let me recompute with this correction
    best_theta2 = 0.0
    best_R2 = float('inf')
    
    for i in range(360):
        theta = i * math.pi / 180.0
        max_proj = 0.0
        for px, py, _ in inner:
            dx = px - cx
            dy = py - cy
            for k in range(6):
                angle = theta + k * math.pi / 3.0
                nx = math.cos(angle)
                ny = math.sin(angle)
                proj = abs(dx * nx + dy * ny) + 1.0  # add circumradius
                if proj > max_proj:
                    max_proj = proj
        if max_proj < best_R2:
            best_R2 = max_proj
            best_theta2 = i
    
    # Refine
    for i in range(-20, 21):
        theta_deg = best_theta2 + i * 0.05
        theta = theta_deg * math.pi / 180.0
        max_proj = 0.0
        for px, py, _ in inner:
            dx = px - cx
            dy = py - cy
            for k in range(6):
                angle = theta + k * math.pi / 3.0
                nx = math.cos(angle)
                ny = math.sin(angle)
                proj = abs(dx * nx + dy * ny) + 1.0
                if proj > max_proj:
                    max_proj = proj
        if max_proj < best_R2:
            best_R2 = max_proj
            best_theta2 = theta_deg
    
    side_length = 2.0 * best_R2 / sqrt3
    
    # Return the result
    # inner: list of [x, y, angle_degrees]
    # center: (x, y)
    # side: outer side length
    # angle: outer hexagon rotation in degrees
    
    return inner, (cx, cy), side_length, best_theta2
# EVOLVE-BLOCK-END
