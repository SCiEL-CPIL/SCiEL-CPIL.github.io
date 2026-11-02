import math
import numpy as np
from itertools import product

def _hex_vertices(cx, cy, angle_deg=0.0):
    """
    Generate vertices of a regular hexagon with side length 1,
    centered at (cx, cy), rotated by angle_deg.
    Vertices are at angles 0, 60, 120, 180, 240, 300 relative to the rotation.
    """
    rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = rad + i * math.pi / 3.0
        x = cx + math.cos(theta)
        y = cy + math.sin(theta)
        verts.append((x, y))
    return verts

def _axial_to_cartesian(q, r):
    """
    Convert axial coordinates to Cartesian coordinates for unit hexagons.
    Side length = 1.
    Distance between centers = sqrt(3).
    """
    x = 1.5 * q
    y = math.sqrt(3) * (r + q / 2.0)
    return x, y

def _get_neighbors(q, r):
    """
    Get the 6 neighbors of (q, r) in axial coordinates.
    """
    return [
        (q + 1, r),
        (q - 1, r),
        (q, r + 1),
        (q, r - 1),
        (q + 1, r - 1),
        (q - 1, r + 1)
    ]

def _min_outer_side_for_cluster(points, theta_deg):
    """
    Calculate the minimum side length S of a regular hexagon
    centered at (0,0) with rotation theta_deg
    that contains all given points.
    """
    rad = math.radians(theta_deg)
    # Normals to the sides of the hexagon
    # For a hexagon with "flat top" (vertices at 0, 60...),
    # the sides are at 30, 90, 150...
    # The normals point to the midpoints of the sides.
    # Midpoints are at 30, 90, 150, 210, 270, 330.
    # So normals are at 30, 90, 150 (and opposites).
    # If we rotate the hexagon by theta, normals are at 30+theta, 90+theta, 150+theta.
    
    # Wait, let's verify.
    # Vertices at 0, 60, 120...
    # Side between 0 and 60 is at angle 30.
    # Normal to this side is at 30.
    # So yes, normals are at 30 + theta, 90 + theta, 150 + theta.
    
    max_proj = 0.0
    for i in range(3):
        normal_angle = math.radians(30 + theta_deg + i * 60)
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        
        max_abs_proj = 0.0
        for x, y in points:
            proj = abs(x * nx + y * ny)
            if proj > max_abs_proj:
                max_abs_proj = proj
        
        if max_abs_proj > max_proj:
            max_proj = max_abs_proj
    
    # The apothem (distance from center to side) is S * sqrt(3) / 2
    # So S * sqrt(3) / 2 >= max_proj
    # S >= 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / math.sqrt(3)

def construct():
    # Start with central hexagon
    cluster = [(0, 0)]
    
    # Add first ring (6 hexagons)
    # These are the closest, so always add them
    neighbors = _get_neighbors(0, 0)
    for n in neighbors:
        cluster.append(n)
    
    # Greedily add 4 more hexagons
    # We want to minimize the bounding hexagon side length
    for _ in range(4):
        best_candidate = None
        best_score = float('inf')
        best_theta = 0.0
        
        # Generate candidates: neighbors of current cluster
        candidates = set()
        for q, r in cluster:
            for n in _get_neighbors(q, r):
                if n not in cluster:
                    candidates.add(n)
        
        # Evaluate each candidate
        for cand in candidates:
            temp_cluster = cluster + [cand]
            
            # Get Cartesian points for this temporary cluster
            points = []
            for q, r in temp_cluster:
                cx, cy = _axial_to_cartesian(q, r)
                verts = _hex_vertices(cx, cy, 0.0)
                points.extend(verts)
            
            # Find best theta for this cluster
            # Scan theta from 0 to 60 in steps of 1
            min_s = float('inf')
            min_theta = 0.0
            for theta in range(0, 61, 1):
                s = _min_outer_side_for_cluster(points, theta)
                if s < min_s:
                    min_s = s
                    min_theta = theta
            
            if min_s < best_score:
                best_score = min_s
                best_candidate = cand
                best_theta = min_theta
        
        # Commit the best candidate
        if best_candidate is not None:
            cluster.append(best_candidate)
    
    # Final refinement of theta for the final cluster
    points = []
    for q, r in cluster:
        cx, cy = _axial_to_cartesian(q, r)
        verts = _hex_vertices(cx, cy, 0.0)
        points.extend(verts)
    
    # Coarse scan
    min_s = float('inf')
    min_theta = 0.0
    for theta in range(0, 61, 1):
        s = _min_outer_side_for_cluster(points, theta)
        if s < min_s:
            min_s = s
            min_theta = theta
            
    # Fine scan around best theta
    for theta in np.linspace(min_theta - 2, min_theta + 2, 41):
        if theta < 0: theta += 60
        if theta >= 60: theta -= 60
        s = _min_outer_side_for_cluster(points, theta)
        if s < min_s:
            min_s = s
            min_theta = theta

    # Prepare output
    inner = []
    for q, r in cluster:
        x, y = _axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
        
    center = (0.0, 0.0)
    side = min_s
    angle = min_theta
    
    return inner, center, side, angle
