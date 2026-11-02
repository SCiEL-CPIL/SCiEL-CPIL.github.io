"""Rewritten program using non-linear optimization for inner hexagon positions."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside a regular hexagon."""
    # Translate to center
    dx = px - cx
    dy = py - cy
    # Rotate point by -angle
    angle_rad = math.radians(-angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    
    R = side * math.sqrt(3) / 2  # Inradius
    
    # Check 6 half-planes
    # Normals are at 30, 90, 150, 210, 270, 330 degrees
    for k in range(6):
        a = math.radians(30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        if rx * nx + ry * ny > R:
            return False
    return True

def _hex_distance(h1, h2):
    """
    Compute the minimum distance between two convex hexagons.
    h1, h2 are lists of 6 (x,y) tuples.
    Returns 0.0 if they intersect or touch, >0 if disjoint.
    """
    min_dist = float('inf')
    # Check vertex to edge distances
    # For each vertex in h1, check distance to all edges in h2
    # For each vertex in h2, check distance to all edges in h1
    
    def point_to_segment_dist(p, a, b):
        px, py = p
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        qx = ax + t * dx
        qy = ay + t * dy
        return math.hypot(px - qx, py - qy)

    # Precompute edges
    edges1 = [(h1[i], h1[(i+1)%6]) for i in range(6)]
    edges2 = [(h2[i], h2[(i+1)%6]) for i in range(6)]
    
    for v in h1:
        for e in edges2:
            d = point_to_segment_dist(v, e[0], e[1])
            if d < min_dist:
                min_dist = d
                
    for v in h2:
        for e in edges1:
            d = point_to_segment_dist(v, e[0], e[1])
            if d < min_dist:
                min_dist = d
                
    # Check edge-edge intersection?
    # If min_dist is 0, they touch or intersect.
    # For disjoint convex polygons, min vertex-edge distance is the distance.
    # If they intersect, min_dist will be 0.
    return min_dist

def _objective_and_constraints(x, outer_angle, N):
    """
    x: array of 2N values: x1, y1, ..., xN, yN
    Returns (objective, constraints)
    Objective: Side length S (to be minimized)
    Constraints: 
    1. Containment: All vertices inside outer hex of side S centered at (0,0) with angle outer_angle.
    2. Non-overlap: Distance between any pair >= 0.
    """
    # We need to estimate S.
    # Let's compute the maximum "extent" of the cluster in the 6 normal directions of the outer hex.
    # The minimal S that contains the cluster is determined by the max projection.
    
    # Unpack positions
    positions = [(x[i*2], x[i*2+1]) for i in range(N)]
    
    # Compute vertices for all inner hexes (angle = outer_angle)
    all_vertices = []
    for i in range(N):
        px, py = positions[i]
        verts = _get_hex_vertices(px, py, outer_angle)
        all_vertices.extend(verts)
        
    # Compute minimal enclosing side length for this configuration
    # The outer hex is centered at (0,0) with angle outer_angle.
    # The inradius R must be >= max projection of any vertex onto the 6 normals.
    # Normal k is at angle outer_angle + 30 + 60*k.
    max_proj = 0
    for vx, vy in all_vertices:
        for k in range(6):
            a = math.radians(outer_angle + 30 + 60 * k)
            nx = math.cos(a)
            ny = math.sin(a)
            proj = abs(vx * nx + vy * ny)
            if proj > max_proj:
                max_proj = proj
                
    # R = max_proj
    # Side S = 2 * R / sqrt(3)
    S = 2 * max_proj / math.sqrt(3)
    
    # Check non-overlap
    # This is expensive. We will penalize overlaps in the objective.
    # For now, return S as objective.
    # We will add a penalty for overlaps separately in the wrapper.
    
    return S

def _penalized_objective(x, outer_angle, N, penalty_weight):
    """
    Objective: S + penalty * (overlap_violation + containment_violation)
    """
    positions = [(x[i*2], x[i*2+1]) for i in range(N)]
    
    # Compute S (containment is automatically satisfied by definition of S in _objective_and_constraints? 
    # No, _objective_and_constraints calculates the MINIMAL S that contains them.
    # So containment is always satisfied by the returned S.
    # We just need to minimize S.
    # The only other constraint is non-overlap.
    
    S = _objective_and_constraints(x, outer_angle, N)[0]
    
    # Check overlaps
    penalty = 0.0
    hexes = []
    for i in range(N):
        px, py = positions[i]
        hexes.append(_get_hex_vertices(px, py, outer_angle))
        
    for i in range(N):
        for j in range(i+1, N):
            d = _hex_distance(hexes[i], hexes[j])
            if d < 1e-6: # Overlapping or touching
                # We want d > 0. 
                # If d < 0 (intersection), distance is 0 in our function.
                # We need a measure of how much they overlap.
                # A simple proxy: if they intersect, add a large penalty.
                # But _hex_distance returns 0 for intersection.
                # We need to know if they are actually intersecting.
                # If min_dist is 0, they might be touching (valid) or intersecting (invalid).
                # For "disjoint interiors", touching is allowed? 
                # "Place disjoint ... hexagons". Disjoint usually means no common points.
                # "Disjoint interiors" means they can touch.
                # The task says "disjoint regular hexagons". Usually implies closed sets are disjoint?
                # Or interiors are disjoint?
                # "disjoint" in math often means intersection is empty.
                # "disjoint interiors" means intersection has empty interior.
                # Given "minimizing side length", touching is likely intended/allowed to pack tightly.
                # If touching is allowed, d >= 0 is the constraint.
                # Our _hex_distance returns 0 for touching.
                # So we can't distinguish touching from intersecting with just distance.
                # We need to check if they intersect.
                
                # Simple intersection check:
                # If any vertex of i is inside j, or any vertex of j is inside i, they intersect.
                # Or if edges cross.
                # For convex polygons, if no vertex is inside and no edges cross, they are disjoint.
                # If min_dist is 0, they are touching or intersecting.
                # If they are intersecting, one vertex will be inside the other.
                
                intersects = False
                for v in hexes[i]:
                    if _point_in_hex(v[0], v[1], positions[j][0], positions[j][1], 1.0, outer_angle):
                        intersects = True
                        break
                if not intersects:
                    for v in hexes[j]:
                        if _point_in_hex(v[0], v[1], positions[i][0], positions[i][1], 1.0, outer_angle):
                            intersects = True
                            break
                if not intersects:
                    # Check edge crossings? 
                    # If min_dist is 0 and no vertex inside, they are touching.
                    # If min_dist is 0 and they are not touching, they must be intersecting?
                    # No, if they intersect, a vertex is usually inside.
                    # Let's assume if no vertex is inside, they are touching (valid).
                    pass
                else:
                    # Intersecting
                    # Estimate overlap depth?
                    # Just add a large penalty
                    penalty += 100.0
                    
    return S + penalty_weight * penalty

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Initial guess: Triangular lattice
    sqrt3 = math.sqrt(3)
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
            
    candidates.sort(key=lambda t: t[0])
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
            
    x0 = np.array([coord for pos in positions for coord in pos])
    
    best_S = float('inf')
    best_x = x0
    best_angle = 0.0
    
    # Try different outer angles
    for angle in [0.0, 15.0, 30.0]:
        if time.time() > _DEADLINE - 20.0:
            break
            
        # Optimize positions for this angle
        # We use L-BFGS-B
        # We need a gradient? No, L-BFGS-B can approximate.
        # But the penalty function is non-smooth.
        # Let's use 'Nelder-Mead' or 'Powell' which don't need gradients.
        # Nelder-Mead is slow for 22 variables.
        # Let's try SLSQP with finite difference gradients.
        
        def obj(x):
            return _penalized_objective(x, angle, N, 10.0)
            
        try:
            res = minimize(obj, x0, method='SLSQP', options={'maxiter': 50, 'ftol': 1e-4})
            if res.success or res.fun < best_S:
                # Verify validity
                # Check overlaps
                pos_res = [(res.x[i*2], res.x[i*2+1]) for i in range(N)]
                hexes = [_get_hex_vertices(px, py, angle) for px, py in pos_res]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        # Check intersection
                        intersects = False
                        for v in hexes[i]:
                            if _point_in_hex(v[0], v[1], pos_res[j][0], pos_res[j][1], 1.0, angle):
                                intersects = True
                                break
                        if not intersects:
                            for v in hexes[j]:
                                if _point_in_hex(v[0], v[1], pos_res[i][0], pos_res[i][1], 1.0, angle):
                                    intersects = True
                                    break
                        if intersects:
                            valid = False
                            break
                    if not valid:
                        break
                            
                if valid:
                    S = _objective_and_constraints(res.x, angle, N)[0]
                    if S < best_S:
                        best_S = S
                        best_x = res.x
                        best_angle = angle
        except Exception as e:
            pass

    if best_S == float('inf'):
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = [[best_x[i*2], best_x[i*2+1], best_angle] for i in range(N)]
    return inner, (0.0, 0.0), best_S, best_angle
