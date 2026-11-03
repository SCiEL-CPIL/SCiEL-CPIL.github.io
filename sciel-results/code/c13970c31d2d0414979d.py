import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12

# Precompute unit hexagon vertices (pointy top)
# Angles: 30, 90, 150, 210, 270, 330 degrees
unit_hex_vertices = np.array([
    [np.cos(np.pi/6 + k*np.pi/3), np.sin(np.pi/6 + k*np.pi/3)] for k in range(6)
])

# Outer hexagon normals (perpendicular to edges)
# If vertices are at 30, 90..., edges are between them.
# Normal to edge between 30 and 90 is at 60 deg.
# Normals at: 0, 60, 120, 180, 240, 300
outer_normals = np.array([
    [np.cos(k*np.pi/3), np.sin(k*np.pi/3)] for k in range(6)
])

def get_hex_vertices(center, angle, side=1.0):
    """Return vertices of a hexagon."""
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return center + (unit_hex_vertices * side).T @ rot.T

def point_in_hexagon_dist(point, center, side, angle=0.0):
    """
    Returns signed distance from point to hexagon boundary.
    Negative if inside, positive if outside.
    """
    # Translate point to local coords
    p = point - center
    # Rotate point to align with hexagon
    rot_inv = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
    p_local = rot_inv @ p
    
    # Calculate distance to each of the 6 half-planes
    apothem = np.sqrt(3) / 2 * side
    # Normals for local hexagon (aligned with axes)
    dists = np.dot(outer_normals, p_local)
    max_dist = np.max(dists)
    return max_dist - apothem

def check_overlap_and_penetration(hex1_verts, hex2_verts):
    """
    Checks if two convex hexagons overlap.
    Returns (overlap_depth, is_overlapping).
    Uses Separating Axis Theorem (SAT).
    If separated, returns (0, False).
    If overlapping, returns (max_penetration, True).
    """
    # The potential separating axes are the normals of the edges of hex1 and hex2.
    # For a regular hexagon, the edge normals are at angles 0, 60, 120, 180, 240, 300 (relative to the hexagon's orientation).
    # We need to project all vertices of both hexagons onto these axes.
    
    # Precompute normals for hex1 and hex2 in global frame
    # Hex1 normals
    n1 = hex1_verts[1] - hex1_verts[0]
    n1 = np.array([-n1[1], n1[0]])
    n1_norm = np.linalg.norm(n1)
    n1 = n1 / n1_norm if n1_norm > 1e-9 else np.array([1.0, 0.0])
    
    # Instead of calculating one normal and rotating, it's faster to use the known relative angles.
    # But since we don't pass angles here, we derive normals from vertices.
    # Actually, for regular hexagons, the 6 normals are just the directions from center to vertices of the *dual* hexagon?
    # No, the normals to the edges are perpendicular to the edges.
    # Edge 0-1 vector is V1-V0. Normal is perp.
    
    # Let's just generate the 12 candidate axes (6 from hex1, 6 from hex2)
    axes = []
    
    # Hex1 edges
    for i in range(6):
        p1 = hex1_verts[i]
        p2 = hex1_verts[(i+1)%6]
        edge = p2 - p1
        # Normal
        n = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-9:
            axes.append(n / norm)
            
    # Hex2 edges
    for i in range(6):
        p1 = hex2_verts[i]
        p2 = hex2_verts[(i+1)%6]
        edge = p2 - p1
        # Normal
        n = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-9:
            axes.append(n / norm)
            
    max_penetration = 0.0
    for axis in axes:
        # Project hex1 vertices onto axis
        proj1 = np.dot(hex1_verts, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        
        # Project hex2 vertices onto axis
        proj2 = np.dot(hex2_verts, axis)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 < min2 or max2 < min1:
            # Separated
            return 0.0, False
            
        # Calculate overlap on this axis
        overlap = min(max1, max2) - max(min1, min2)
        if overlap > max_penetration:
            max_penetration = overlap
            
    # If we get here, they are not separated on any axis, so they overlap.
    return max_penetration, True

def objective(params):
    """
    params: [x1, y1, th1, ..., x12, y12, th12, S]
    """
    S = params[-1]
    if S < 3.0: S = 3.0
    
    penalty = 0.0
    centers = []
    angles = []
    vertices_list = []
    
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        th = params[i*3+2]
        center = np.array([x, y])
        centers.append(center)
        angles.append(th)
        verts = get_hex_vertices(center, th)
        vertices_list.append(verts)
        
        # Containment penalty
        # Check if all vertices are inside the outer hexagon
        # Using half-plane distance is faster than checking all vertices against all edges
        # But we already have vertices. Let's check max distance to boundary.
        # If max_dist > apothem, it's outside.
        # We can compute the signed distance for each vertex to the outer hexagon boundary.
        # Outer hexagon is centered at (0,0) with side S and angle 0.
        # Apotem_outer = S * sqrt(3) / 2
        # Normals are outer_normals.
        
        # Vectorized containment check
        # p_local = verts - center_outer(0,0)
        # dists = dot(outer_normals, p_local.T)
        # max_dist = max(dists)
        # penetration = max(0, max_dist - apothem_outer)
        
        apothem_outer = S * np.sqrt(3) / 2
        # verts is (6, 2)
        # outer_normals is (6, 2)
        # We need dot product of each vertex with each normal? No.
        # The condition for point p to be inside hexagon with normals N and apothem A is:
        # max_i (N_i . p) <= A
        # So for each vertex v, we calculate max_i (N_i . v).
        
        # Efficiently:
        # D = verts @ outer_normals.T  -> (6, 6)
        # max_per_vertex = np.max(D, axis=1) -> (6,)
        # pen = np.max(np.maximum(0, max_per_vertex - apothem_outer))
        
        D = verts @ outer_normals.T
        max_per_vertex = np.max(D, axis=1)
        pen = np.max(np.maximum(0, max_per_vertex - apothem_outer))
        penalty += pen ** 2
        
    # Overlap penalty
    for i in range(N):
        for j in range(i+1, N):
            c1 = centers[i]
            c2 = centers[j]
            dist_c = np.linalg.norm(c1 - c2)
            # Circumradius is 1. If dist > 2.0, no overlap possible.
            if dist_c > 2.0:
                continue
            
            # Check overlap
            v1s = vertices_list[i]
            v2s = vertices_list[j]
            
            pen, is_over = check_overlap_and_penetration(v1s, v2s)
            
            if is_over:
                penalty += 10.0 * pen ** 2
            # else: no penalty
            
    return S + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Bounds: x, y in [-5, 5], theta in [0, pi/3], S in [3.0, 10.0]
    bounds = []
    for _ in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, np.pi/3))
    bounds.append((3.0, 10.0))
    
    # Heuristic initial point
    # Triangular lattice
    init = []
    idx = 0
    # 3 rows of 4
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * np.sqrt(3)
            y = (row - 1.0) * 1.5
            th = 0.0
            init.extend([x, y, th])
    init.append(5.0) # Initial S
    
    best_result = None
    best_score = np.inf
    
    # Run DE
    try:
        res = differential_evolution(
            objective, bounds, 
            maxiter=100, 
            popsize=15, 
            seed=42,
            init='sobol',
            polish=False, # We will polish manually to control time
            x0=init,
            tol=1e-4
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
            
    except Exception as e:
        pass
        
    # Check time
    if time.time() > deadline:
        if best_result is None:
             return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 10.0, 0.0
        inner = []
        for i in range(N):
            x = best_result[i*3]
            y = best_result[i*3+1]
            th = best_result[i*3+2]
            inner.append([x, y, np.degrees(th)])
        S = best_result[-1]
        return inner, (0.0, 0.0), S, 0.0

    # Polish with L-BFGS-B if time allows
    if best_result is not None:
        try:
            res_nm = minimize(
                objective, best_result, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_result = res_nm.x
        except Exception:
            pass
            
    if best_result is None:
        return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 10.0, 0.0
        
    inner = []
    for i in range(N):
        x = best_result[i*3]
        y = best_result[i*3+1]
        th = best_result[i*3+2]
        inner.append([x, y, np.degrees(th)])
    S = best_result[-1]
    return inner, (0.0, 0.0), S, 0.0
