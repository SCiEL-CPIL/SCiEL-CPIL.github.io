import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given angle."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def _outer_contains(outer_center, outer_side, outer_angle_deg, inner_verts):
    """Check if outer hexagon contains all inner vertices. Returns min required side."""
    oc = np.array(outer_center)
    oa = np.radians(outer_angle_deg)
    # Outer hexagon vertices
    outer_verts = []
    for i in range(6):
        theta = oa + i * np.pi / 3
        outer_verts.append(oc + outer_side * np.array([np.cos(theta), np.sin(theta)]))
    outer_verts = np.array(outer_verts)
    
    # For containment, check each inner vertex is inside outer hexagon
    # Use the 6 half-plane constraints
    # Each edge of outer hexagon defines a half-plane
    min_side = 0.0
    for iv in inner_verts:
        # For each edge, compute the required side length
        for i in range(6):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % 6]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-15)
            # The distance from center to this edge (apothem) = side * sqrt(3)/2
            # Point must satisfy: dot(point - oc, normal) <= apothem
            # dot(iv - oc, normal) <= side * sqrt(3)/2
            d = np.dot(iv - oc, normal)
            if d > 0:
                req_side = d / (np.sqrt(3) / 2)
                min_side = max(min_side, req_side)
    return min_side

def _objective(params, outer_angle_deg):
    """Objective: outer side length needed to contain all inner hexagons."""
    n = N
    # params: 3*n values: (x, y, angle_deg) for each inner hex
    inner_verts_all = []
    for i in range(n):
        x, y, ang = params[3*i], params[3*i+1], params[3*i+2]
        verts = _hex_vertices(x, y, ang)
        inner_verts_all.append(verts)
    inner_verts_all = np.concatenate(inner_verts_all, axis=0)
    
    # Also need to ensure inner hexagons don't overlap
    # For now, just minimize outer side; add penalty for overlaps
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, a1 = params[3*i], params[3*i+1], params[3*i+2]
            x2, y2, a2 = params[3*j], params[3*j+1], params[3*j+2]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            # Minimum distance between centers of two non-overlapping unit hexagons
            # When optimally oriented, min center distance is 2*sqrt(3) for edge-sharing
            # But with rotation, it can be less. Conservative: 2*sqrt(3)
            min_dist = 2 * np.sqrt(3)
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    outer_center = [0.0, 0.0]
    # Compute required outer side
    req_side = _outer_contains(outer_center, 100.0, outer_angle_deg, inner_verts_all)
    
    return req_side + 100 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    best_result = None
    best_side = float('inf')
    
    # Initial configuration: hexagonal lattice
    # 1 center + 6 in ring 1 + 4 in ring 2
    spacing = 2 * np.sqrt(3)
    
    # Ring 1: 6 hexagons at 60-degree intervals
    ring1 = []
    for i in range(6):
        theta = i * np.pi / 3
        ring1.append([spacing * np.cos(theta), spacing * np.sin(theta), 0.0])
    
    # Ring 2: select 4 positions from the 12 possible
    # Positions at distance 2*spacing from center, at angles offset by 30 degrees
    ring2_candidates = []
    for i in range(12):
        theta = i * np.pi / 6 + np.pi / 6  # offset by 30 deg
        r = 2 * spacing * np.cos(np.pi/6)  # actual radial distance for hex lattice
        # In hex lattice, ring 2 positions are at various distances
        # Let me use the proper hex lattice coordinates
        pass
    
    # Better: use axial coordinates for hex grid
    # Axial coords to pixel: x = size * (q + r/2), y = size * r * sqrt(3)/2
    # For unit hexagons, "size" (center to vertex) = 1
    # Center to center of adjacent = 2 * apothem = 2 * sqrt(3)/2 * ... 
    # Actually center-to-center of touching hexagons = 2 * apothem = sqrt(3) * 2 = 2*sqrt(3)? 
    # No: apothem = sqrt(3)/2 * side = sqrt(3)/2. Center to center = 2 * apothem = sqrt(3).
    # Wait, that's for the inradius. Let me recalculate.
    # Regular hexagon side s: circumradius = s, inradius = s*sqrt(3)/2
    # Two hexagons sharing an edge: center distance = 2 * inradius = 2 * sqrt(3)/2 = sqrt(3)
    
    spacing = np.sqrt(3)
    
    # Axial coordinate system: q, r
    # Pixel: x = spacing * (q + r/2), y = spacing * r * sqrt(3)/2
    # Wait, that's not right either. Let me use the standard.
    # For pointy-top hexagons: x = size * sqrt(3) * (q + r/2), y = size * 3/2 * r
    # For flat-top: x = size * 3/2 * q, y = size * sqrt(3) * (r + q/2)
    
    # Let me just place them manually for a good initial config
    # Center at origin
    init = [[0.0, 0.0, 0.0]]
    # 6 around center
    for i in range(6):
        theta = i * np.pi / 3
        init.append([spacing * np.cos(theta), spacing * np.sin(theta), 0.0])
    # 4 more: pick 4 from ring 2
    # Ring 2 in hex grid: positions like (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1) in axial
    # Convert axial to cartesian for flat-top hexagons
    # Flat-top: x = 1.5 * q, y = sqrt(3) * (r + q/2)  [in units of side length]
    # Center to center = sqrt(3) * side... let me just use the spacing
    
    # Ring 2 axial coords (q,r): (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1)
    ring2_axial = [(2,0), (1,1), (0,2), (-1,1)]
    for q, r in ring2_axial:
        x = spacing * (q + r/2) * 2  # adjust
        y = spacing * r * np.sqrt(3) / 2 * 2
        # Let me just compute directly
        x = 1.5 * spacing * q
        y = np.sqrt(3) * spacing * (r + q/2)
        init.append([x, y, 0.0])
    
    # That gives 11 hexagons. Now optimize.
    x0 = np.array(init).flatten()
    
    for outer_rot in [0.0, 30.0, 60.0]:
        if time.time() > deadline - 30:
            break
        
        result = minimize(_objective, x0, args=(outer_rot,), method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-8})
        
        if result.success or result.fun < best_side:
            # Compute actual outer side
            params = result.x
            inner_verts_all = []
            for i in range(N):
                x, y, ang = params[3*i], params[3*i+1], params[3*i+2]
                verts = _hex_vertices(x, y, ang)
                inner_verts_all.append(verts)
            inner_verts_all = np.concatenate(inner_verts_all, axis=0)
            
            outer_center = [0.0, 0.0]
            req_side = _outer_contains(outer_center, 100.0, outer_rot, inner_verts_all)
            
            if req_side < best_side:
                best_side = req_side
                inner_list = []
                for i in range(N):
                    inner_list.append([params[3*i], params[3*i+1], params[3*i+2]])
                best_result = (inner_list, outer_center, req_side, outer_rot)
        
        if time.time() > deadline - 10:
            break
    
    if best_result is None:
        # Fallback: return initial config with estimated side
        inner_list = [list(row) for row in init]
        inner_verts_all = []
        for x, y, ang in init:
            verts = _hex_vertices(x, y, ang)
            inner_verts_all.append(verts)
        inner_verts_all = np.concatenate(inner_verts_all, axis=0)
        outer_center = [0.0, 0.0]
        req_side = _outer_contains(outer_center, 100.0, 0.0, inner_verts_all)
        best_result = (inner_list, outer_center, req_side, 0.0)
    
    return best_result

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
