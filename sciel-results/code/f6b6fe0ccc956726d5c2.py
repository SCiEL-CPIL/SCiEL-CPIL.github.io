import time
import math
import numpy as np

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    verts = []
    for i in range(6):
        theta = math.radians(60 * i + 30)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        rx = cx + x * cos_a - y * sin_a
        ry = cy + x * sin_a + y * cos_a
        verts.append((rx, ry))
    return verts

def _check_non_overlap(inner, outer_side, outer_center, outer_angle, eps=1e-6):
    # Fast circle separation check: distance between centers >= 2 * inradius
    # Inradius of unit hex is sqrt(3)/2. Separation required is sqrt(3).
    min_dist = math.sqrt(3) - eps
    n = len(inner)
    for i in range(n):
        x1, y1, a1 = inner[i]
        for j in range(i + 1, n):
            x2, y2, a2 = inner[j]
            dx = x1 - x2
            dy = y1 - y2
            if dx*dx + dy*dy < min_dist*min_dist:
                return False
    return True

def _compute_outer_side(inner, outer_center, outer_angle):
    # Find minimum side length S and rotation R such that all inner hexes are inside.
    # We approximate by finding the minimum enclosing hexagon.
    # For a regular hexagon, the width in direction theta is S * (cos(theta - R) + cos(theta - R + 60) + ... )? 
    # Actually, support function of regular hexagon of side S, center 0, rotated R:
    # h(u) = S * max_{k=0..5} (u . v_k) where v_k are vertices of unit hexagon rotated R.
    # We need h_outer(u) >= h_inner_i(u) for all u, for all i.
    # This is equivalent to: for all i, for all u, S * h_unit(u - R) >= h_inner_i(u - R).
    # Let's sample angles to find the required S.
    
    # To be fast and accurate enough, sample 360 angles.
    max_req = 0.0
    for deg in range(0, 360, 1):
        theta = math.radians(deg)
        # Direction vector
        ux, uy = math.cos(theta), math.sin(theta)
        
        # Max projection of inner hexes in this direction
        max_proj_inner = -1e9
        for x, y, a_deg in inner:
            a = math.radians(a_deg)
            cos_a, sin_a = math.cos(a), math.sin(a)
            # Project 6 vertices of this inner hex
            for k in range(6):
                t = math.radians(60 * k + 30)
                vx = math.cos(t)
                vy = math.sin(t)
                # Local vertex
                lx = vx
                ly = vy
                # Rotated local vertex
                rx = lx * cos_a - ly * sin_a
                ry = lx * sin_a + ly * cos_a
                # Global vertex
                gx = x + rx
                gy = y + ry
                proj = gx * ux + gy * uy
                if proj > max_proj_inner:
                    max_proj_inner = proj
        
        # Min projection of inner hexes in this direction (for the opposite side)
        min_proj_inner = 1e9
        for x, y, a_deg in inner:
            a = math.radians(a_deg)
            cos_a, sin_a = math.cos(a), math.sin(a)
            for k in range(6):
                t = math.radians(60 * k + 30)
                vx = math.cos(t)
                vy = math.sin(t)
                lx = vx
                ly = vy
                rx = lx * cos_a - ly * sin_a
                ry = lx * sin_a + ly * cos_a
                gx = x + rx
                gy = y + ry
                proj = gx * ux + gy * uy
                if proj < min_proj_inner:
                    min_proj_inner = proj

        # The outer hexagon centered at (cx, cy) must contain these.
        # Shift to outer center
        max_proj_inner -= outer_center[0] * ux + outer_center[1] * uy
        min_proj_inner -= outer_center[0] * ux + outer_center[1] * uy
        
        # The outer hexagon's support in direction u is S * h_unit(u - R)
        # And in direction -u is S * h_unit(-u - R)
        # We need S * h_unit(u - R) >= max_proj_inner
        # and S * h_unit(-u - R) >= -min_proj_inner
        
        # Calculate h_unit for unit hexagon (side 1) rotated by R
        # h_unit(v) = max_{k} (v . w_k) where w_k are unit hex vertices (side 1)
        # Wait, standard regular hexagon with side 1 has circumradius 1.
        # Vertices are at distance 1.
        # h_unit(v) = max( cos(angle_diff + 60k) )? No.
        # h_unit(v) = max_{k=0..5} (v . (cos(60k+30), sin(60k+30)))? 
        # Actually, vertices of unit hex (side 1) are at angles 30, 90, 150, 210, 270, 330.
        # Let's compute h_unit directly.
        
        R = math.radians(outer_angle)
        cos_R, sin_R = math.cos(R), math.sin(R)
        
        # Rotate u by -R to get u' in hex frame
        ux_p = ux * cos_R + uy * sin_R
        uy_p = -ux * sin_R + uy * cos_R
        
        # h_unit(u')
        h1 = 0.0
        h2 = 0.0
        # -u in hex frame
        nux_p = -ux_p
        nuy_p = -uy_p
        
        for k in range(6):
            t = math.radians(60 * k + 30)
            wx = math.cos(t)
            wy = math.sin(t)
            p1 = ux_p * wx + uy_p * wy
            p2 = nux_p * wx + nuy_p * wy
            if p1 > h1: h1 = p1
            if p2 > h2: h2 = p2
            
        req1 = max_proj_inner / h1 if h1 > 1e-9 else 0
        req2 = -min_proj_inner / h2 if h2 > 1e-9 else 0
        req = max(req1, req2)
        if req > max_req:
            max_req = req
            
    return max_req

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial configuration: Hexagonal lattice
    # Generate positions
    # Flat-top or pointy-top? Let's use pointy-top for standard hex grid.
    # Spacing: sqrt(3) in x, 1.5 in y for offset rows?
    # Let's use axial coordinates.
    # q, r
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # Side length 1.
    
    positions = []
    # Generate a cluster of 11 hexes.
    # Center (0,0)
    # 6 neighbors
    # 4 more to make 11? Or just a 3x3-ish blob?
    # Let's try a compact shape.
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1) -> 7
    # (2,0), (-2,0), (0,2), (0,-2) -> 11?
    # Let's just use a hexagonal number shape.
    # k=0: 1
    # k=1: 7
    # k=2: 19 (too many)
    # We need 11.
    # Let's take the 7 neighbors + 4 more.
    coords = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    # Add (2,0), (-2,0), (0,2), (0,-2)
    coords += [(2,0), (-2,0), (0,2), (0,-2)]
    
    # Convert to cartesian
    inner = []
    for q, r in coords:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    # Initial outer
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    outer_side = _compute_outer_side(inner, outer_center, outer_angle)
    
    best_side = outer_side
    best_inner = [row[:] for row in inner]
    best_center = outer_center
    best_angle = outer_angle
    
    # Simulated Annealing
    # Parameters to perturb:
    # 1. Global rotation of the whole cluster
    # 2. Shear / skew of the lattice? Hard to do with fixed coords.
    # Let's perturb individual positions slightly? No, that breaks the lattice structure.
    # Better: Perturb the global rotation and the outer hexagon rotation/center.
    # And maybe a global scale? No, side length is fixed at 1.
    
    # Actually, rotating the inner cluster relative to the outer hexagon can help.
    # Also, the outer hexagon can be rotated.
    
    T = 0.1
    T_min = 0.001
    alpha = 0.99
    
    current_side = best_side
    current_inner = best_inner
    current_center = best_center
    current_angle = best_angle
    current_cluster_rot = 0.0
    
    while time.time() < deadline and T > T_min:
        # Perturb
        d_cluster_rot = (random.random() - 0.5) * T * 10
        d_outer_rot = (random.random() - 0.5) * T * 10
        d_cx = (random.random() - 0.5) * T
        d_cy = (random.random() - 0.5) * T
        
        new_cluster_rot = current_cluster_rot + d_cluster_rot
        new_outer_rot = current_angle + d_outer_rot
        new_cx = current_center[0] + d_cx
        new_cy = current_center[1] + d_cy
        
        # Apply cluster rotation to inner positions
        cos_r, sin_r = math.cos(math.radians(new_cluster_rot)), math.sin(math.radians(new_cluster_rot))
        new_inner = []
        for x, y, a in current_inner:
            nx = x * cos_r - y * sin_r
            ny = x * sin_r + y * cos_r
            new_inner.append([nx, ny, a + new_cluster_rot])
            
        # Check non-overlap (should be preserved if rotation is global and lattice is rigid)
        # But floating point might cause issues. We assume rigid rotation preserves distances.
        
        new_side = _compute_outer_side(new_inner, (new_cx, new_cy), new_outer_rot)
        
        if new_side < current_side or random.random() < math.exp((current_side - new_side) / T):
            current_side = new_side
            current_inner = new_inner
            current_center = (new_cx, new_cy)
            current_angle = new_outer_rot
            current_cluster_rot = new_cluster_rot
            
            if current_side < best_side:
                best_side = current_side
                best_inner = [row[:] for row in current_inner]
                best_center = current_center
                best_angle = current_angle
                
        T *= alpha
        
    return best_inner, best_center, best_side, best_angle

import random
random.seed(42)
