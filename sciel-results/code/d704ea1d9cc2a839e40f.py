import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_verts(center, angle_deg):
    """Return vertices of a unit regular hexagon centered at center with rotation angle_deg."""
    angle = np.deg2rad(angle_deg)
    base_angles = np.arange(6) * np.pi / 3
    angles = base_angles + angle
    x = center[0] + np.cos(angles)
    y = center[1] + np.sin(angles)
    return np.column_stack((x, y))

def support_function(phi):
    """Support function of a unit hexagon (side 1, circumradius 1) in direction phi."""
    phi_mod = (phi + np.pi/6) % (np.pi/3) - np.pi/6
    return np.cos(phi_mod)

def min_dist_hex(v1, v2):
    """
    Approximate signed distance between two hexagons given their vertices.
    Returns negative if overlapping, positive if disjoint.
    """
    min_d = np.inf
    overlap = False
    
    # Helper to check point in polygon (hex)
    def point_in_hex(p, verts):
        signs = []
        for i in range(6):
            ex = verts[(i+1)%6, 0] - verts[i, 0]
            ey = verts[(i+1)%6, 1] - verts[i, 1]
            px = p[0] - verts[i, 0]
            py = p[1] - verts[i, 1]
            cross = ex * py - ey * px
            if cross > 1e-9:
                signs.append(1)
            elif cross < -1e-9:
                signs.append(-1)
            else:
                signs.append(0)
        non_zero = [s for s in signs if s != 0]
        if not non_zero:
            return True
        if all(s == non_zero[0] for s in non_zero):
            return True
        return False

    # Check vertices of v1 against v2
    for i in range(6):
        p = v1[i]
        if point_in_hex(p, v2):
            overlap = True
            d_to_edge = np.inf
            for j in range(6):
                p1 = v2[j]
                p2 = v2[(j+1)%6]
                vx = p2[0] - p1[0]
                vy = p2[1] - p1[1]
                wx = p[0] - p1[0]
                wy = p[1] - p1[1]
                c1 = wx * vx + wy * vy
                c2 = vx * vx + vy * vy
                if c2 < 1e-12:
                    t = 0
                else:
                    t = c1 / c2
                    t = max(0, min(1, t))
                proj_x = p1[0] + t * vx
                proj_y = p1[1] + t * vy
                d = np.sqrt((p[0]-proj_x)**2 + (p[1]-proj_y)**2)
                d_to_edge = min(d_to_edge, d)
            min_d = min(min_d, -d_to_edge)
            
    # Check vertices of v2 against v1
    for i in range(6):
        p = v2[i]
        if point_in_hex(p, v1):
            overlap = True
            d_to_edge = np.inf
            for j in range(6):
                p1 = v1[j]
                p2 = v1[(j+1)%6]
                vx = p2[0] - p1[0]
                vy = p2[1] - p1[1]
                wx = p[0] - p1[0]
                wy = p[1] - p1[1]
                c1 = wx * vx + wy * vy
                c2 = vx * vx + vy * vy
                if c2 < 1e-12:
                    t = 0
                else:
                    t = c1 / c2
                    t = max(0, min(1, t))
                proj_x = p1[0] + t * vx
                proj_y = p1[1] + t * vy
                d = np.sqrt((p[0]-proj_x)**2 + (p[1]-proj_y)**2)
                d_to_edge = min(d_to_edge, d)
            min_d = min(min_d, -d_to_edge)
            
    if overlap:
        return min_d
        
    # If not overlapping, find min distance between vertices and edges
    for i in range(6):
        p = v1[i]
        for j in range(6):
            p1 = v2[j]
            p2 = v2[(j+1)%6]
            vx = p2[0] - p1[0]
            vy = p2[1] - p1[1]
            wx = p[0] - p1[0]
            wy = p[1] - p1[1]
            c1 = wx * vx + wy * vy
            c2 = vx * vx + vy * vy
            if c2 < 1e-12:
                t = 0
            else:
                t = c1 / c2
                t = max(0, min(1, t))
            proj_x = p1[0] + t * vx
            proj_y = p1[1] + t * vy
            d = np.sqrt((p[0]-proj_x)**2 + (p[1]-proj_y)**2)
            min_d = min(min_d, d)
            
    return min_d

DEADLINE = 0

def objective(params):
    if time.time() > DEADLINE:
        return 1e6
        
    positions = []
    angles = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        theta = params[3*i+2]
        positions.append((x, y))
        angles.append(theta)
        
    normals = np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60))
    
    max_extent = 0.0
    for i in range(N):
        v = np.array(positions[i])
        theta = np.deg2rad(angles[i])
        for k in range(6):
            n = normals[k]
            phi = np.arctan2(n[1], n[0]) - theta
            h = support_function(phi)
            extent = np.dot(v, n) + h
            if extent > max_extent:
                max_extent = extent
                
    S_min = (2.0 / np.sqrt(3.0)) * max_extent
    
    penalty = 0.0
    verts_list = []
    for i in range(N):
        v = hex_verts(positions[i], angles[i])
        verts_list.append(v)
        
    for i in range(N):
        for j in range(i+1, N):
            d = min_dist_hex(verts_list[i], verts_list[j])
            if d < 0:
                penalty += (-d)**2
                
    return S_min + 100.0 * penalty

def construct():
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + 145.0
    
    best_score = np.inf
    best_params = None
    
    # Grid initialization
    # Honeycomb pattern
    # Row 0: 4 hexes
    # Row 1: 4 hexes
    # Row 2: 4 hexes
    # This might not fit in a hexagon, but it's a good start.
    
    # Let's use a random restart with a few iterations
    num_restarts = 4
    
    for r in range(num_restarts):
        if time.time() > DEADLINE:
            break
            
        params = np.zeros(3*N)
        for i in range(N):
            angle = np.random.uniform(0, 2*np.pi)
            rad = np.random.uniform(0, 2)
            params[3*i] = rad * np.cos(angle)
            params[3*i+1] = rad * np.sin(angle)
            params[3*i+2] = np.random.uniform(0, 60)
            
        res = minimize(objective, params, method='Nelder-Mead', 
                       options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        theta = best_params[3*i+2]
        inner.append([x, y, theta])
        
    positions = [(best_params[3*i], best_params[3*i+1]) for i in range(N)]
    angles = [best_params[3*i+2] for i in range(N)]
    
    normals = np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60))
    max_extent = 0.0
    for i in range(N):
        v = np.array(positions[i])
        theta = np.deg2rad(angles[i])
        for k in range(6):
            n = normals[k]
            phi = np.arctan2(n[1], n[0]) - theta
            h = support_function(phi)
            extent = np.dot(v, n) + h
            if extent > max_extent:
                max_extent = extent
                
    S = (2.0 / np.sqrt(3.0)) * max_extent
    
    return inner, (0.0, 0.0), S, 0.0
