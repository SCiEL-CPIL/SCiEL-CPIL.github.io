"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    sq3 = math.sqrt(3)
    u = (sq3, 0.0)
    v = (sq3/2.0, 1.5)
    
    centers = []
    # Center
    centers.append((0.0, 0.0))
    
    # Ring 1
    neighbors = [
        (u[0], u[1]),
        (v[0], v[1]),
        (v[0]-u[0], v[1]-u[1]),
        (-u[0], -u[1]),
        (-v[0], -v[1]),
        (-v[0]+u[0], -v[1]+u[1])
    ]
    centers.extend(neighbors)
    
    # Ring 2 (4 specific ones)
    centers.append((2*u[0], 2*u[1]))
    centers.append((2*v[0], 2*v[1]))
    centers.append((-2*u[0], -2*u[1]))
    centers.append((-2*v[0], -2*v[1]))
    
    all_vertices = []
    for cx, cy in centers:
        verts = [
            (cx + 1.0, cy),
            (cx + 0.5, cy + sq3/2.0),
            (cx - 0.5, cy + sq3/2.0),
            (cx - 1.0, cy),
            (cx - 0.5, cy - sq3/2.0),
            (cx + 0.5, cy - sq3/2.0)
        ]
        all_vertices.extend(verts)
    
    best_S = float('inf')
    best_rotation = 0.0
    
    # Check 60 rotations
    for deg in range(0, 60, 1):
        if time.time() - start_time > 140:
            break
        phi = math.radians(deg)
        cos_p = math.cos(-phi)
        sin_p = math.sin(-phi)
        
        req_S = 0.0
        for x, y in all_vertices:
            xr = x * cos_p - y * sin_p
            yr = x * sin_p + y * cos_p
            
            val1 = abs(xr)
            val2 = abs(xr + yr/sq3)
            val3 = abs(-xr + yr/sq3)
            
            curr_max = max(val1, val2, val3)
            if curr_max > req_S:
                req_S = curr_max
                
        if req_S < best_S:
            best_S = req_S
            best_rotation = deg
            
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    center = (0.0, 0.0)
    side = best_S
    angle = float(best_rotation)
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
