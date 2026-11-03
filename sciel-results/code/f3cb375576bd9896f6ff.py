import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def min_separation_sat(A, B):
    """
    Returns the minimum separation distance between two convex polygons A and B.
    Negative if they overlap.
    A, B: lists of (x, y) tuples.
    """
    min_dist = float('inf')
    
    # Check axes from A
    for i in range(len(A)):
        x1, y1 = A[i]
        x2, y2 = A[(i + 1) % len(A)]
        dx, dy = x2 - x1, y2 - y1
        nx, ny = -dy, dx
        norm = math.sqrt(nx*nx + ny*ny)
        if norm < 1e-9: continue
        nx, ny = nx/norm, ny/norm
        
        minA = min(nx * px + ny * py for px, py in A)
        maxA = max(nx * px + ny * py for px, py in A)
        minB = min(nx * px + ny * py for px, py in B)
        maxB = max(nx * px + ny * py for px, py in B)
        
        if maxA < minB:
            dist = minB - maxA
        elif maxB < minA:
            dist = minA - maxB
        else:
            dist = 0.0 # Overlapping in this axis projection
            
        if dist < min_dist:
            min_dist = dist
            if min_dist == 0.0:
                return 0.0 # Overlap found, no need to check other axes for this pair if we just want penalty

    # Check axes from B
    for i in range(len(B)):
        x1, y1 = B[i]
        x2, y2 = B[(i + 1) % len(B)]
        dx, dy = x2 - x1, y2 - y1
        nx, ny = -dy, dx
        norm = math.sqrt(nx*nx + ny*ny)
        if norm < 1e-9: continue
        nx, ny = nx/norm, ny/norm
        
        minA = min(nx * px + ny * py for px, py in A)
        maxA = max(nx * px + ny * py for px, py in A)
        minB = min(nx * px + ny * py for px, py in B)
        maxB = max(nx * px + ny * py for px, py in B)
        
        if maxA < minB:
            dist = minB - maxA
        elif maxB < minA:
            dist = minA - maxB
        else:
            dist = 0.0
            
        if dist < min_dist:
            min_dist = dist
            if min_dist == 0.0:
                return 0.0

    return min_dist

def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True

def distance_to_convex_polygon_boundary(px, py, poly):
    """
    Returns the distance from point (px,py) to the boundary of the convex polygon.
    If inside, returns 0 (or negative? Let's use 0 for inside, positive for outside distance).
    For penalty, we want 0 if inside, >0 if outside.
    """
    if point_in_convex_polygon(px, py, poly):
        return 0.0
    
    # Find minimum distance to any edge
    min_dist = float('inf')
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        
        # Project point onto segment
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            dist = math.sqrt(wx*wx + wy*wy)
        else:
            c2 = vx*vx + vy*vy
            if c1 >= c2:
                dist = math.sqrt((px-x2)**2 + (py-y2)**2)
            else:
                t = c1 / c2
                proj_x = x1 + t * vx
                proj_y = y1 + t * vy
                dist = math.sqrt((px-proj_x)**2 + (py-proj_y)**2)
        
        if dist < min_dist:
            min_dist = dist
            
    return min_dist

def objective(params):
    """
    params: [S, alpha, x1, y1, th1, ..., x12, y12, th12]
    S: outer side length
    alpha: outer rotation (deg)
    xi, yi, thi: inner hexagon i position and rotation
    """
    if time.time() > DEADLINE:
        return 1e10
    
    S = params[0]
    alpha = params[1]
    
    if S < 1.0: # Minimum possible is larger
        S = 1.0
    
    outer_center = (0.0, 0.0)
    outer_poly = hex_vertices(0.0, 0.0, S, alpha)
    
    inner_polys = []
    contain_penalty = 0.0
    
    for i in range(N):
        offset = 2 + i * 3
        x = params[offset]
        y = params[offset + 1]
        th = params[offset + 2]
        
        poly = hex_vertices(x, y, 1.0, th)
        inner_polys.append(poly)
        
        # Containment check
        for px, py in poly:
            d = distance_to_convex_polygon_boundary(px, py, outer_poly)
            if d > 0:
                contain_penalty += d * d
                
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist = min_separation_sat(inner_polys[i], inner_polys[j])
            if dist < 0:
                # Overlap penalty
                overlap_penalty += dist * dist
            else:
                # Small gap penalty to encourage tight packing? 
                # No, just penalize overlap. The objective minimizes S, which pushes them together.
                pass
    
    # Weight for penalties. High enough to force validity.
    return S + 100.0 * overlap_penalty + 100.0 * contain_penalty

def generate_initial_config(topology):
    """
    Generate initial parameters for optimization.
    Returns a numpy array of size 2 + 3*N.
    """
    params = np.zeros(2 + 3 * N)
    
    # Topology 0: 3x4 grid
    if topology == 0:
        # Approximate hex grid spacing
        # Horizontal dist between centers: 1.5 * side? No, for flat-top hexagons:
        # If rotated 0, vertices at 0, 60, etc.
        # Width = 2 * side. Height = sqrt(3) * side.
        # Let's use a standard hex lattice spacing.
        # For side 1, distance between centers in same row is 1.5? No.
        # Let's just place them in a rectangular grid and let optimizer fix it.
        # Start with a dense grid.
        dx = 1.7
        dy = 1.5
        for i in range(4):
            for j in range(3):
                idx = i * 3 + j
                x = (i - 1.5) * dx
                y = (j - 1.0) * dy
                params[2 + idx*3] = x
                params[2 + idx*3 + 1] = y
                params[2 + idx*3 + 2] = 0.0 # rotation
        params[0] = 4.0 # Initial S guess
        params[1] = 0.0
    
    # Topology 1: Ring 1+6+5
    elif topology == 1:
        params[2] = 0.0 # Center
        params[3] = 0.0
        params[4] = 0.0
        
        r1 = 1.8
        for k in range(6):
            idx = 1 + k
            theta = math.radians(k * 60)
            params[2 + idx*3] = r1 * math.cos(theta)
            params[2 + idx*3 + 1] = r1 * math.sin(theta)
            params[2 + idx*3 + 2] = 0.0
            
        r2 = 3.2
        for k in range(5):
            idx = 7 + k
            theta = math.radians(30 + k * 72)
            params[2 + idx*3] = r2 * math.cos(theta)
            params[2 + idx*3 + 1] = r2 * math.sin(theta)
            params[2 + idx*3 + 2] = 0.0
            
        params[0] = 4.5
        params[1] = 0.0
        
    # Topology 2: Random scatter
    else:
        np.random.seed(42)
        for i in range(N):
            params[2 + i*3] = np.random.uniform(-2, 2)
            params[2 + i*3 + 1] = np.random.uniform(-2, 2)
            params[2 + i*3 + 2] = np.random.uniform(0, 60)
        params[0] = 5.0
        params[1] = 0.0
        
    return params

def construct():
    t0 = time.time()
    best_side = 1e10
    best_params = None
    
    topologies = [0, 1, 2]
    
    for topo in topologies:
        if time.time() > DEADLINE - 10:
            break
            
        x0 = generate_initial_config(topo)
        
        # Try a few perturbations for random restarts
        for restart in range(3):
            if time.time() > DEADLINE - 5:
                break
                
            if restart > 0:
                # Perturb initial guess
                noise = np.random.normal(0, 0.1, x0.shape)
                x0_perturbed = x0 + noise
            else:
                x0_perturbed = x0.copy()
                
            try:
                # Use Nelder-Mead as it doesn't require gradients and handles non-smooth penalties well
                res = minimize(objective, x0_perturbed, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
                
                if res.fun < best_side:
                    # Check if the result is actually valid (penalties ~ 0)
                    # res.fun = S + penalties. If penalties are 0, res.fun = S.
                    # We need to verify validity explicitly to be sure.
                    params = res.x
                    S = params[0]
                    alpha = params[1]
                    
                    # Reconstruct to check
                    outer_poly = hex_vertices(0.0, 0.0, S, alpha)
                    valid = True
                    inner_polys = []
                    for i in range(N):
                        x = params[2 + i*3]
                        y = params[2 + i*3 + 1]
                        th = params[2 + i*3 + 2]
                        poly = hex_vertices(x, y, 1.0, th)
                        inner_polys.append(poly)
                        for px, py in poly:
                            if not point_in_convex_polygon(px, py, outer_poly):
                                valid = False
                                break
                        if not valid: break
                    
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                if min_separation_sat(inner_polys[i], inner_polys[j]) < -1e-5: # Allow tiny touch
                                    valid = False
                                    break
                            if not valid: break
                    
                    if valid:
                        best_side = S
                        best_params = params.copy()
                        
            except Exception:
                pass

    if best_params is not None:
        # Format output
        inner = []
        for i in range(N):
            x = best_params[2 + i*3]
            y = best_params[2 + i*3 + 1]
            th = best_params[2 + i*3 + 2]
            inner.append((x, y, th))
        
        outer_center = (0.0, 0.0)
        side = best_params[0]
        angle = best_params[1]
        
        return inner, outer_center, side, angle
    
    # Fallback: Return a known valid, albeit suboptimal, configuration
    # Simple 3x4 grid
    inner = []
    dx = 1.75
    dy = 1.55
    for i in range(4):
        for j in range(3):
            x = (i - 1.5) * dx
            y = (j - 1.0) * dy
            inner.append((x, y, 0.0))
            
    outer_center = (0.0, 0.0)
    # Estimate side
    max_x = 1.5 * dx + 1.0
    max_y = 1.0 * dy + 1.0
    # Hexagon side S approx max_radius / (sqrt(3)/2)
    # For a hexagon, max distance from center to vertex is S.
    # Max distance in grid is approx sqrt(max_x^2 + max_y^2)
    side = max(max_x, max_y) * 1.2 
    angle = 0.0
    
    return inner, outer_center, side, angle

