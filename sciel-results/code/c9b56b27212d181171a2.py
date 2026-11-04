"""Unit regular hexagons inside a regular hexagon n=12."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0  # safety margin before 150s kill

def signed_dist_to_hex(pt, cx, cy, side, angle_deg):
    """
    Signed distance from point to hexagon boundary.
    Positive if outside, negative if inside.
    Hexagon has circumradius = side, centered at (cx, cy), rotated by angle_deg.
    """
    a = math.radians(angle_deg)
    dx, dy = pt[0] - cx, pt[1] - cy
    # Rotate into local frame
    ca, sa = math.cos(-a), math.sin(-a)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    s3 = side * math.sqrt(3.0) / 2.0
    # Six half-plane constraints: dot((lx,ly), normal_k) <= s3
    # Outward normals at angles a + pi/6 + k*pi/3
    max_v = 0.0
    for k in range(6):
        nk = a + math.pi / 6.0 + k * math.pi / 3.0
        val = lx * math.cos(nk) + ly * math.sin(nk) - s3
        if val > max_v:
            max_v = val
    return max_v

def compute_objective(params, outer_center, outer_side, outer_angle):
    """
    Objective: outer_side + penalty for constraint violations.
    params: [x0,y0,angle0, x1,y1,angle1, ..., x11,y11,angle11]
    """
    penalty = 0.0
    
    # Unpack
    xs = params[0::3]
    ys = params[1::3]
    angles = params[2::3]
    
    # Precompute vertices for each inner hexagon
    verts_list = []
    for i in range(N):
        verts = []
        a = math.radians(angles[i])
        for k in range(6):
            theta = a + k * math.pi / 3.0
            verts.append((xs[i] + math.cos(theta), ys[i] + math.sin(theta)))
        verts_list.append(verts)
    
    # Penalty 1: inner hexagons must fit inside outer hexagon
    for i in range(N):
        for v in verts_list[i]:
            sd = signed_dist_to_hex(v, outer_center[0], outer_center[1], outer_side, outer_angle)
            if sd > 0:
                penalty += sd * sd
    
    # Penalty 2: inner hexagons must not overlap
    for i in range(N):
        for j in range(i + 1, N):
            # Check vertices of i against hex j
            for v in verts_list[i]:
                sd = signed_dist_to_hex(v, xs[j], ys[j], 1.0, angles[j])
                if sd > 0:
                    penalty += sd * sd
            # Check vertices of j against hex i
            for v in verts_list[j]:
                sd = signed_dist_to_hex(v, xs[i], ys[i], 1.0, angles[i])
                if sd > 0:
                    penalty += sd * sd
    
    return outer_side + penalty

def hex_lattice_init(spacing=1.8):
    """Hexagonal lattice initialization."""
    positions = []
    dx = spacing
    dy = spacing * math.sqrt(3) / 2
    for row in range(-3, 4):
        for col in range(-3, 4):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append([x, y, 0.0])
    return positions[:N]

def grid_init(spacing=1.8):
    """Square grid initialization."""
    positions = []
    for row in range(-3, 4):
        for col in range(-3, 4):
            positions.append([col * spacing, row * spacing, 0.0])
    return positions[:N]

def optimize(params0, outer_center, outer_side, outer_angle, max_time=25.0):
    """Multi-stage optimization: L-BFGS-B then Nelder-Mead."""
    t0 = time.time()
    
    def obj(p):
        return compute_objective(p, outer_center, outer_side, outer_angle)
    
    bounds = [(-25.0, 25.0), (-25.0, 25.0), (0.0, 360.0)] * N
    
    # Stage 1: L-BFGS-B
    res1 = minimize(obj, params0, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
    best_p = res1.x
    best_v = res1.fun
    
    if time.time() - t0 > max_time:
        return best_p, best_v
    
    # Stage 2: Nelder-Mead polish
    res2 = minimize(obj, best_p, method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
    if res2.fun < best_v:
        best_p = res2.x
        best_v = res2.fun
    
    return best_p, best_v

def construct():
    t_start = time.time()
    
    best_score = float('inf')
    best_params = None
    best_outer = None
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Try different outer side lengths
    side_candidates = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
    
    for outer_side in side_candidates:
        if time.time() - t_start > TIME_LIMIT - 15:
            break
        
        inradius = outer_side * math.sqrt(3) / 2
        
        # Try different initializations
        for init_func in [hex_lattice_init, grid_init]:
            if time.time() - t_start > TIME_LIMIT - 15:
                break
            
            positions = init_func(spacing=1.8)
            params0 = np.array([p[0] for p in positions] + [p[1] for p in positions] + [p[2] for p in positions])
            
            # Scale to fit within outer hexagon
            max_coord = max(abs(p[0]) for p in positions) if positions else 1.0
            if max_coord > 0:
                scale = (inradius * 0.7) / max_coord
                for i in range(N):
                    params0[3*i] *= scale
                    params0[3*i+1] *= scale
            
            params, val = optimize(params0, outer_center, outer_side, outer_angle, max_time=20.0)
            
            # Check validity: penalty should be ~0
            if val < outer_side + 1e-4:
                if outer_side < best_score:
                    best_score = outer_side
                    best_params = params
                    best_outer = (outer_center, outer_side, outer_angle)
            
            if time.time() - t_start > TIME_LIMIT - 10:
                break
    
    # If no valid packing found, use the best attempt with a larger side
    if best_params is None:
        outer_side = 8.0
        positions = hex_lattice_init(spacing=1.8)
        params0 = np.array([p[0] for p in positions] + [p[1] for p in positions] + [p[2] for p in positions])
        inradius = outer_side * math.sqrt(3) / 2
        max_coord = max(abs(p[0]) for p in positions) if positions else 1.0
        if max_coord > 0:
            scale = (inradius * 0.7) / max_coord
            for i in range(N):
                params0[3*i] *= scale
                params0[3*i+1] *= scale
        params, val = optimize(params0, outer_center, outer_side, outer_angle, max_time=20.0)
        best_score = outer_side
        best_params = params
        best_outer = (outer_center, outer_side, outer_angle)
    
    # Extract inner positions
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    return inner, best_outer[0], best_outer[1], best_outer[2]
