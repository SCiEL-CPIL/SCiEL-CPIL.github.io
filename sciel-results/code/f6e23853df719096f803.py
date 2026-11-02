"""
Structural change: use 6-fold rotational symmetry to reduce the search space.
12 hexagons = 2 orbits of 6 under 60° rotation.
Optimize 6 parameters (2 orbits × (x, y, angle)) with a smooth objective.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # Safety margin before 150s wall-clock kill
S3 = math.sqrt(3)
PI = math.pi

# Base hexagon vertices (unit side, flat-top)
BASE_VERTS = np.array([
    [1.0, 0.0],
    [0.5, S3/2],
    [-0.5, S3/2],
    [-1.0, 0.0],
    [-0.5, -S3/2],
    [0.5, -S3/2]
])

def rotate_point(x, y, angle_rad):
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    return x * ca - y * sa, x * sa + y * ca

def get_orbit(cx, cy, angle_deg, orbit_size=6):
    """Generate orbit of hexagons under 60° rotation about origin."""
    hexes = []
    a0 = math.radians(angle_deg)
    for k in range(orbit_size):
        rot_angle = math.radians(60 * k)
        # Rotate center
        rx, ry = rotate_point(cx, cy, rot_angle)
        # Total rotation angle
        total_angle = angle_deg + 60 * k
        hexes.append([rx, ry, total_angle])
    return hexes

def get_all_hexes(params):
    """
    params: [x1, y1, a1, x2, y2, a2]
    Returns list of 12 hexagons (x, y, angle_deg).
    """
    hexes = []
    hexes.extend(get_orbit(params[0], params[1], params[2]))
    hexes.extend(get_orbit(params[3], params[4], params[5]))
    return hexes

def hexagon_vertices(cx, cy, angle_deg):
    """Get 6 vertices of a hexagon."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        bx, by = BASE_VERTS[i]
        rx, ry = rotate_point(bx, by, a)
        verts.append([cx + rx, cy + ry])
    return np.array(verts)

def sat_overlap(v1, v2):
    """Check if two hexagons overlap using SAT. Returns True if they overlap."""
    # Quick rejection: bounding boxes
    min1 = v1.min(axis=0)
    max1 = v1.max(axis=0)
    min2 = v2.min(axis=0)
    max2 = v2.max(axis=0)
    if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
        return False
    
    # SAT with 12 normals (6 from each hexagon)
    for verts in [v1, v2]:
        for i in range(6):
            e = verts[(i+1)%6] - verts[i]
            n = np.array([e[1], -e[0]])
            norm = np.linalg.norm(n)
            if norm < 1e-10:
                continue
            n /= norm
            proj1 = v1 @ n
            proj2 = v2 @ n
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return False
    return True

def overlap_penalty(hexes, beta=5.0):
    """Compute smooth overlap penalty for all pairs."""
    penalty = 0.0
    n = len(hexes)
    for i in range(n):
        for j in range(i+1, n):
            dx = hexes[i][0] - hexes[j][0]
            dy = hexes[i][1] - hexes[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            # Minimum distance for disjoint hexagons is sqrt(3) ≈ 1.732
            if dist < S3:
                # Check actual overlap with SAT
                v1 = hexagon_vertices(hexes[i][0], hexes[i][1], hexes[i][2])
                v2 = hexagon_vertices(hexes[j][0], hexes[j][1], hexes[j][2])
                if sat_overlap(v1, v2):
                    # Estimate penetration depth
                    pen = S3 - dist
                    if pen > 0:
                        penalty += pen * pen * beta
    return penalty

def compute_min_side(hexes):
    """Compute minimum side length of containing regular hexagon."""
    # Get all vertices
    all_verts = []
    for h in hexes:
        verts = hexagon_vertices(h[0], h[1], h[2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Center: mean of all vertices
    center = np.mean(all_verts, axis=0)
    offsets = all_verts - center
    
    # Find optimal rotation
    best_side = float('inf')
    best_phi = 0.0
    
    # Coarse scan
    for phi_deg in range(0, 60, 1):
        phi = math.radians(phi_deg)
        max_extent = 0.0
        for k in range(6):
            normal_angle = phi + PI/6 + k * PI/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            proj = offsets[:, 0] * nx + offsets[:, 1] * ny
            max_proj = np.max(np.abs(proj))
            side_k = (2.0 / S3) * max_proj
            if side_k > max_extent:
                max_extent = side_k
        if max_extent < best_side:
            best_side = max_extent
            best_phi = phi_deg
    
    # Fine scan
    for phi_deg in np.arange(best_phi - 1, best_phi + 1, 0.05):
        if phi_deg < 0 or phi_deg >= 60:
            continue
        phi = math.radians(phi_deg)
        max_extent = 0.0
        for k in range(6):
            normal_angle = phi + PI/6 + k * PI/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            proj = offsets[:, 0] * nx + offsets[:, 1] * ny
            max_proj = np.max(np.abs(proj))
            side_k = (2.0 / S3) * max_proj
            if side_k > max_extent:
                max_extent = side_k
        if max_extent < best_side:
            best_side = max_extent
            best_phi = phi_deg
    
    return best_side, best_phi, center

def objective(params, beta=5.0):
    """Objective: side length + overlap penalty."""
    hexes = get_all_hexes(params)
    side, phi, center = compute_min_side(hexes)
    pen = overlap_penalty(hexes, beta)
    return side + pen

def construct():
    t0 = time.time()
    
    # Initial guess: two orbits arranged symmetrically
    # Orbit 1: closer to center
    # Orbit 2: further from center
    best_params = np.array([
        1.0, 0.0, 0.0,   # Orbit 1: x, y, angle
        2.5, 0.0, 0.0    # Orbit 2: x, y, angle
    ])
    
    best_side = float('inf')
    best_phi = 0.0
    best_center = np.array([0.0, 0.0])
    
    # Multiple random restarts
    n_restarts = 15
    for r in range(n_restarts):
        if time.time() - t0 > DEADLINE - 10:
            break
        
        if r == 0:
            params = best_params.copy()
        else:
            # Random initialization
            r1 = random.uniform(0.5, 2.0)
            a1 = random.uniform(0, 60)
            r2 = random.uniform(2.0, 3.5)
            a2 = random.uniform(0, 60)
            # Convert polar to cartesian for orbit centers
            params = np.array([
                r1 * math.cos(math.radians(a1)), r1 * math.sin(math.radians(a1)), random.uniform(0, 60),
                r2 * math.cos(math.radians(a2 + 30)), r2 * math.sin(math.radians(a2 + 30)), random.uniform(0, 60)
            ])
        
        # Optimize with Nelder-Mead
        try:
            res = minimize(objective, params, method='Nelder-Mead',
                          options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500, 'maxfev': 1000})
            
            if res.fun < best_side + 0.1:
                # Verify validity
                hexes = get_all_hexes(res.x)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        v1 = hexagon_vertices(hexes[i][0], hexes[i][1], hexes[i][2])
                        v2 = hexagon_vertices(hexes[j][0], hexes[j][1], hexes[j][2])
                        if sat_overlap(v1, v2):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    side, phi, center = compute_min_side(hexes)
                    if side < best_side:
                        best_side = side
                        best_phi = phi
                        best_center = center
                        best_params = res.x.copy()
        except:
            pass
    
    # Final polish if time allows
    if time.time() - t0 < DEADLINE - 5:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 300, 'maxfev': 500})
            hexes = get_all_hexes(res.x)
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    v1 = hexagon_vertices(hexes[i][0], hexes[i][1], hexes[i][2])
                    v2 = hexagon_vertices(hexes[j][0], hexes[j][1], hexes[j][2])
                    if sat_overlap(v1, v2):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                side, phi, center = compute_min_side(hexes)
                if side < best_side:
                    best_side = side
                    best_phi = phi
                    best_center = center
        except:
            pass
    
    # Get final hexagons
    final_hexes = get_all_hexes(best_params)
    
    return final_hexes, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_phi)

import random
