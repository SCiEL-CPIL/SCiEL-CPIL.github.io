import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0

# Precompute unit hexagon vertices (flat-top orientation)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))

def construct():
    t0 = time.time()
    
    # Unit hexagon vertices
    uverts = _UNIT_HEX_VERTS
    
    # Function to compute the minimum outer side length for a given configuration
    def compute_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = []
        for i in range(N):
            c = math.cos(angles_rad[i])
            s = math.sin(angles_rad[i])
            rx = uverts[:, 0] * c - uverts[:, 1] * s
            ry = uverts[:, 0] * s + uverts[:, 1] * c
            vx = rx + positions[i, 0]
            vy = ry + positions[i, 1]
            all_verts.append(np.column_stack((vx, vy)))
        
        all_verts = np.vstack(all_verts)
        center = np.mean(all_verts, axis=0)
        offsets = all_verts - center
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            # Normals for the 6 sides of the outer hexagon
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            # Side length s relates to max projection p by p = s * sqrt(3)/2
            side_vals = (2.0 / math.sqrt(3)) * max_projs
            return np.max(side_vals)
        
        # Coarse scan
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune
        try:
            from scipy.optimize import minimize_scalar
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    # Collision check: Separating Axis Theorem
    def hexagons_disjoint(h1, h2):
        # h1, h2 are (x, y, angle_rad)
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5: # Max distance between centers for touching is 2
            return True
            
        a1, a2 = h1[2], h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        v1x = h1[0] + uverts[:, 0] * c1 - uverts[:, 1] * s1
        v1y = h1[1] + uverts[:, 0] * s1 + uverts[:, 1] * c1
        v2x = h2[0] + uverts[:, 0] * c2 - uverts[:, 1] * s2
        v2y = h2[1] + uverts[:, 0] * s2 + uverts[:, 1] * c2
        
        # Edge normals
        # Edge i is from vert i to vert i+1
        # Normal is perpendicular to edge
        # For unit hex, edge length is 1.
        # We can just use the rotated edge vectors
        
        # Pre-compute edge vectors for unit hex
        # e_i = v_{i+1} - v_i
        # Normal n_i = (-e_y, e_x) normalized
        
        # Since uverts are fixed, we can precompute normals
        # But they need rotation.
        
        # Let's just compute the 6 normals for each hexagon
        # Normals are at angles a + 0, a + 60, ... ? No.
        # The normals of the sides of a regular hexagon are at angles a + 30 + k*60?
        # Let's check. Vertices at 0, 60, 120...
        # Edge 0: 0 to 60. Midpoint angle 30. Normal is 30+90=120?
        # Vector (1,0) to (0.5, 0.866). Vec = (-0.5, 0.866). Normal = (-0.866, -0.5). Angle 210?
        # Actually, for a regular polygon, the outward normals are aligned with the bisectors of the vertices.
        # Vertex angles: a, a+60, ...
        # Normal angles: a+30, a+90, ...
        
        n1_angles = a1 + np.pi/6 + np.arange(6) * np.pi/3
        n2_angles = a2 + np.pi/6 + np.arange(6) * np.pi/3
        
        n1x = np.cos(n1_angles)
        n1y = np.sin(n1_angles)
        n2x = np.cos(n2_angles)
        n2y = np.sin(n2_angles)
        
        for i in range(6):
            nx, ny = n1x[i], n1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = n2x[i], n2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def is_valid(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
                if not hexagons_disjoint(h1, h2):
                    return False
        return True

    # Objective function for L-BFGS-B
    # Parameters: x0, y0, a0, x1, y1, a1, ...
    # Returns: side_length + penalty
    def objective(params):
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
            
        # Check collisions
        angles_rad = np.radians(ang)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
                h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
                if not hexagons_disjoint(h1, h2):
                    # Approximate penalty based on distance
                    # Ideally we'd compute the actual overlap distance, but that's hard.
                    # A simple penalty: 1000 * (1 - dist/2) if dist < 2?
                    # Or just a large constant.
                    penalty += 100.0
                    # Actually, let's try to be a bit smarter for gradients.
                    # If we use a soft penalty, it might help.
                    # But SAT is boolean.
                    # Let's stick to a large penalty to push them apart.
        
        if penalty > 0:
            return penalty # + side_length? No, just penalty to force feasibility first.
        
        side, _, _, _ = compute_outer_side(pos, ang)
        return side

    # Honeycomb initial configuration
    sqrt3 = math.sqrt(3)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0

    best_side = float('inf')
    best_params = np.zeros(3*N)
    for i in range(N):
        best_params[3*i] = base_positions[i, 0]
        best_params[3*i+1] = base_positions[i, 1]
        best_params[3*i+2] = base_angles[i]
        
    best_side, best_phi, best_cx, best_cy = compute_outer_side(base_positions, base_angles)

    # Strategy: Multiple L-BFGS-B runs from different initializations
    # 1. Honeycomb
    # 2. Random perturbations of honeycomb
    # 3. Random configurations
    
    time_budget = DEADLINE * 0.8
    time_per_run = time_budget / 10.0
    
    runs = []
    # Run 1: Base
    runs.append(best_params.copy())
    
    # Run 2-6: Perturbed honeycomb
    for k in range(5):
        p = best_params.copy()
        for i in range(N):
            p[3*i] += random.gauss(0, 0.5)
            p[3*i+1] += random.gauss(0, 0.5)
            p[3*i+2] += random.gauss(0, 10)
        runs.append(p)
        
    # Run 7-10: Random
    for k in range(4):
        p = np.zeros(3*N)
        for i in range(N):
            p[3*i] = random.uniform(-5, 5)
            p[3*i+1] = random.uniform(-5, 5)
            p[3*i+2] = random.uniform(0, 60)
        runs.append(p)

    for init_p in runs:
        if time.time() - t0 > DEADLINE - 5:
            break
            
        try:
            res = minimize(objective, init_p, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-6})
            
            if res.fun < best_side:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                
                if is_valid(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_outer_side(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_params = res.x.copy()
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
        except Exception:
            pass

    # Final Polish with L-BFGS-B on the best solution
    if time.time() - t0 < DEADLINE - 2:
        try:
            res = minimize(objective, best_params, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-8, 'gtol': 1e-8})
            if res.fun < best_side:
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                
                if is_valid(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_outer_side(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_params = res.x.copy()
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
        except Exception:
            pass

    # Extract best solution
    best_positions = np.zeros((N, 2))
    best_angles = np.zeros(N)
    for i in range(N):
        best_positions[i, 0] = best_params[3*i]
        best_positions[i, 1] = best_params[3*i+1]
        best_angles[i] = best_params[3*i+2]
        
    # Final verification
    if not is_valid(best_positions, best_angles):
        # Fallback to honeycomb
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_phi, best_cx, best_cy = compute_outer_side(best_positions, best_angles)
    else:
        # Recompute side to be sure
        best_side, best_phi, best_cx, best_cy = compute_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
