import math
import time
import random
import numpy as np
from scipy.optimize import minimize_scalar

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    _ux = _UNIT_HEX_VERTS[:, 0]
    _uy = _UNIT_HEX_VERTS[:, 1]
    _enx = _EDGE_NORMALS[:, 0]
    _eny = _EDGE_NORMALS[:, 1]
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def hexagons_disjoint_fast(h1, h2):
        """
        h1, h2: (x, y, angle_rad)
        Returns True if disjoint.
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        # Quick bounding box / distance check
        # Min distance between centers for unit hexagons to be disjoint is 2*sqrt(3)/2 * 2? No.
        # Distance between centers of touching hexes is sqrt(3).
        # So if dist > 1.0 (circumradius sum is 2, but inradius is sqrt(3)/2 ~ 0.866), 
        # actually min center distance for disjoint is 2 * inradius = sqrt(3) ~ 1.732.
        # If dist_sq > 3.5, they are likely disjoint.
        if dx*dx + dy*dy > 3.5:
            return True
            
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        
        # Separating Axis Theorem
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def compute_min_outer_side(positions, angles_deg):
        """
        Computes the minimum side length of the outer hexagon containing all inner hexagons.
        Also optimizes the outer hexagon rotation angle.
        """
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        
        # Center the configuration to simplify the outer hexagon calculation
        # The outer hexagon is centered at the centroid of the inner hexagons' vertices?
        # Or we can just find the min enclosing hexagon.
        # For a regular hexagon, the center that minimizes the side length is the center of the bounding box?
        # Actually, for a regular hexagon, the optimal center is the one that minimizes the max projection.
        # Let's fix the center to the centroid of the inner hexagons' centers for simplicity,
        # or better, calculate the centroid of all vertices.
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            # Normals of the outer hexagon
            # A regular hexagon with side S has inradius S*sqrt(3)/2.
            # The distance from center to edge is S*sqrt(3)/2.
            # So if max projection is P, then S*sqrt(3)/2 >= P => S >= 2*P/sqrt(3).
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / math.sqrt(3)) * max_projs
            return np.max(side_vals)
        
        # Scan for best phi
        best_side = float('inf')
        best_phi = 0.0
        # Coarse scan
        for phi_deg in range(0, 60, 2):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
                
        # Fine scan
        lo = max(0.0, best_phi - 2.0)
        hi = min(60.0, best_phi + 2.0)
        try:
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded', options={'xatol': 1e-6})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    def is_valid(positions, angles_rad):
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast((positions[i,0], positions[i,1], angles_rad[i]), 
                                              (positions[j,0], positions[j,1], angles_rad[j])):
                    return False
        return True

    def relax(positions, angles_rad, steps=1000, dt=0.01, damping=0.95):
        """
        Physics-based relaxation.
        We treat each hexagon as a particle with position and velocity.
        Forces:
        1. Repulsion between overlapping hexagons.
        2. Weak attraction to the center to keep them compact.
        """
        pos = positions.copy()
        vel = np.zeros_like(pos)
        
        # We keep angles fixed during relaxation to simplify the force calculation?
        # Or we can also relax angles? Relaxing angles is harder.
        # Let's fix angles in the relaxation phase and only optimize positions.
        # The initial angles are set in the initialization step.
        
        for step in range(steps):
            forces = np.zeros_like(pos)
            
            # Repulsion
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i,0] - pos[j,0]
                    dy = pos[i,1] - pos[j,1]
                    dist_sq = dx*dx + dy*dy
                    dist = math.sqrt(dist_sq) if dist_sq > 1e-6 else 1e-3
                    
                    # Check if they are overlapping or close
                    # Minimum distance between centers for disjoint hexes is sqrt(3) ~ 1.732
                    # Let's use a soft repulsion if dist < 2.0
                    if dist < 2.0:
                        # Force magnitude increases as distance decreases
                        # F = k * (r_min - dist)
                        r_min = 1.75 # Slightly larger than sqrt(3) to ensure separation
                        if dist < r_min:
                            f_mag = 100.0 * (r_min - dist)
                            if dist > 1e-6:
                                fx = f_mag * dx / dist
                                fy = f_mag * dy / dist
                                forces[i,0] += fx
                                forces[i,1] += fy
                                forces[j,0] -= fx
                                forces[j,1] -= fy
                        else:
                            # Weak repulsion to prevent them from getting too close
                            f_mag = 10.0 * (2.0 - dist)
                            if dist > 1e-6:
                                fx = f_mag * dx / dist
                                fy = f_mag * dy / dist
                                forces[i,0] += fx
                                forces[i,1] += fy
                                forces[j,0] -= fx
                                forces[j,1] -= fy

            # Centering force (weak) to keep them together
            for i in range(N):
                forces[i,0] -= 0.1 * pos[i,0]
                forces[i,1] -= 0.1 * pos[i,1]
                
            # Update velocities and positions
            vel = vel * damping + forces * dt
            pos = pos + vel * dt
            
        return pos

    def run_one_attempt():
        t_start = time.time()
        
        # Initialization: Random or Honeycomb-like
        # Let's try a mix of random and structured
        if random.random() < 0.5:
            # Honeycomb lattice perturbed
            sqrt3 = math.sqrt(3)
            a1 = np.array([sqrt3, 0.0])
            a2 = np.array([sqrt3/2.0, 1.5])
            lattice_coords = [
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, -1), (-1, 1), (2, -1), (-2, 1),
                (1, 1), (-1, -1), (2, 0),
            ]
            pos = np.zeros((N, 2))
            for idx, (i, j) in enumerate(lattice_coords):
                pos[idx] = i * a1 + j * a2
            pos += np.random.uniform(-0.2, 0.2, (N, 2))
            angles = np.zeros(N)
        else:
            # Random ring
            r = random.uniform(2.0, 3.0)
            pos = np.zeros((N, 2))
            angles = np.zeros(N)
            for i in range(N):
                theta = 2 * math.pi * i / N
                pos[i, 0] = r * math.cos(theta)
                pos[i, 1] = r * math.sin(theta)
                angles[i] = theta
            pos += np.random.uniform(-0.5, 0.5, (N, 2))
            angles += np.random.uniform(-0.5, 0.5, N)
            
        angles_rad = np.radians(angles)
        
        # Relaxation
        pos_relaxed = relax(pos, angles_rad, steps=2000, dt=0.02, damping=0.98)
        
        # Check validity
        if not is_valid(pos_relaxed, angles_rad):
            # If invalid, try a few more relaxation steps with stronger repulsion
            # Or just reject
            # Let's try to fix it by pushing them apart
            for _ in range(500):
                forces = np.zeros_like(pos_relaxed)
                for i in range(N):
                    for j in range(i+1, N):
                        dx = pos_relaxed[i,0] - pos_relaxed[j,0]
                        dy = pos_relaxed[i,1] - pos_relaxed[j,1]
                        dist_sq = dx*dx + dy*dy
                        dist = math.sqrt(dist_sq) if dist_sq > 1e-6 else 1e-3
                        if dist < 1.8:
                            f_mag = 500.0 * (1.8 - dist)
                            if dist > 1e-6:
                                fx = f_mag * dx / dist
                                fy = f_mag * dy / dist
                                forces[i,0] += fx
                                forces[i,1] += fy
                                forces[j,0] -= fx
                                forces[j,1] -= fy
                pos_relaxed += forces * 0.005
                
            if not is_valid(pos_relaxed, angles_rad):
                return None
                
        # Calculate side
        side, phi, cx, cy = compute_min_outer_side(pos_relaxed, angles)
        
        return side, pos_relaxed, angles, phi, cx, cy

    best_side = float('inf')
    best_result = None
    
    # Main loop: Multiple attempts
    attempts = 0
    while time.time() - t0 < DEADLINE - 5:
        attempts += 1
        result = run_one_attempt()
        if result:
            side, pos, ang, phi, cx, cy = result
            if side < best_side:
                best_side = side
                best_result = (pos, ang, phi, cx, cy)
                
    if best_result is None:
        # Fallback: Honeycomb
        sqrt3 = math.sqrt(3)
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
        side, phi, cx, cy = compute_min_outer_side(pos, ang)
        best_result = (pos, ang, phi, cx, cy)
        best_side = side
        
    pos, ang, phi, cx, cy = best_result
    
    inner = []
    for i in range(N):
        inner.append([float(pos[i, 0]), float(pos[i, 1]), float(ang[i])])
        
    center = (float(cx), float(cy))
    side = float(best_side)
    angle = float(phi)
    
    return inner, center, side, angle

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side}, Center: {center}, Angle: {angle}")
