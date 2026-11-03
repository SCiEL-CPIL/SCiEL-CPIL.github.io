import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    start_time = time.time()
    DEADLINE = 130.0
    
    sqrt3 = np.sqrt(3)
    
    # Unit hexagon vertices (flat-top, side 1)
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    # Hexagonal lattice arrangement for 11 unit hexagons
    positions = [(0.0, 0.0)]
    for i in range(6):
        angle = np.radians(60 * i)
        positions.append((sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)))
    for i in [0, 2, 4, 5]:
        angle = np.radians(60 * i)
        positions.append((2 * sqrt3 * np.cos(angle), 2 * sqrt3 * np.sin(angle)))
    
    positions = np.array(positions)
    n_pos = len(positions)
    
    def outer_side_length(phi, theta):
        """phi: global rotation of inner config, theta: outer hex rotation"""
        c, s = np.cos(phi), np.sin(phi)
        rot = np.array([[c, -s], [s, c]])
        
        # All vertices of all inner hexagons (all with rotation 0, then globally rotated)
        # Inner hex vertices at each position: pos + hex_verts
        all_verts = []
        for pos in positions:
            all_verts.append(pos + hex_verts)
        all_verts = np.vstack(all_verts)  # (66, 2)
        
        # Apply global rotation
        rotated_verts = all_verts @ rot.T
        
        # Outer hex normals
        angles = theta + np.radians(np.array([30, 90, 150, 210, 270, 330]))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
        
        supports = rotated_verts @ normals.T  # (66, 6)
        max_support = np.max(supports)
        
        return (2.0 / sqrt3) * max_support
    
    best_S = np.inf
    best_phi = 0.0
    best_theta = 0.0
    
    # Coarse grid
    for phi_deg in range(0, 180, 5):
        if time.time() - start_time > DEADLINE:
            break
        for theta_deg in range(0, 60, 5):
            S = outer_side_length(np.radians(phi_deg), np.radians(theta_deg))
            if S < best_S:
                best_S = S
                best_phi = np.radians(phi_deg)
                best_theta = np.radians(theta_deg)
    
    # Local refinement
    if time.time() - start_time < DEADLINE:
        res = minimize(
            lambda x: outer_side_length(x[0], x[1]),
            x0=[best_phi, best_theta],
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11}
        )
        if res.fun < best_S:
            best_S = res.fun
            best_phi = res.x[0]
            best_theta = res.x[1]
    
    # Apply global rotation to positions
    c, s = np.cos(best_phi), np.sin(best_phi)
    rot = np.array([[c, -s], [s, c]])
    final_positions = positions @ rot.T
    
    inner = [[float(p[0]), float(p[1]), 0.0] for p in final_positions]
    outer_center = (0.0, 0.0)
    outer_side = float(best_S)
    outer_angle = float(np.degrees(best_theta))
    
    return inner, outer_center, outer_side, outer_angle
