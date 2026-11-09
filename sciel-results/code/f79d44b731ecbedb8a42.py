import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice: 1 center + 6 ring1 + 4 ring2
    positions = [(0.0, 0.0)]
    for i in range(6):
        angle = i * np.pi / 3
        positions.append((2.0 * np.cos(angle), 2.0 * np.sin(angle)))
    # 4 from ring 2, chosen for compactness
    positions.append((4.0, 0.0))
    positions.append((2.0, 2.0 * sqrt3))
    positions.append((-2.0, 2.0 * sqrt3))
    positions.append((-4.0, 0.0))
    
    assert len(positions) == N
    inner_angles = [0.0] * N
    
    def hex_vertices(cx, cy, rot_deg):
        rot = np.radians(rot_deg)
        return np.array([(cx + np.cos(rot + i * np.pi / 3), 
                          cy + np.sin(rot + i * np.pi / 3)) for i in range(6)])
    
    all_verts = np.vstack([hex_vertices(px, py, a) for (px, py), a in zip(positions, inner_angles)])
    
    def min_outer_S(ox, oy, phi, verts):
        dx = verts[:, 0] - ox
        dy = verts[:, 1] - oy
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        cos_n = np.cos(normal_angles)
        sin_n = np.sin(normal_angles)
        dots = dx[:, None] * cos_n[None, :] + dy[:, None] * sin_n[None, :]
        return np.max(np.abs(dots)) * 2.0 / sqrt3
    
    def objective(p):
        return min_outer_S(p[0], p[1], p[2], all_verts)
    
    best_S = objective([0.0, 0.0, 0.0])
    best_p = [0.0, 0.0, 0.0]
    
    for phi0 in [0.0, np.pi/6, np.pi/3, np.pi/4, np.pi/2, 2*np.pi/3]:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(objective, [0.0, 0.0, phi0], method='Nelder-Mead',
                          options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < best_S:
                best_S = res.fun
                best_p = res.x
        except Exception:
            pass
    
    ox, oy, phi = best_p
    S = best_S
    
    inner = [[px, py, a] for (px, py), a in zip(positions, inner_angles)]
    return inner, (ox, oy), S, float(np.degrees(phi))
# EVOLVE-BLOCK-END
