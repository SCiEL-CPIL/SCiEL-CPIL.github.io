"""Packing 12 unit hexagons in a hexagon - minimizing outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Honeycomb lattice initial guess
    u = np.array([np.sqrt(3), 0.0])
    v = np.array([np.sqrt(3)/2, 1.5])
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1),
    ]
    
    centers = np.array([i * u + j * v for i, j in axial_coords])
    cx0 = np.mean(centers[:, 0])
    cy0 = np.mean(centers[:, 1])
    
    # Compute initial S
    max_r = 0
    for c in centers:
        for k in range(6):
            theta = k * np.pi / 3
            px, py = c[0] + np.cos(theta), c[1] + np.sin(theta)
            r = np.sqrt((px - cx0)**2 + (py - cy0)**2)
            max_r = max(max_r, r)
    S0 = max_r / (np.sqrt(3) / 2) + 0.5
    
    def penalized_objective(x, penalty=500.0):
        # x = [x0,y0,th0, ..., x11,y11,th11, cx, cy, S, phi]
        inner = x[:3*N].reshape(N, 3)
        cx, cy, S, phi = x[3*N], x[3*N+1], x[3*N+2], x[3*N+3]
        
        inradius_outer = S * np.sqrt(3) / 2
        phi_rad = np.radians(phi)
        
        ith_rad = np.radians(inner[:, 2])
        k_angles = np.arange(6) * np.pi / 3
        v_angles = ith_rad[:, None] + k_angles[None, :]
        vx = inner[:, 0, None] + np.cos(v_angles)
        vy = inner[:, 1, None] + np.sin(v_angles)
        
        m_angles = phi_rad + np.arange(6) * np.pi / 3
        nx = np.cos(m_angles)
        ny = np.sin(m_angles)
        
        val = (vx[:, :, None] - cx) * nx[None, None, :] + \
              (vy[:, :, None] - cy) * ny[None, None, :] - inradius_outer
        
        violations = np.maximum(0, val)
        penalty_val = np.sum(violations ** 2)
        
        return S + penalty * penalty_val
    
    x0 = np.zeros(3*N + 4)
    for i in range(N):
        x0[3*i] = centers[i][0]
        x0[3*i+1] = centers[i][1]
        x0[3*i+2] = 0.0
    x0[3*N] = cx0
    x0[3*N+1] = cy0
    x0[3*N+2] = S0
    x0[3*N+3] = 0.0
    
    bounds = [(None, None)] * (3*N) + [(None, None), (None, None), (1.0, None), (0, 360)]
    
    best_S = S0
    best_x = x0.copy()
    
    try:
        res = minimize(penalized_objective, x0, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
        
        if res.x[3*N+2] < best_S:
            best_S = res.x[3*N+2]
            best_x = res.x.copy()
        
        if time.time() - start_time > deadline - 15:
            raise TimeoutError()
        
        # Polish with higher precision
        res2 = minimize(penalized_objective, best_x, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12})
        
        if res2.x[3*N+2] < best_S:
            best_S = res2.x[3*N+2]
            best_x = res2.x.copy()
            
    except Exception:
        pass
    
    inner = []
    for i in range(N):
        ix, iy, ith = best_x[3*i:3*i+3]
        inner.append([float(ix), float(iy), float(ith)])
    
    cx_best = float(best_x[3*N])
    cy_best = float(best_x[3*N+1])
    S_best = float(best_x[3*N+2])
    phi_best = float(best_x[3*N+3])
    
    return inner, (cx_best, cy_best), S_best, phi_best

construct()
