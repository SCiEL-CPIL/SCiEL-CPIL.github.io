import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    sqrt3 = np.sqrt(3)
    
    # Generate honeycomb positions for unit flat-top hexagons
    positions = []
    for c in range(-5, 6):
        for r in range(-5, 6):
            x = 1.5 * c
            # Offset for odd columns
            offset = (sqrt3 / 2) if (c % 2 != 0) else 0
            # Handle negative c: c % 2 in Python gives 1 for negative odd
            if c < 0:
                offset = (sqrt3 / 2) if ((c + 1) % 2 != 0) else 0
            y = sqrt3 * r + offset
            positions.append((x, y))
    
    positions = np.array(positions)
    
    # Pick 12 closest to origin for compactness
    dists = np.linalg.norm(positions, axis=1)
    idx = np.argsort(dists)[:N]
    centers = positions[idx]
    
    inner_angles = np.zeros(N)
    
    def support_hex(theta, alpha):
        js = np.arange(6) * np.pi / 3
        return np.max(np.cos(theta - alpha - js))
    
    def containment_violation(params):
        cx, cy, S, beta = params
        if S <= 0:
            return 1e6
        inradius_outer = S * sqrt3 / 2
        max_violation = 0
        for i in range(N):
            xi, yi = centers[i]
            alpha_i = inner_angles[i]
            dx = xi - cx
            dy = yi - cy
            for k in range(6):
                theta_k = beta + k * np.pi / 3
                nx = np.cos(theta_k)
                ny = np.sin(theta_k)
                dist = dx * nx + dy * ny
                sup = support_hex(theta_k, alpha_i)
                violation = dist + sup - inradius_outer
                if violation > max_violation:
                    max_violation = violation
        return max_violation
    
    def objective(params):
        cx, cy, S, beta = params
        if S <= 0:
            return 1e6
        return S + 100 * max(0, containment_violation(params))
    
    cx0, cy0 = np.mean(centers, axis=0)
    max_dist = np.max(np.linalg.norm(centers - [cx0, cy0], axis=1))
    S0 = max_dist + 2.0
    
    best_S = np.inf
    best_params = None
    
    for beta_start in [0, np.pi/6, np.pi/3, 2*np.pi/3, np.pi/2]:
        x0_try = np.array([cx0, cy0, S0, beta_start])
        res = minimize(objective, x0_try, method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-7, 'fatol': 1e-7})
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x
    
    cx, cy, S, beta = best_params
    
    inner = []
    for i in range(N):
        inner.append([float(centers[i][0]), float(centers[i][1]), 0.0])
    
    return inner, (float(cx), float(cy)), float(S), float(np.degrees(beta))
