"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    d = np.sqrt(3)  # distance between adjacent hex centers in tessellation
    
    # 7 hexagons in a flower: 1 center + 6 around
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * np.pi / 3
        centers.append((d * np.cos(angle), d * np.sin(angle)))
    
    # Add 4 more hexagons. Let's try extending in 4 directions.
    # The 6 surrounding are at angles 0,60,120,180,240,300.
    # Adjacent to hex at angle 0 and 60: position (3d/2, d*sqrt(3)/2) = (3, sqrt(3))
    # Let's add 4 hexagons at:
    # (3, sqrt(3)), (3, -sqrt(3)), (-3, sqrt(3)), (-3, -sqrt(3))
    # These are at distance sqrt(9+3)=sqrt(12)=2*sqrt(3) from origin
    
    extra = [
        (3.0, np.sqrt(3)),
        (3.0, -np.sqrt(3)),
        (-3.0, np.sqrt(3)),
        (-3.0, -np.sqrt(3)),
    ]
    for x, y in extra:
        centers.append((x, y))
    
    inner = [[float(x), float(y), 0.0] for x, y in centers]
    
    # Find minimal enclosing regular hexagon
    def hexagon_violation(params, pts):
        cx, cy, s, theta = params
        if s < 1e-6:
            return 1e10
        r = s * np.sqrt(3) / 2
        max_viol = 0.0
        for k in range(6):
            nk = theta + k * np.pi / 3
            cn = np.cos(nk)
            sn = np.sin(nk)
            for px, py in pts:
                val = (px - cx) * cn + (py - cy) * sn - r
                if val > max_viol:
                    max_viol = val
        return max_viol
    
    def objective(params):
        cx, cy, s, theta = params
        if s < 1e-6:
            return 1e10
        viol = hexagon_violation(params, centers)
        if viol > 0:
            return s + viol * viol * 1000 + viol * 10
        return s
    
    cx0 = np.mean([c[0] for c in centers])
    cy0 = np.mean([c[1] for c in centers])
    max_dist = max(np.hypot(c[0] - cx0, c[1] - cy0) for c in centers)
    s0 = max_dist * 2 / np.sqrt(3) + 0.5
    
    best_s = float('inf')
    best_params = None
    
    for theta0 in np.linspace(0, np.pi/3, 7):
        for s_scale in [0.9, 1.0, 1.1]:
            x0 = [cx0, cy0, s0 * s_scale, theta0]
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x.copy()
    
    if best_params is not None:
        res = minimize(objective, best_params, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x.copy()
    
    cx, cy, s, theta = best_params
    
    return inner, (float(cx), float(cy)), float(s), float(np.degrees(theta))
# EVOLVE-BLOCK-END
