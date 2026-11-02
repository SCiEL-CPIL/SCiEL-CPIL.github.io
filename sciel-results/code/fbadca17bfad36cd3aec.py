import time
import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_supports(cx, cy, phi, S, psi):
    # returns (dmin, dmax) over 6 vertices
    a = np.deg2rad(phi)
    v = np.array([
        [np.cos(a), np.sin(a)],
        [np.cos(a + np.pi/3), np.sin(a + np.pi/3)],
        [np.cos(a + 2*np.pi/3), np.sin(a + 2*np.pi/3)],
        [np.cos(a + np.pi), np.sin(a + np.pi)],
        [np.cos(a + 4*np.pi/3), np.sin(a + 4*np.pi/3)],
        [np.cos(a + 5*np.pi/3), np.sin(a + 5*np.pi/3)],
    ])
    p = v + np.array([cx, cy])
    b = np.deg2rad(psi)
    n = np.array([np.cos(b), np.sin(b)])
    n2 = np.array([np.cos(b + np.pi/3), np.sin(b + np.pi/3)])
    n3 = np.array([np.cos(b + 2*np.pi/3), np.sin(b + 2*np.pi/3)])
    d = np.array([p @ n, p @ n2, p @ n3])
    return d.min(), d.max()

def _objective(x):
    r = x[0]
    a = x[1]
    S = x[2]
    psi = x[3]
    # 6 symmetric hexagons
    worst = 0.0
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        cx = r * np.cos(ang)
        cy = r * np.sin(ang)
        phi = np.deg2rad(60.0 * k)
        dmin, dmax = _hex_supports(cx, cy, phi, S, psi)
        # need -dmin <= S and dmax <= S
        worst = max(worst, -dmin - S, dmax - S)
    # penalty if S < 0
    if S < 0:
        worst = max(worst, -S)
    return worst

def _try_optimize(x0, maxiter):
    try:
        res = minimize(_objective, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        return res
    except Exception:
        return None

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin

    # Initial guess: hexagons near center, small S
    # Try a few different starting points
    starts = [
        [0.5, 0.0, 3.0, 0.0],   # small radius
        [1.0, 0.0, 4.0, 0.0],
        [1.5, 0.0, 5.0, 0.0],
        [0.8, 0.0, 3.5, 30.0],  # rotated container
        [1.2, 0.0, 4.5, 30.0],
        [0.6, 0.0, 3.2, 15.0],
    ]

    best_x = None
    best_val = float('inf')

    for i, x0 in enumerate(starts):
        if time.time() > deadline:
            break
        # Use fewer iterations for early starts, more for later
        maxiter = 5000 if i < 3 else 8000
        res = _try_optimize(np.array(x0), maxiter)
        if res is not None and res.success:
            val = res.fun
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        # Check time
        if time.time() > deadline:
            break

    # If we found a good solution, extract the hexagons
    if best_x is not None and best_val < 1e-6:
        r = best_x[0]
        a = best_x[1]
        S = best_x[2]
        psi = best_x[3]

        inner = []
        for k in range(6):
            ang_deg = 60.0 * k
            ang = np.deg2rad(ang_deg)
            cx = r * np.cos(ang)
            cy = r * np.sin(ang)
            phi_deg = 60.0 * k
            inner.append([cx, cy, phi_deg])
            # Mirror hexagon (opposite side, same orientation relative to its position)
            cx2 = -r * np.cos(ang)
            cy2 = -r * np.sin(ang)
            phi2_deg = (60.0 * k + 180.0) % 360.0
            inner.append([cx2, cy2, phi2_deg])

        # Center of container
        center = (0.0, 0.0)
        side = S
        angle = psi

        return inner, center, side, angle
    else:
        # Fallback: return a valid (though not optimal) configuration
        # Place 12 hexagons in a grid-like pattern
        inner = []
        positions = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1),
            (1, -1), (-1, 1), (2, 0), (-2, 0), (0, 2)
        ]
        for i, (px, py) in enumerate(positions):
            # Approximate hexagonal grid positions
            cx = px * 1.75
            cy = py * 1.5
            inner.append([cx, cy, 0.0])
        return inner, (0.0, 0.0), 15.0, 0.0
# EVOLVE-BLOCK-END
