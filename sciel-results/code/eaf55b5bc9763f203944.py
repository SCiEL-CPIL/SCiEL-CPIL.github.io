"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def hex_vertices(cx, cy, s, rot_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation rot_deg."""
    rot = np.radians(rot_deg)
    angles = rot + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def hexagonal_distance_point(x, y, cx, cy, rot_deg):
    """
    Compute the minimum side length S such that point (x,y) is inside
    a regular hexagon with center (cx,cy), side S, rotation rot_deg.
    """
    dx, dy = x - cx, y - cy
    rot = np.radians(-rot_deg)
    rx = dx * np.cos(rot) - dy * np.sin(rot)
    ry = dx * np.sin(rot) + dy * np.cos(rot)
    # Reference hexagon: vertices at 0,60,120,180,240,300 degrees
    # Edge normals at 30,90,150,210,270,330 degrees
    angles = np.radians(30 + 60 * np.arange(6))
    dots = np.abs(rx * np.cos(angles) + ry * np.sin(angles))
    # Inradius = S * sqrt(3)/2, so S = 2*dots/sqrt(3)
    return np.max(dots) * 2.0 / np.sqrt(3.0)

def compute_max_hex_dist(inner_xy, cx, cy, rot):
    """Compute max hexagonal distance of all inner hexagon vertices from outer hexagon center."""
    max_d = 0.0
    for i in range(len(inner_xy)):
        ix, iy = inner_xy[i]
        verts = hex_vertices(ix, iy, 1.0, 0.0)
        for v in verts:
            d = hexagonal_distance_point(v[0], v[1], cx, cy, rot)
            if d > max_d:
                max_d = d
    return max_d

def objective(params):
    cx, cy, rot = params[0], params[1], params[2]
    inner_xy = params[3:].reshape(N, 2)
    return compute_max_hex_dist(inner_xy, cx, cy, rot)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Honeycomb initial positions
    positions = []
    positions.append([0.0, 0.0])
    for k in range(6):
        angle = k * np.pi / 3.0
        positions.append([np.sqrt(3.0) * np.cos(angle), np.sqrt(3.0) * np.sin(angle)])
    for k in range(5):
        angle = k * np.pi / 3.0 + np.pi / 6.0
        positions.append([2.0 * np.sqrt(3.0) * np.cos(angle), 2.0 * np.sqrt(3.0) * np.sin(angle)])
    positions = np.array(positions)

    best_val = float('inf')
    best_params = None

    rng = np.random.RandomState(42)
    num_restarts = 30

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            pos_init = positions.copy()
        else:
            pos_init = positions + rng.normal(0, 0.5, positions.shape)
            if restart % 3 == 1:
                scale = 1.0 + rng.uniform(-0.15, 0.15)
                pos_init = positions * scale + rng.normal(0, 0.3, positions.shape)
            elif restart % 3 == 2:
                # Try a more spread out initial
                pos_init = positions * (1.0 + rng.uniform(0.0, 0.3)) + rng.normal(0, 0.4, positions.shape)

        cx_init = rng.uniform(-1, 1)
        cy_init = rng.uniform(-1, 1)
        rot_init = rng.uniform(0, 60)

        params = np.concatenate([[cx_init, cy_init, rot_init], pos_init.flatten()])

        try:
            result = minimize(objective, params, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-6})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

        if time.time() > deadline:
            break

    if best_params is None:
        # Fallback
        cx, cy, rot = 0.0, 0.0, 0.0
        inner_xy = positions
    else:
        cx, cy, rot = best_params[0], best_params[1], best_params[2]
        inner_xy = best_params[3:].reshape(N, 2)

    inner = [[float(inner_xy[i][0]), float(inner_xy[i][1]), 0.0] for i in range(N)]
    return inner, (float(cx), float(cy)), float(best_val), float(rot)
# EVOLVE-BLOCK-END
