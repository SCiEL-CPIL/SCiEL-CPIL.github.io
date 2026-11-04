"""Packing 12 unit hexagons in minimal enclosing regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with given center, side, rotation."""
    angle = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0 + angle
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def enclosing_hexagon_side(inner_hexes, outer_center, outer_angle_deg):
    """
    Compute the minimum side length of a regular hexagon (at given center and angle)
    that encloses all inner hexagons.
    """
    cx, cy = outer_center
    theta = np.radians(outer_angle_deg)
    normal_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0 + theta
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_projection = 0.0
    for (ix, iy, iangle) in inner_hexes:
        verts = hexagon_vertices(ix, iy, 1.0, iangle)
        diffs = verts - np.array([cx, cy])
        projections = diffs @ normals
        max_projection = max(max_projection, np.max(projections))
    
    return max_projection / (np.sqrt(3) / 2.0)

def objective(params):
    """
    params: [x1,y1,ang1, ..., x12,y12,ang12, cx, cy, cout]
    """
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        ang = params[3*i+2]
        inner.append((x, y, ang))
    
    cx = params[3*N]
    cy = params[3*N+1]
    cout = params[3*N+2]
    
    s = enclosing_hexagon_side(inner, (cx, cy), cout)
    
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            min_dist = np.sqrt(3) - 0.005
            if dist < min_dist:
                overlap_penalty += (min_dist - dist) ** 2
    
    return s + 100.0 * overlap_penalty

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    sqrt3 = np.sqrt(3)
    
    # Hexagonal cluster initial layout
    hex_positions = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (-1, -1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2),
        (1, -1), (-1, 1),
    ]
    
    initial_inner = []
    for (q, r) in hex_positions:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        initial_inner.append((x, y, 0.0))
    
    params0 = np.zeros(3 * N + 3)
    for i, (x, y, a) in enumerate(initial_inner):
        params0[3*i] = x
        params0[3*i+1] = y
        params0[3*i+2] = a
    params0[3*N] = 0.0
    params0[3*N+1] = 0.0
    params0[3*N+2] = 0.0
    
    best_params = params0.copy()
    best_score = objective(params0)
    
    # Try different outer angles
    for test_angle in [0, 30, 60, 90, 120, 150]:
        params_test = params0.copy()
        params_test[3*N+2] = test_angle
        score_test = objective(params_test)
        if score_test < best_score:
            best_score = score_test
            best_params = params_test
    
    # Nelder-Mead optimization
    if time.time() < deadline:
        try:
            result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Polish pass
    if time.time() < deadline:
        try:
            result2 = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x
        except Exception:
            pass
    
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        ang = best_params[3*i+2]
        inner.append([x, y, ang])
    
    cx = best_params[3*N]
    cy = best_params[3*N+1]
    cout = best_params[3*N+2]
    
    final_side = enclosing_hexagon_side(inner, (cx, cy), cout)
    
    return inner, (cx, cy), final_side, cout
# EVOLVE-BLOCK-END
