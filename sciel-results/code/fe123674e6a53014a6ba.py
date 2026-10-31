"""Packing unit hexagons in a regular hexagon using differential evolution."""
import numpy as np
from scipy.optimize import differential_evolution

N = 11

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point is inside a regular hexagon using support functions."""
    angle = np.radians(angle_deg)
    apothem = side * np.sqrt(3) / 2
    dx = px - cx
    dy = py - cy
    ca, sa = np.cos(-angle), np.sin(-angle)
    rx = ca * dx - sa * dy
    ry = sa * dx + ca * dy
    for i in range(6):
        a = i * np.pi / 3
        nx, ny = np.cos(a), np.sin(a)
        proj = rx * nx + ry * ny
        if proj > apothem:
            return False, proj - apothem
    return True, 0.0

def hexagon_support_distance(h1, h2, h1_angle, h2_angle):
    """Approximate minimum distance between two unit hexagons."""
    dx = h2[0] - h1[0]
    dy = h2[1] - h1[1]
    dist = np.sqrt(dx*dx + dy*dy)
    return dist - 2.0

def penalty_function(params, outer_side):
    """Compute penalty for a given configuration."""
    n = len(params) // 3
    positions = params.reshape(n, 3)
    centers = positions[:, :2]
    angles = positions[:, 2]
    
    total_penalty = 0.0
    cx, cy = 0.0, 0.0
    container_angle = 0.0
    
    for i in range(n):
        c = centers[i]
        a = angles[i]
        for j in range(6):
            va = a + j * np.pi / 3
            vx = c[0] + np.cos(va)
            vy = c[1] + np.sin(va)
            inside, violation = point_in_hexagon(vx, vy, cx, cy, outer_side, container_angle)
            if not inside:
                total_penalty += violation ** 2
    
    for i in range(n):
        for j in range(i+1, n):
            d = hexagon_support_distance(centers[i], centers[j], angles[i], angles[j])
            if d < 0:
                total_penalty += d ** 2 * 10.0
    
    return total_penalty

# EVOLVE-BLOCK-START
def construct():
    """
    Return (inner, center, side, angle) where:
    inner: list of [x, y, angle_degrees] for N unit hexagons
    center: (x, y) of outer hexagon
    side: side length of outer hexagon
    angle: rotation of outer hexagon
    """
    def initial_config():
        configs = []
        configs.append([0.0, 0.0, 0.0])
        hex_dist = np.sqrt(3)
        for i in range(6):
            a = i * np.pi / 3
            configs.append([hex_dist * np.cos(a), hex_dist * np.sin(a), 0.0])
        ring2_dist = 2 * hex_dist
        for i in range(12):
            a = i * np.pi / 6 + np.pi / 6
            configs.append([ring2_dist * np.cos(a), ring2_dist * np.sin(a), 0.0])
        return configs[:N]
    
    init_configs = initial_config()
    
    max_dist = 0
    for cfg in init_configs:
        for j in range(6):
            va = np.radians(cfg[2]) + j * np.pi / 3
            vx = cfg[0] + np.cos(va)
            vy = cfg[1] + np.sin(va)
            d = np.sqrt(vx*vx + vy*vy)
            if d > max_dist:
                max_dist = d
    initial_side = max_dist * 1.2
    
    n_params = N * 3 + 1
    
    def objective(x):
        outer_side = x[-1]
        inner_params = x[:N*3]
        if outer_side < 1.0:
            return 1e6
        return penalty_function(inner_params, outer_side) + 0.1 * outer_side
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 180))
    bounds.append((2.0, 15.0))
    
    init_x = []
    for cfg in init_configs:
        init_x.extend([cfg[0], cfg[1], cfg[2]])
    init_x.append(initial_side)
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=150,
        popsize=12,
        tol=1e-5,
        seed=42,
        polish=False
    )
    
    best_x = result.x
    outer_side = best_x[-1]
    inner_params = best_x[:N*3].reshape(N, 3)
    
    inner = []
    for i in range(N):
        x, y, angle_deg = inner_params[i]
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, float(outer_side), angle
# EVOLVE-BLOCK-END
