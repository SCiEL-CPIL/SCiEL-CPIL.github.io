"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    """Place 11 unit hexagons on a hexagonal lattice inside a regular hexagon.
    We select a compact subset of the 19-hex lattice cluster (7 center + 4 outer)
    and optimize the outer hexagon rotation to minimize side length.
    """
    s3 = math.sqrt(3.0)
    u = (1.5, s3 / 2.0)
    v = (0.5, s3 / 2.0)

    # Generate 11 points: 7 inner (dist sqrt(3) or 0) + 4 outer (dist sqrt(7))
    # Inner: (0,0), (1,0), (0,1), (-1,0), (0,-1), (1,-1), (-1,1)
    # Outer: (1,1), (0,2), (-1,1), (-2,0)
    indices = [
        (0, 0),
        (1, 0), (0, 1), (-1, 0), (0, -1), (1, -1), (-1, 1),
        (1, 1), (0, 2), (-1, 1), (-2, 0)
    ]
    
    centers = []
    for i, j in indices:
        x = i * u[0] + j * v[0]
        y = i * u[1] + j * v[1]
        centers.append((x, y))

    # Optimize outer hexagon rotation theta
    # S(theta) = 2/sqrt(3) * (1 + max_i max_k |C_i . n_k(theta)|)
    # n_k(theta) = (cos(theta + k*pi/3), sin(theta + k*pi/3))
    
    def calc_side(theta):
        max_proj = 0.0
        for cx, cy in centers:
            for k in range(6):
                ang = theta + k * math.pi / 3.0
                nx = math.cos(ang)
                ny = math.sin(ang)
                proj = abs(cx * nx + cy * ny)
                if proj > max_proj:
                    max_proj = proj
        return (2.0 / s3) * (1.0 + max_proj)

    # Search for optimal theta in [0, pi/6]
    # The function is piecewise linear in sin/cos, so min is at critical points
    # or boundaries. Critical points occur when |C_i . n_k| = |C_j . n_m| etc.
    # Simple grid search is safe and fast enough.
    best_theta = 0.0
    best_side = float('inf')
    
    # Grid search with 1000 steps
    for step in range(1001):
        theta = (math.pi / 6.0) * (step / 1000.0)
        side = calc_side(theta)
        if side < best_side:
            best_side = side
            best_theta = theta
            
    # Refine around best_theta
    for step in range(-100, 101):
        theta = best_theta + (math.pi / 6.0) * (step / 10000.0)
        if theta < 0: theta += math.pi / 3.0 # Wrap? No, symmetry is pi/6
        if theta > math.pi/6: theta -= math.pi/6
        side = calc_side(theta)
        if side < best_side:
            best_side = side
            best_theta = theta

    # Convert best_theta to degrees for output
    angle_deg = math.degrees(best_theta)
    
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    
    return inner, (0.0, 0.0), best_side, angle_deg
# EVOLVE-BLOCK-END
