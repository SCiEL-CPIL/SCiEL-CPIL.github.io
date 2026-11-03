"""Structural change: spectral graph optimization for hexagon packing."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    # Precompute unit hexagon vertices (side length 1, centered at origin)
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])

    def support_unit(n_vec):
        """Support function of unit hexagon in direction n_vec (unit vector)."""
        projections = unit_verts @ n_vec
        return np.max(projections)

    def calc_min_side(positions, theta_deg):
        """
        Calculate minimum side length S of outer hexagon centered at origin
        with rotation theta_deg that contains all unit hexagons at given positions.
        positions: list of (x, y)
        theta_deg: rotation angle of outer hexagon in degrees
        """
        theta_rad = math.radians(theta_deg)
        
        max_proj = 0.0
        for k in range(6):
            # Normal directions for outer hexagon (pointing outward from center)
            normal_angle = math.radians(30 + k * 60) + theta_rad
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            n_vec = np.array([nx, ny])
            
            h_unit = support_unit(n_vec)
            
            for x, y in positions:
                proj = x * nx + y * ny + h_unit
                if proj > max_proj:
                    max_proj = proj
        
        # Side length S: for a regular hexagon, the apothem is S*sqrt(3)/2
        # The support distance from center to face is S*sqrt(3)/2
        # So S = max_proj / (sqrt(3)/2) = 2*max_proj/sqrt(3)
        return 2.0 * max_proj / sqrt3

    def objective(params):
        """
        params: [x_0, y_0, x_1, y_1, ..., x_11, y_11, theta]
        theta is outer hexagon rotation in radians
        """
        if time.time() > deadline:
            return 1e10
        
        pos = [(params[i], params[i+1]) for i in range(0, 2*N, 2)]
        theta_rad = params[2*N]
        theta_deg = math.degrees(theta_rad)
        
        S = calc_min_side(pos, theta_deg)
        
        # Penalty for overlaps: hexagons of side 1 need center distance >= 2
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 2.0 - 1e-9:
                    penalty += (2.0 - dist) ** 2
        
        return S + 100.0 * penalty

    # Generate initial positions using a spectral/adjacency graph approach
    # Define a honeycomb-like adjacency: 12 hexagons in a compact cluster
    # Use a 3-4-3-2 arrangement or similar compact shape
    
    # Start with a reasonable initial configuration: 19-hex cluster minus 7
    # 19-hex cluster (radius 2 hexagonal grid)
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q) <= 2 and abs(r) <= 2 and abs(q+r) <= 2:
                x = q + r/2.0
                y = r * sqrt3/2.0
                centers.append((x, y))
    
    # centers has 19 points. Remove 7 to get 12.
    # Remove corners (6 at distance 4) and one side hex
    corner_centers = []
    side_centers = []
    inner_centers = []
    
    for x, y in centers:
        dist = math.sqrt(x*x + y*y)
        if dist > 3.8:  # Corners at distance 4.0
            corner_centers.append((x, y))
        elif dist > 3.0:  # Side hexes at distance sqrt(12) ~ 3.46
            side_centers.append((x, y))
        else:
            inner_centers.append((x, y))
    
    # inner_centers should have 7 (center + 6 ring)
    # side_centers should have 6
    # corner_centers should have 6
    # Total: 7 + 6 + 6 = 19
    
    # Keep all 7 inner, keep 5 of 6 side (remove one), remove all 6 corners
    # That gives 7 + 5 = 12
    
    final_centers = inner_centers + side_centers[:5]
    
    if len(final_centers) != 12:
        # Fallback: create a 4x3 honeycomb grid
        final_centers = []
        for row in range(4):
            y = row * sqrt3
            for col in range(3):
                if row % 2 == 0:
                    x = (col - 1) * 2.0
                else:
                    x = (col - 1) * 2.0 + 1.0
                final_centers.append((x, y))
    
    # Center the arrangement at origin
    xs = [p[0] for p in final_centers]
    ys = [p[1] for p in final_centers]
    cx_offset = (min(xs) + max(xs)) / 2.0
    cy_offset = (min(ys) + max(ys)) / 2.0
    centered_positions = [(x - cx_offset, y - cy_offset) for x, y in final_centers]
    
    # Convert to flat parameter vector
    init_flat = []
    for x, y in centered_positions:
        init_flat.extend([x, y])
    init_flat.append(0.0)  # theta = 0 initially
    
    # Optimize using Nelder-Mead
    best_params = init_flat[:]
    best_score = objective(init_flat)
    
    try:
        res = minimize(objective, init_flat, method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-10,
                                'adaptive': True})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Try a few different initial rotations to escape local minima
    if time.time() < deadline - 30:
        for theta_init in [math.pi/6, math.pi/3, -math.pi/6, -math.pi/3]:
            init_flat2 = init_flat[:]
            init_flat2[2*N] = theta_init
            try:
                res2 = minimize(objective, init_flat2, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10,
                                        'adaptive': True})
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_params = res2.x
            except Exception:
                pass
            if time.time() > deadline - 10:
                break
    
    # Extract results
    pos = [(best_params[i], best_params[i+1]) for i in range(0, 2*N, 2)]
    theta_rad = best_params[2*N]
    theta_deg = math.degrees(theta_rad)
    
    final_side = calc_min_side(pos, theta_deg)
    
    inner = []
    for x, y in pos:
        inner.append([x, y, 0.0])
    
    return inner, (0.0, 0.0), final_side, theta_deg
