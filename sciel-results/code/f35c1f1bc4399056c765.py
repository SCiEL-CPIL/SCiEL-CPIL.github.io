import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    sqrt3 = math.sqrt(3)

    # Honeycomb adjacency: 3 rows of 4
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7 (offset by sqrt3/2)
    # Row 2: 8,9,10,11
    # Adjacent pairs (sharing an edge in honeycomb)
    adjacency = [
        (0,1),(1,2),(2,3),  # row 0 horizontal
        (4,5),(5,6),(6,7),  # row 1 horizontal
        (8,9),(9,10),(10,11), # row 2 horizontal
        (0,4),(1,4),(1,5),(2,5),(2,6),(3,6),(3,7),  # row0-row1
        (4,8),(5,8),(5,9),(6,9),(6,10),(7,10),(7,11), # row1-row2
    ]
    
    # Minimum center distance for non-overlapping unit hexagons
    # Two unit hexagons sharing an edge have center distance = 2 * inradius = sqrt(3)
    min_dist = sqrt3 - 1e-9

    # Hexagon vertices for a unit hexagon at origin, angle 0 (flat-top)
    hex_verts_0 = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2],
    ])

    def get_hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return cx + side * (R @ hex_verts_0.T).T

    def compute_outer_side(positions, angle_deg=0.0):
        """Compute minimum outer hexagon side to contain all inner hexagons."""
        angle_rad = math.radians(angle_deg)
        
        # Generate all vertices of all inner hexagons
        all_verts = []
        for i in range(N):
            cx, cy = positions[i]
            verts = get_hex_vertices(cx, cy, 1.0, 0.0)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        max_width = 0.0
        for k in range(6):
            normal_angle = math.radians(30 + k * 60) + angle_rad
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            projections = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            width = projections.max() - projections.min()
            if width > max_width:
                max_width = width
        
        return max_width / sqrt3

    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        positions = params.reshape(N, 2)
        
        # Compute outer side length needed
        outer_side = compute_outer_side(positions)
        
        # Check ALL pairs for overlap to ensure validity
        # This is O(N^2) but N=12 is small (66 pairs)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < min_dist*min_dist:
                    dist = math.sqrt(dist_sq)
                    deficit = min_dist - dist
                    # Strong penalty to prevent overlap
                    penalty += deficit * deficit * 100.0
        
        return outer_side + penalty

    # Initial guess: standard honeycomb lattice
    # This is a valid packing, so it will never fail the area check
    init_positions = np.zeros((N, 2))
    for row in range(3):
        for col in range(4):
            idx = row * 4 + col
            x = col * sqrt3 + (sqrt3/2 if row % 2 == 1 else 0)
            y = row * 1.5
            init_positions[idx] = [x, y]
    
    # Center the configuration
    init_positions -= init_positions.mean(axis=0)
    
    init_params = init_positions.flatten()
    
    best_params = init_params.copy()
    best_val = objective(init_params)
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            objective,
            init_params,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'maxfev': 10000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        # Only accept if it's better AND valid (penalty is small)
        # We check penalty by re-evaluating or trusting the objective if it's lower
        # Since penalty is added, a lower objective implies better fit or less overlap.
        # However, we must ensure the final result is valid.
        if result.fun < best_val:
            # Check if the result is valid (no significant overlap)
            res_positions = result.x.reshape(N, 2)
            is_valid = True
            for i in range(N):
                for j in range(i+1, N):
                    dx = res_positions[i, 0] - res_positions[j, 0]
                    dy = res_positions[i, 1] - res_positions[j, 1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < (min_dist - 1e-6)**2:
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                best_val = result.fun
                best_params = result.x.copy()
                
    except Exception:
        pass
    
    # Try a few more restarts with small perturbations if time allows
    if time.time() < deadline - 30:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            perturbation = np.random.normal(0, 0.5, size=24)
            start_params = best_params + perturbation
            try:
                result = minimize(
                    objective,
                    start_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'maxfev': 6000,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < best_val:
                    res_positions = result.x.reshape(N, 2)
                    is_valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = res_positions[i, 0] - res_positions[j, 0]
                            dy = res_positions[i, 1] - res_positions[j, 1]
                            dist_sq = dx*dx + dy*dy
                            if dist_sq < (min_dist - 1e-6)**2:
                                is_valid = False
                                break
                        if not is_valid:
                            break
                    
                    if is_valid:
                        best_val = result.fun
                        best_params = result.x.copy()
            except Exception:
                pass
    
    # Extract final positions
    final_positions = best_params.reshape(N, 2)
    
    # Compute final outer side
    final_outer_side = compute_outer_side(final_positions)
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(final_positions[i, 0]), float(final_positions[i, 1]), 0.0])
    
    # Outer hexagon centered at origin
    return inner, (0.0, 0.0), float(final_outer_side), 0.0
# EVOLVE-BLOCK-END
