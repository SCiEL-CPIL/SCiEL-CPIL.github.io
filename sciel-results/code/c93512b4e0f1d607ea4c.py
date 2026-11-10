import numpy as np
from itertools import combinations
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0
    
    # Generate 19 centers for a hexagon of side 2 (radius 2 in axial coords)
    # Axial coords (q, r) where |q| <= 2, |r| <= 2, |q+r| <= 2
    centers_axial = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                centers_axial.append((q, r))
    
    # Convert to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    centers_cart = []
    for q, r in centers_axial:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers_cart.append(np.array([x, y]))
    centers_cart = np.array(centers_cart)
    
    # Precompute support of unit hexagon in 6 directions of outer hexagon (phi=0)
    # Outer hex vertices at 0, 60, ...
    # Edge normals at 30, 90, ...
    # n_k = (cos(30+60k), sin(30+60k))
    # h_unit(n) = max dot product of unit hex vertices with n
    # Unit hex vertices: (cos(60i), sin(60i))
    unit_vertices = np.array([
        [np.cos(np.deg2rad(60 * i)), np.sin(np.deg2rad(60 * i))] for i in range(6)
    ])
    
    # Directions of outer hex edges (normals)
    # If outer hex has angle 0, vertices at 0, 60...
    # Edge between 0 and 60 has normal at 30.
    outer_normals = np.array([
        [np.cos(np.deg2rad(30 + 60 * k)), np.sin(np.deg2rad(30 + 60 * k))] for k in range(6)
    ])
    
    # h_unit for these 6 directions
    h_unit = np.max(unit_vertices @ outer_normals.T, axis=0)
    
    best_s = float('inf')
    best_indices = None
    
    # Brute force all combinations of 12
    n_total = len(centers_cart)
    for indices in combinations(range(n_total), 12):
        if time.time() > deadline - 10:
            break
            
        C = centers_cart[list(indices)]
        
        # Compute max projection for each of the 6 outer normals
        # Proj = C @ n_k
        # Max Proj = max(C @ n_k)
        # Required inradius r_i = max(Proj) + h_unit[k]
        # s = r_i * 2 / sqrt(3)
        
        # C is (12, 2), outer_normals is (6, 2)
        # Projections: (12, 6)
        projs = C @ outer_normals.T
        max_projs = np.max(projs, axis=0)
        
        required_inradii = max_projs + h_unit
        s = np.max(required_inradii) * 2.0 / np.sqrt(3)
        
        if s < best_s:
            best_s = s
            best_indices = indices

    if best_indices is None:
        # Fallback to a simple configuration
        best_indices = list(range(12))
        # Recalculate s for fallback
        C = centers_cart[list(best_indices)]
        projs = C @ outer_normals.T
        max_projs = np.max(projs, axis=0)
        required_inradii = max_projs + h_unit
        best_s = np.max(required_inradii) * 2.0 / np.sqrt(3)

    # Construct return values
    inner_hex_data = np.zeros((12, 3))
    for i, idx in enumerate(best_indices):
        inner_hex_data[i, 0] = centers_cart[idx, 0]
        inner_hex_data[i, 1] = centers_cart[idx, 1]
        inner_hex_data[i, 2] = 0.0  # Angle of inner hexagons is 0

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
