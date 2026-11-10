import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600 - 30  # Safety margin

    # Precompute base normals for unit hexagon (flat-top orientation)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    base_angles = np.deg2rad(np.arange(30, 360, 60))
    base_normals = np.column_stack((np.cos(base_angles), np.sin(base_angles)))  # (6, 2)

    def hex_vertices(cx, cy, s, a_deg):
        a_rad = np.deg2rad(a_deg)
        angles = a_rad + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack((
            cx + s * np.cos(angles),
            cy + s * np.sin(angles)
        ))

    def get_normals(a_deg):
        """Get the 6 edge normals for a hexagon with given angle."""
        a_rad = np.deg2rad(a_deg)
        c, s = np.cos(a_rad), np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        return base_normals @ R.T

    def is_separated(v1, v2, n1, n2):
        """
        Check if two hexagons are separated.
        v1, v2: (6, 2) vertex arrays
        n1, n2: (6, 2) normal arrays for each hexagon
        Returns True if separated (valid), False if intersecting (invalid).
        """
        for i in range(6):
            n = n1[i]
            p1 = v1 @ n
            p2 = v2 @ n
            if p1.max() < p2.min() - 1e-6 or p2.max() < p1.min() - 1e-6:
                return True
        for i in range(6):
            n = n2[i]
            p1 = v1 @ n
            p2 = v2 @ n
            if p1.max() < p2.min() - 1e-6 or p2.max() < p1.min() - 1e-6:
                return True
        return False

    def get_min_side(positions, angles):
        """
        Compute the minimum side length of the outer hexagon.
        positions: (11, 2)
        angles: (11,)
        """
        outer_normals = base_normals  # (6, 2)
        
        max_proj = 0.0
        for i in range(11):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            projs = verts @ outer_normals.T  # (6, 6)
            max_proj = max(max_proj, np.max(projs))
        
        # Apothem = S * sqrt(3) / 2
        # max_proj <= Apothem
        # S >= 2 * max_proj / sqrt(3)
        S = 2 * max_proj / np.sqrt(3)
        return S

    def smooth_max(vals, beta=100.0):
        """Smooth approximation of max using log-sum-exp."""
        m = np.max(vals)
        if m < -500:
            return m
        return m + np.log(np.sum(np.exp((vals - m) * beta)) / len(vals)) / beta

    def get_min_side_smooth(positions, angles, beta=100.0):
        """
        Smooth version of get_min_side using log-sum-exp for max.
        """
        outer_normals = base_normals  # (6, 2)
        
        all_projs = []
        for i in range(11):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            projs = verts @ outer_normals.T  # (6, 6)
            all_projs.append(projs.flatten())
        
        all_projs = np.concatenate(all_projs)
        max_proj_smooth = smooth_max(all_projs, beta)
        
        # Apothem = S * sqrt(3) / 2
        # max_proj <= Apothem
        # S >= 2 * max_proj / sqrt(3)
        S = 2 * max_proj_smooth / np.sqrt(3)
        return S

    def containment_penalty(positions, angles, S):
        """
        Penalty for inner hexagons not being contained in outer hexagon of side S.
        Returns 0 if all contained, else positive penalty.
        """
        outer_normals = base_normals  # (6, 2)
        apothem = S * np.sqrt(3) / 2
        penalty = 0.0
        for i in range(11):
            verts = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            projs = verts @ outer_normals.T  # (6, 6)
            violations = np.maximum(0, projs - apothem)
            penalty += np.sum(violations ** 2)
        return penalty

    def objective_smooth(x, beta=100.0):
        if time.time() > deadline:
            return 1e10
        
        positions = x[:22].reshape(11, 2)
        angles = x[22:33]
        
        # Calculate min side using smooth surrogate
        S = get_min_side_smooth(positions, angles, beta)
        
        # Check overlaps and compute penalty
        penalty = 0.0
        for i in range(11):
            v1 = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            n1 = get_normals(angles[i])
            for j in range(i+1, 11):
                v2 = hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                n2 = get_normals(angles[j])
                if not is_separated(v1, v2, n1, n2):
                    # Use a smooth-ish penalty based on distance
                    d = np.linalg.norm(positions[i] - positions[j])
                    # If they are overlapping, d is likely < 2.0 for same orientation
                    # We want to push them apart.
                    penalty += max(0, 2.0 - d) ** 2
        
        # Containment penalty
        pen = penalty + 0.1 * containment_penalty(positions, angles, S)
        
        return S + 100 * pen

    def objective_exact(x):
        """Exact objective for final polishing."""
        if time.time() > deadline:
            return 1e10
        
        positions = x[:22].reshape(11, 2)
        angles = x[22:33]
        
        # Calculate min side exactly
        S = get_min_side(positions, angles)
        
        # Check overlaps and compute penalty
        penalty = 0.0
        for i in range(11):
            v1 = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            n1 = get_normals(angles[i])
            for j in range(i+1, 11):
                v2 = hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                n2 = get_normals(angles[j])
                if not is_separated(v1, v2, n1, n2):
                    d = np.linalg.norm(positions[i] - positions[j])
                    penalty += max(0, 2.0 - d) ** 2
        
        pen = penalty + 0.1 * containment_penalty(positions, angles, S)
        
        return S + 100 * pen

    # Initial guess
    # 7 hexes in flower pattern
    # 4 hexes in corners
    sqrt3 = np.sqrt(3)
    init_pos = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        [3, sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [3, -sqrt3]
    ])
    init_angles = np.zeros(11)
    
    x0 = np.concatenate([init_pos.flatten(), init_angles])
    
    best_x = x0.copy()
    best_val = objective_exact(x0)
    
    rng = np.random.default_rng(42)

    # Phase 1: Smooth optimization with low beta (wide basin search)
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb best-so-far
            noise_scale = 0.5 / (restart + 1)
            x_start = best_x + rng.normal(0, noise_scale, size=best_x.shape)
        
        try:
            result = minimize(lambda x: objective_smooth(x, beta=50.0), x_start, method='Nelder-Mead', 
                              options={'maxiter': 15000, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_val:
                best_x = result.x.copy()
                best_val = objective_exact(best_x)
        except Exception:
            pass
    
    # Phase 2: Refine with higher beta (sharper max approximation)
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_start = best_x.copy()
        else:
            noise_scale = 0.2
            x_start = best_x + rng.normal(0, noise_scale, size=best_x.shape)
        
        try:
            result = minimize(lambda x: objective_smooth(x, beta=200.0), x_start, method='Nelder-Mead', 
                              options={'maxiter': 15000, 'xatol': 1e-7, 'fatol': 1e-9})
            if result.fun < best_val:
                best_x = result.x.copy()
                best_val = objective_exact(best_x)
        except Exception:
            pass

    # Phase 3: Final high-precision polish with exact objective
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_start = best_x.copy()
        else:
            noise_scale = 0.1
            x_start = best_x + rng.normal(0, noise_scale, size=best_x.shape)
        
        try:
            result = minimize(objective_exact, x_start, method='Nelder-Mead', 
                              options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_val:
                best_x = result.x.copy()
                best_val = objective_exact(best_x)
        except Exception:
            pass

    # Extract best result
    positions = best_x[:22].reshape(11, 2)
    angles = best_x[22:33]
    
    # Final validation
    S = get_min_side(positions, angles)
    S = max(S, np.sqrt(11))
    
    # Check overlaps again with final positions
    valid = True
    for i in range(11):
        v1 = hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        n1 = get_normals(angles[i])
        for j in range(i+1, 11):
            v2 = hex_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
            n2 = get_normals(angles[j])
            if not is_separated(v1, v2, n1, n2):
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to seed
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    inner_hex_data = np.column_stack((positions, angles))
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = S

    return inner_hex_data, outer_hex_data, outer_hex_side_length
