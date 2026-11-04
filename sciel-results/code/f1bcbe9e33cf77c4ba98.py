import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

def construct():
    sqrt3 = np.sqrt(3)
    
    # Unit hexagon vertices (flat-to-flat, 0-degree rotation)
    verts_local = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    # Container normals for a flat-to-flat hexagon (pointing outward from center)
    # For a regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300,
    # the outward normals to the edges are at angles 30, 90, 150, 210, 270, 330
    normals_base = np.array([
        [np.cos(np.pi/6), np.sin(np.pi/6)],
        [np.cos(np.pi/2), np.sin(np.pi/2)],
        [np.cos(5*np.pi/6), np.sin(5*np.pi/6)],
        [np.cos(7*np.pi/6), np.sin(7*np.pi/6)],
        [np.cos(3*np.pi/2), np.sin(3*np.pi/2)],
        [np.cos(11*np.pi/6), np.sin(11*np.pi/6)]
    ])
    
    def get_max_projection(positions, container_angle):
        """
        Compute the minimum side length s such that all hexagon vertices
        fit inside a regular hexagon of side s, centered at origin, rotated by container_angle.
        """
        # Rotate positions by -container_angle to align container with base orientation
        c = np.cos(-container_angle)
        s_a = np.sin(-container_angle)
        R = np.array([[c, -s_a], [s_a, c]])
        
        # All vertices in world coords, then rotate to container frame
        all_verts = positions[:, np.newaxis, :] + verts_local[np.newaxis, :, :]
        all_verts_rot = np.einsum('ij,kmj->kmi', R, all_verts)
        
        # Project onto the 6 normals
        projections = np.einsum('ijk,nj->ikn', all_verts_rot, normals_base)
        
        # For a hexagon of side s centered at origin, the constraint is:
        # |projection| <= s * sqrt(3) / 2 for each normal direction
        # So s >= max(|projection|) / (sqrt(3)/2)
        max_abs_proj = np.max(np.abs(projections))
        return max_abs_proj / (sqrt3 / 2)
    
    def cost(x):
        positions = x[:24].reshape(12, 2)
        container_angle = x[24]
        
        s = get_max_projection(positions, container_angle)
        
        # Add penalty for overlapping hexagons
        # Two hexagons overlap if their centers are too close
        # Minimum distance between centers of two unit hexagons that don't overlap:
        # For two identical regular hexagons, the minimum center distance to avoid overlap
        # depends on relative rotation. For simplicity, use a conservative bound.
        # The inradius of a unit hexagon is sqrt(3)/2 ≈ 0.866
        # So centers must be at least 2 * sqrt(3)/2 = sqrt(3) apart for axis-aligned hexagons
        # But with rotation, this can be smaller. Use sqrt(3) as a safe lower bound.
        penalty = 0.0
        min_dist = sqrt3  # conservative minimum distance
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    penalty += (min_dist - d) ** 2
        
        return s + 100.0 * penalty
    
    # Initial positions: hexagonal lattice arrangement
    # Try a few different initial configurations and pick the best
    
    best_s = np.inf
    best_positions = None
    best_angle = 0.0
    
    # Configuration 1: 3 rows of 4, straight grid
    positions1 = []
    for row in range(3):
        y = (row - 1) * sqrt3
        for col in range(4):
            x = (col - 1.5) * 2.0
            positions1.append([x, y])
    positions1 = np.array(positions1)
    
    # Configuration 2: 3 rows of 4, with offset on top/bottom rows
    positions2 = []
    for row in range(3):
        y = (row - 1) * sqrt3
        for col in range(4):
            x = (col - 1.5) * 2.0
            if row != 1:  # top and bottom rows offset by 1
                x += 1.0
            positions2.append([x, y])
    positions2 = np.array(positions2)
    
    # Configuration 3: Hexagonal cluster pattern
    # Center at (0,0), ring of 6, then partial second ring
    positions3 = []
    positions3.append([0.0, 0.0])  # center
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        positions3.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
    # Ring 2: 5 more hexagons (to make 12 total)
    for k in range(5):
        angle = k * 2 * np.pi / 5 + np.pi / 5
        positions3.append([4.0 * np.cos(angle), 4.0 * np.sin(angle)])
    positions3 = np.array(positions3)
    
    initial_configs = [positions1, positions2, positions3]
    
    for pos_init in initial_configs:
        x0 = np.concatenate([pos_init.flatten(), [0.0]])
        
        try:
            res = minimize(cost, x0, method='Nelder-Mead', 
                          options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if res.fun < best_s:
                best_s = res.fun
                best_positions = res.x[:24].reshape(12, 2)
                best_angle = res.x[24]
        except:
            pass
    
    # If optimization didn't find something good, try differential evolution for a small search
    if best_s > 4.5:
        # Define bounds for differential evolution
        # Positions roughly in [-5, 5], angle in [-pi, pi]
        bounds = [(-5, 5)] * 24 + [(-np.pi, np.pi)]
        
        try:
            de_result = differential_evolution(cost, bounds, seed=42, maxiter=200, 
                                               tol=1e-6, polish=True)
            if de_result.fun < best_s:
                best_s = de_result.fun
                best_positions = de_result.x[:24].reshape(12, 2)
                best_angle = de_result.x[24]
        except:
            pass
    
    # Final computation of side length
    s_final = get_max_projection(best_positions, best_angle)
    
    # Build inner list
    inner = []
    for i in range(12):
        inner.append([float(best_positions[i][0]), float(best_positions[i][1]), 0.0])
    
    center = (0.0, 0.0)
    side = float(s_final)
    angle = float(np.degrees(best_angle))
    
    return inner, center, side, angle
