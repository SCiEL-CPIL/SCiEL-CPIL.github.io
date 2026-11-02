import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Define the 6 normals of the outer hexagon (flat-top orientation)
    # Normals point to the midpoints of the edges.
    # Angles: 30, 90, 150, 210, 270, 330 degrees
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    # Inner hexagon vertices relative to center (side length 1)
    # Flat-top hexagon vertices:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    inner_verts_rel = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    def objective(params):
        # params: [x1, y1, x2, y2, ..., x12, y12]
        centers = params.reshape(N, 2)
        
        # 1. Calculate required outer side length S
        # Get all vertices of all inner hexagons
        # Shape: (N, 6, 2)
        vertices = centers[:, np.newaxis, :] + inner_verts_rel[np.newaxis, :, :]
        
        # Project all vertices onto the 6 normals
        # vertices: (N, 6, 2)
        # normals: (6, 2)
        # projections: (N, 6, 6) -> for each hex, each vertex, each normal
        projections = np.einsum('ijk,kj->ik', vertices, normals)
        
        # The constraint is |projection| <= S * sqrt(3) / 2
        # So S >= 2 * max(|projections|) / sqrt(3)
        max_proj = np.max(np.abs(projections))
        S_req = 2.0 * max_proj / np.sqrt(3)
        
        # 2. Calculate overlap penalty
        # Minimum distance between centers of two non-overlapping aligned hexagons is sqrt(3)
        min_dist = np.sqrt(3)
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return S_req + 100.0 * penalty

    def initial_guess(type):
        # Generate a honeycomb lattice configuration
        u = np.array([np.sqrt(3), 0.0])
        v = np.array([np.sqrt(3)/2, 1.5])
        
        centers = []
        if type == 0:
            # 3x4 grid
            for i in range(3):
                for j in range(4):
                    c = i * u + j * v
                    centers.append(c)
        elif type == 1:
            # 4x3 grid (swapped)
            for i in range(4):
                for j in range(3):
                    c = i * u + j * v
                    centers.append(c)
        elif type == 2:
            # Hexagonal cluster (center + 6 + 5 more?)
            # Let's just shift the 3x4 grid
            for i in range(3):
                for j in range(4):
                    c = i * u + j * v
                    centers.append(c)
        elif type == 3:
            # Random perturbation around 3x4
            for i in range(3):
                for j in range(4):
                    c = i * u + j * v + np.random.normal(0, 0.1, 2)
                    centers.append(c)
        else:
            # 2x6 grid
            for i in range(2):
                for j in range(6):
                    c = i * u + j * v
                    centers.append(c)

        centers = np.array(centers)
        # Center the configuration
        centers -= np.mean(centers, axis=0)
        return centers.flatten()

    best_params = None
    best_score = float('inf')
    
    # Try several initial guesses
    for k in range(10):
        if time.time() > deadline:
            break
            
        guess_type = k % 5
        x0 = initial_guess(guess_type)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.success or res.fun < best_score:
                if objective(res.x) < best_score:
                    best_score = objective(res.x)
                    best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to a valid but loose configuration
        # Just return the initial 3x4 grid centered
        x0 = initial_guess(0)
        best_params = x0

    centers = best_params.reshape(N, 2)
    
    # Calculate final S
    vertices = centers[:, np.newaxis, :] + inner_verts_rel[np.newaxis, :, :]
    projections = np.einsum('ijk,kj->ik', vertices, normals)
    max_proj = np.max(np.abs(projections))
    S_final = 2.0 * max_proj / np.sqrt(3)
    
    # Format output
    # inner: list of [x, y, angle]
    inner = []
    for i in range(N):
        inner.append([float(centers[i][0]), float(centers[i][1]), 0.0])
    
    # Outer hexagon: center (0,0), side S_final, angle 0 (flat top)
    return inner, (0.0, 0.0), float(S_final), 0.0

# EVOLVE-BLOCK-END
