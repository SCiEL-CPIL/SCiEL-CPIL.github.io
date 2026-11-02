import numpy as np
from scipy.optimize import differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Precompute hexagon vertices offsets for a unit hexagon (side 1)
    # Vertices are at distance 1 from center
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_offsets = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Define the 12 grid positions (indices) for a 3x4 lattice
    i_vals = [0, 1, 2]
    j_vals = [0, 1, 2, 3]
    grid_indices = [(i, j) for i in i_vals for j in j_vals]
    
    def get_centers(u, v):
        centers = []
        for i, j in grid_indices:
            centers.append((i * u[0] + j * v[0], i * u[1] + j * v[1]))
        return np.array(centers)

    def objective(params):
        ux, uy, vx, vy, phi = params
        u = np.array([ux, uy])
        v = np.array([vx, vy])
        
        centers = get_centers(u, v)
        centroid = np.mean(centers, axis=0)
        centers_rel = centers - centroid
        
        # Check overlap: minimum distance between any two centers must be >= sqrt(3)
        # Calculate all pairwise distances
        diff = centers_rel[:, np.newaxis, :] - centers_rel[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < np.sqrt(3) - 1e-6:
            return 1000.0 + (np.sqrt(3) - min_dist)
            
        # Rotate the configuration by phi relative to the container
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        R = np.array([[c_phi, -s_phi], [s_phi, c_phi]])
        centers_rot = centers_rel @ R.T
        
        # Generate all 72 vertices (12 hexagons * 6 vertices)
        verts = centers_rot[:, np.newaxis, :] + hex_offsets[np.newaxis, :, :]
        verts = verts.reshape(-1, 2)
        
        # Calculate required container side length
        # Container is a regular hexagon centered at origin.
        # Condition: max projection of any vertex onto the 6 normal directions <= apothem
        # Apothem = S * sqrt(3) / 2
        # Normals are perpendicular to the sides.
        # For a hexagon with vertices at 0, 60, ..., the normals are at 30, 90, ...
        normal_angles = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 6, endpoint=False)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        projections = verts @ normals.T
        max_proj = np.max(np.abs(projections))
        
        S = max_proj * (2.0 / np.sqrt(3))
        return S

    # Optimization bounds
    # Standard honeycomb: u=(sqrt(3), 0), v=(sqrt(3)/2, 1.5)
    # Allow some flexibility for shear and rotation
    bounds = [
        (1.5, 2.5),  # ux
        (-0.5, 0.5), # uy
        (0.5, 2.5),  # vx
        (0.5, 2.5),  # vy
        (0, np.pi/3) # phi
    ]
    
    # Initial guess
    x0 = [np.sqrt(3), 0.0, np.sqrt(3)/2, 1.5, 0.0]
    
    # Run differential evolution
    result = differential_evolution(objective, bounds, seed=42, maxiter=150, tol=1e-5, polish=True)
    
    best_params = result.x
    ux, uy, vx, vy, phi = best_params
    u = np.array([ux, uy])
    v = np.array([vx, vy])
    
    centers = get_centers(u, v)
    centroid = np.mean(centers, axis=0)
    centers_final = centers - centroid
    
    # Construct the return values
    inner = []
    for c in centers_final:
        # Angle is 0.0 as we kept hexagons aligned with global axes
        inner.append([float(c[0]), float(c[1]), 0.0])
        
    return inner, (0.0, 0.0), float(result.fun), 0.0
# EVOLVE-BLOCK-END
