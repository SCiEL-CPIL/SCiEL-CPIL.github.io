# EVOLVE-BLOCK-START
import numpy as np

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    def compute_metric(points):
        # Calculate pairwise distances
        # Using scipy is allowed and faster
        from scipy.spatial.distance import pdist
        dists = pdist(points)
        if len(dists) == 0:
            return 0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0
        return (d_min / d_max) ** 2

    def repulsion_optimize(points, steps=5000, temp_start=1.0, temp_end=0.01):
        points = points.copy()
        current_metric = compute_metric(points)
        
        for i in range(steps):
            # Calculate forces
            # Force is repulsive: F = (r_i - r_j) / dist^3
            # We want to maximize min distance, so we push close points apart.
            # A simple approach: for each pair, if dist < threshold, push apart.
            # Or just sum of repulsive forces.
            
            forces = np.zeros_like(points)
            # Calculate all pairwise distances
            dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
            # Avoid division by zero
            dists[np.eye(n, dtype=bool)] = 1.0
            
            # Repulsive force magnitude 1/d^2, direction (r_i - r_j)
            # F_ij = (r_i - r_j) / d^3
            # Sum over j
            # This is O(N^2)
            for j in range(n):
                for k in range(n):
                    if j == k:
                        continue
                    dist = dists[j, k]
                    # Force on j from k
                    # We want to move j away from k
                    # Force vector = (r_j - r_k) / dist^3
                    # We use a strong repulsion to ensure separation
                    force = (points[j] - points[k]) / (dist ** 3)
                    forces[j] += force
            
            # Normalize forces to prevent explosion
            force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            force_norms[force_norms == 0] = 1.0
            forces = forces / force_norms
            
            # Update positions
            # Temperature cools down
            temp = temp_start * (temp_end / temp_start) ** (i / steps)
            points += forces * temp
            
            # Project back to unit sphere to maintain scale?
            # The metric is scale invariant, so we can normalize at the end.
            # However, keeping them on the sphere might help convergence.
            # Let's normalize to unit sphere at each step.
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            points = points / norms
            
            # Optional: Check metric and store best
            # This is expensive, so do it every 100 steps
            if i % 100 == 0:
                m = compute_metric(points)
                if m > current_metric:
                    current_metric = m
                    best_points = points.copy()
        
        return points, current_metric

    best_score = 0
    best_points = None
    
    # Initialization 1: Icosahedron + 2 poles
    phi = (1 + np.sqrt(5)) / 2
    N = np.sqrt(1 + phi**2)
    ico_verts = np.array([
        [0, 1/N, phi/N],
        [0, 1/N, -phi/N],
        [0, -1/N, phi/N],
        [0, -1/N, -phi/N],
        [1/N, phi/N, 0],
        [1/N, -phi/N, 0],
        [-1/N, phi/N, 0],
        [-1/N, -phi/N, 0],
        [phi/N, 0, 1/N],
        [phi/N, 0, -1/N],
        [-phi/N, 0, 1/N],
        [-phi/N, 0, -1/N]
    ])
    # Add 2 poles
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    init1 = np.vstack([ico_verts, poles])
    
    # Initialization 2: Random
    np.random.seed(42)
    init2 = np.random.randn(n, d)
    init2 /= np.linalg.norm(init2, axis=1, keepdims=True)
    
    # Initialization 3: Cube + Octahedron
    cube = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ])
    octa = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    init3 = np.vstack([cube, octa])
    init3 /= np.linalg.norm(init3, axis=1, keepdims=True)
    
    inits = [init1, init2, init3]
    
    for init in inits:
        # Run optimization
        pts, score = repulsion_optimize(init, steps=2000)
        if score > best_score:
            best_score = score
            best_points = pts
            
    # Final check and normalization
    if best_points is None:
        return init1
        
    # The metric is scale invariant, but we return the points as is.
    # Ensure no NaNs
    if np.any(np.isnan(best_points)) or np.any(np.isinf(best_points)):
        return init1
        
    return best_points

# EVOLVE-BLOCK-END
