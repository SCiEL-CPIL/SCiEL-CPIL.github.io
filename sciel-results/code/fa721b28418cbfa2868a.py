# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses multi-start BFGS optimization to maximize the minimum pairwise squared distance
    subject to the maximum pairwise squared distance being fixed to 1.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    # Precompute pair indices for efficiency
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    n_pairs = len(pair_indices)
    
    def objective_and_grad(x):
        """
        x is flattened array of shape (n*d,)
        Returns objective value and gradient
        """
        points = x.reshape(n, d)
        
        # Compute all pairwise squared distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        
        # Extract pairwise distances (upper triangle)
        i_idx, j_idx = np.triu_indices(n, k=1)
        pair_sq_dists = sq_dists[i_idx, j_idx]
        
        # Find min and max
        min_sq_dist = np.min(pair_sq_dists)
        max_sq_dist = np.max(pair_sq_dists)
        
        # Use a smooth penalty for max constraint
        # We want max_sq_dist <= 1, use quadratic penalty
        if max_sq_dist > 1.0:
            penalty = 1e4 * (max_sq_dist - 1.0) ** 2
        else:
            penalty = 0.0
        
        # Objective: -min_sq_dist + penalty
        obj = -min_sq_dist + penalty
        
        # Gradient computation
        # d(-min_sq_dist)/dx: only the pair(s) achieving min contribute
        min_idx = np.argmin(pair_sq_dists)
        min_i, min_j = pair_indices[min_idx]
        
        grad = np.zeros_like(x)
        diff_min = points[min_i] - points[min_j]
        grad[min_i * d:(min_i + 1) * d] -= 2.0 * diff_min
        grad[min_j * d:(min_j + 1) * d] += 2.0 * diff_min
        
        # Gradient of penalty term
        if max_sq_dist > 1.0:
            max_idx = np.argmax(pair_sq_dists)
            max_i, max_j = pair_indices[max_idx]
            diff_max = points[max_i] - points[max_j]
            penalty_grad_factor = 2.0 * 1e4 * (max_sq_dist - 1.0)
            grad[max_i * d:(max_i + 1) * d] += penalty_grad_factor * diff_max
            grad[max_j * d:(max_j + 1) * d] -= penalty_grad_factor * diff_max
        
        return obj, grad
    
    def compute_score(points):
        """Compute the metric: (d_min / d_max)^2"""
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        return (d_min / d_max) ** 2
    
    # Generate multiple initial guesses
    np.random.seed(42)
    
    # Start 1: Random
    x0_list = [np.random.randn(n * d) * 0.5]
    
    # Start 2: Points on a sphere
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    phi = np.linspace(0, np.pi, n)
    sphere_points = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    x0_list.append(sphere_points.flatten())
    
    # Start 3: Icosahedron-like (12 vertices) + 2 more
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Add 2 more points: center and one offset
    extra = np.array([[0, 0, 0], [0.5, 0, 0]])
    ico_plus = np.vstack([ico_verts, extra])
    x0_list.append(ico_plus.flatten())
    
    # Start 4: Another random
    np.random.seed(123)
    x0_list.append(np.random.randn(n * d) * 0.3)
    
    # Start 5: Cube-like configuration
    cube_verts = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
        [0, 0, -1], [0, 0, 1]
    ])
    cube_verts = cube_verts / np.linalg.norm(cube_verts, axis=1, keepdims=True)
    x0_list.append(cube_verts.flatten())
    
    best_points = None
    best_score = -1
    
    for start_idx, x0 in enumerate(x0_list):
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='BFGS',
                jac=True,
                options={'maxiter': 1000, 'gtol': 1e-10, 'disp': False}
            )
            
            points = result.x.reshape(n, d)
            
            # Check validity
            if not np.all(np.isfinite(points)):
                continue
            
            score = compute_score(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception as e:
            continue
    
    # If no valid result, fallback to random
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    # Final normalization: scale so that max pairwise distance is reasonable
    # The optimization should have handled this, but let's ensure validity
    if best_points is not None and np.all(np.isfinite(best_points)):
        # Center the points
        centroid = np.mean(best_points, axis=0)
        best_points = best_points - centroid
        
        # Scale if needed to ensure reasonable magnitude
        dists = pdist(best_points)
        if len(dists) > 0 and np.max(dists) > 0:
            # Keep as is, the ratio is scale-invariant
            pass
    
    return best_points


# EVOLVE-BLOCK-END
