# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses log-barrier augmented objective with BFGS optimization.
    Maximizes min pairwise squared distance subject to max pairwise squared distance < 1
    via a log-barrier penalty.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    n_pairs = len(i_idx)
    
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
    
    def objective_and_grad(x, mu=1.0):
        """
        x is flattened array of shape (n*d,)
        Objective: -min_sq_dist + mu * sum(-log(1 - sq_dist_i))
        The log barrier keeps all sq_dist_i < 1.
        """
        points = x.reshape(n, d)
        
        # Compute all pairwise squared distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        
        # Extract pairwise squared distances (upper triangle)
        pair_sq_dists = sq_dists[i_idx, j_idx]
        
        # Check if any distance is too close to 1
        if np.any(pair_sq_dists >= 0.999999):
            return 1e10, np.zeros_like(x)
        
        min_sq_dist = np.min(pair_sq_dists)
        
        # Log barrier term
        barrier = -np.sum(np.log(1.0 - pair_sq_dists))
        
        obj = -min_sq_dist + mu * barrier
        
        # Gradient
        grad = np.zeros_like(x)
        
        # Gradient of -min_sq_dist
        min_idx = np.argmin(pair_sq_dists)
        min_i, min_j = i_idx[min_idx], j_idx[min_idx]
        diff_min = points[min_i] - points[min_j]
        grad[min_i * d:(min_i + 1) * d] -= 2.0 * diff_min
        grad[min_j * d:(min_j + 1) * d] += 2.0 * diff_min
        
        # Gradient of mu * barrier (vectorized)
        inv_1_minus = mu * 2.0 / (1.0 - pair_sq_dists)  # shape (n_pairs,)
        
        # Build diff for all pairs: shape (n_pairs, d)
        pair_diffs = points[i_idx] - points[j_idx]  # shape (n_pairs, d)
        
        # Weighted contributions
        weighted_diffs = inv_1_minus[:, np.newaxis] * pair_diffs  # shape (n_pairs, d)
        
        # Accumulate using np.add.at
        grad_reshaped = grad.reshape(n, d)
        np.add.at(grad_reshaped, i_idx, weighted_diffs)
        np.add.at(grad_reshaped, j_idx, -weighted_diffs)
        
        return obj, grad
    
    # Generate multiple initial guesses
    best_points = None
    best_score = -1
    
    np.random.seed(42)
    
    # Multiple random starts
    for trial in range(50):
        x0 = np.random.randn(n * d) * 0.3
        
        # Also try structured starts
        if trial == 0:
            # Fibonacci sphere
            golden_ratio = (1 + np.sqrt(5)) / 2
            y = np.linspace(-1, 1, n)
            r = np.sqrt(np.maximum(0, 1 - y**2))
            theta = 2 * np.pi * np.arange(n) / golden_ratio
            sphere_pts = np.column_stack([np.cos(theta) * r, np.sin(theta) * r, y])
            x0 = sphere_pts.flatten()
        elif trial == 1:
            # Icosahedron + 2 extra
            phi = (1 + np.sqrt(5)) / 2
            ico_verts = np.array([
                [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
            ])
            ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
            extra = np.array([[0, 0, 0], [0.3, 0, 0]])
            ico_plus = np.vstack([ico_verts, extra])
            x0 = ico_plus.flatten()
        elif trial == 2:
            # Cube vertices + face centers
            cube_verts = np.array([
                [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
                [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
                [0, 0, -1], [0, 0, 1]
            ])
            cube_verts = cube_verts / np.linalg.norm(cube_verts, axis=1, keepdims=True)
            x0 = cube_verts.flatten()
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='BFGS',
                jac=True,
                options={'maxiter': 2000, 'gtol': 1e-12, 'disp': False}
            )
            
            points = result.x.reshape(n, d)
            
            # Check validity
            if not np.all(np.isfinite(points)):
                continue
            
            score = compute_score(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception:
            continue
    
    # If no valid result, fallback
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    # Center the points
    centroid = np.mean(best_points, axis=0)
    best_points = best_points - centroid
    
    return best_points


# EVOLVE-BLOCK-END
