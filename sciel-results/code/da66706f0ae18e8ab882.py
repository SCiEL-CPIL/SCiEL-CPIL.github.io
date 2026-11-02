"""Spectral graph optimization for circle packing in a unit square."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian
import scipy.sparse.linalg as spla
import scipy.optimize

N = 50


def construct():
    """
    Construct centres of N=50 equal non-overlapping circles in [-0.5, 0.5]^2
    maximizing the common radius via spectral graph optimization.
    
    Approach:
    1. Start with a good initial configuration (slightly perturbed grid).
    2. Iteratively:
       a. Build a k-NN graph.
       b. Compute repulsive forces based on the graph (vectorized).
       c. Compute boundary repulsive forces.
       d. Update centers.
    3. Refine using scipy.optimize.
    """
    np.random.seed(42)
    
    # Initial configuration: grid with small random perturbation
    m = int(np.ceil(np.sqrt(N)))  # 8x8 grid, use first 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small perturbation to break symmetry
    centers += np.random.randn(N, 2) * 0.01
    
    # Ensure within bounds
    centers = np.clip(centers, -0.49, 0.49)
    
    k = 8  # k-nearest neighbors
    n_iter = 1000
    learning_rate = 0.005
    epsilon = 1e-4 # Softening parameter
    
    def compute_forces(centers, k, epsilon):
        n = len(centers)
        # Compute pairwise differences and squared distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # Mask self-interactions
        np.fill_diagonal(dist_sq, np.inf)
        
        # Find k nearest neighbors for each point
        # Get indices of k smallest distances
        knn_indices = np.argpartition(dist_sq, k, axis=1)[:, :k]
        
        # Create a mask for the k-NN graph
        mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            mask[i, knn_indices[i]] = True
        
        # Compute forces: sum over j in NN(i) of (x_i - x_j) / (dist^2 + epsilon)
        # We only want to sum over j where mask[i,j] is True
        # inv_dist_sq = 1 / (dist_sq + epsilon)
        # But we must avoid division by zero for non-NN? 
        # We can just set the force contribution to 0 for non-NN.
        
        # To do this efficiently:
        # force = np.zeros_like(centers)
        # for i in range(n):
        #    js = np.where(mask[i])[0]
        #    d2 = dist_sq[i, js]
        #    f = diff[i, js] / (d2 + epsilon)
        #    force[i] = np.sum(f, axis=0)
        
        # Vectorized version using masking:
        # Create a weighted diff matrix
        # We want W[i,j] = 1/(dist_sq[i,j] + epsilon) if mask[i,j] else 0
        # Then force[i] = sum_j W[i,j] * diff[i,j]
        
        weights = np.zeros_like(dist_sq)
        weights[mask] = 1.0 / (dist_sq[mask] + epsilon)
        
        # force[i] = sum_j weights[i,j] * diff[i,j]
        # diff is (N, N, 2)
        # weights is (N, N)
        # We can use einsum or broadcasting
        force = np.einsum('ij,ijk->ik', weights, diff)
        
        return force

    def compute_boundary_forces(centers, r_min):
        # Force to push centers away from boundaries
        # If center is within r_min of boundary, push it inward
        force = np.zeros_like(centers)
        
        # Left: x < -0.5 + r_min
        # Right: x > 0.5 - r_min
        # Bottom: y < -0.5 + r_min
        # Top: y > 0.5 - r_min
        
        margin = r_min * 1.1 # Slightly larger than r_min to allow expansion
        
        # Left
        close_left = centers[:, 0] < -0.5 + margin
        force[close_left, 0] += (-0.5 + margin - centers[close_left, 0])
        
        # Right
        close_right = centers[:, 0] > 0.5 - margin
        force[close_right, 0] -= (centers[close_right, 0] - (0.5 - margin))
        
        # Bottom
        close_bottom = centers[:, 1] < -0.5 + margin
        force[close_bottom, 1] += (-0.5 + margin - centers[close_bottom, 1])
        
        # Top
        close_top = centers[:, 1] > 0.5 - margin
        force[close_top, 1] -= (centers[close_top, 1] - (0.5 - margin))
        
        return force

    def calc_radius(centers):
        # Calculate current minimum radius
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist = np.sqrt(np.min(dist_sq))
        
        # Distance to boundary
        min_boundary = np.min(np.min(0.5 - np.abs(centers), axis=1), axis=0)
        
        return min(min_dist / 2, min_boundary)

    # Main optimization loop
    for it in range(n_iter):
        r_min = calc_radius(centers)
        
        # Build forces
        repulsive_force = compute_forces(centers, k, epsilon)
        boundary_force = compute_boundary_forces(centers, r_min)
        
        # Combine forces
        # Scale repulsive force
        force = 0.1 * repulsive_force + 0.5 * boundary_force
        
        # Update centers
        centers += learning_rate * force
        
        # Clip to valid region (allow up to 0.5)
        centers = np.clip(centers, -0.5, 0.5)
        
        # Adaptive learning rate decay
        learning_rate *= 0.999
    
    # Final refinement with scipy.optimize
    # Objective: maximize radius = min(min_dist/2, min_boundary_dist)
    # Minimize -radius
    
    def objective(flat_centers):
        c = flat_centers.reshape(N, 2)
        # Penalty for out of bounds
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 1e6
        r = calc_radius(c)
        return -r

    # Initial guess
    x0 = centers.flatten()
    
    # Use L-BFGS-B
    try:
        res = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', 
                                      options={'maxiter': 500, 'ftol': 1e-10})
        if res.success:
            centers = res.x.reshape(N, 2)
    except Exception:
        pass

    # Ensure valid output
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
