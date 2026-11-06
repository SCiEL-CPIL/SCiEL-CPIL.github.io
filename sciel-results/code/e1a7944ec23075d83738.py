import numpy as np
import time
import sys

N = 130

def construct():
    """
    Constructs a set of N points on the unit sphere to minimize the covering radius.
    Strategy: Iterative repulsion-based optimization (Lloyd-like algorithm).
    1. Initialize with a Fibonacci spiral (good starting point).
    2. Iteratively move points to maximize the minimum distance to other points.
    3. Use a time budget to ensure termination.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s kill
    
    # 1. Initialization: Fibonacci spiral
    # This provides a reasonably uniform start, better than random.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure points are normalized
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Optimization Parameters
    # We use a "soft" repulsion potential to avoid singularities and allow smooth movement.
    # Potential V(r) = -1/r^2 or similar. Gradient points away from neighbors.
    # We will implement a simple gradient ascent on the "packing" objective:
    # Maximize sum_{i != j} 1 / ||p_i - p_j||^2  (or similar)
    # Actually, for covering radius minimization, we want to maximize the minimum distance to the *nearest* neighbor.
    # A common heuristic is to maximize the sum of inverse squared distances, which pushes points apart.
    
    step_size = 0.01
    # We will use a cooling schedule for step size
    
    # Pre-allocate for efficiency
    # We compute the gradient of the objective function.
    # Objective: Sum_{i} Sum_{j != i} 1 / (1 + ||p_i - p_j||^2)
    # This is a smooth proxy for "spread out".
    
    def compute_gradients(pts):
        """
        Computes the gradient of the repulsion potential for each point.
        Force on i from j is proportional to (p_i - p_j) / ||p_i - p_j||^4 * 2
        We want to move p_i in the direction of the force.
        """
        # pts is (N, 3)
        # diff[i, j] = p_i - p_j
        # This is O(N^2) memory and compute. N=130 is small (16900 pairs), so this is fast.
        
        # Calculate pairwise differences
        # p_i[:, np.newaxis] - p_j[np.newaxis, :]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :] # (N, N, 3)
        
        # Squared distances
        dist_sq = np.sum(diff**2, axis=2) # (N, N)
        
        # Avoid division by zero on the diagonal
        # We ignore self-interaction
        np.fill_diagonal(dist_sq, 1.0) # Prevent div by zero, but we'll mask later
        
        # Inverse squared distance (potential term)
        # We use a softened version to avoid huge gradients at close range
        # Let's use 1 / (dist_sq + epsilon)^2
        epsilon = 1e-6
        inv_dist_sq = 1.0 / (dist_sq + epsilon)**2
        
        # Mask diagonal
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # Gradient of 1/r^2 with respect to p_i is -2 * (p_i - p_j) / r^4
        # So force direction is (p_i - p_j) * 2 / r^4
        # We want to move p_i AWAY from p_j, so we add the vector (p_i - p_j) * weight
        # Weight is proportional to 1/r^4
        
        # grad[i] = sum_j 2 * (p_i - p_j) / (dist_sq[i,j] + eps)^2
        # Note: diff[i,j] is p_i - p_j
        # So we sum over j: diff[i, j] * inv_dist_sq[i, j] * 2
        
        grads = np.sum(diff * inv_dist_sq[:, :, np.newaxis] * 2.0, axis=1)
        
        # Project gradient onto the tangent plane of the sphere at each point
        # Tangent plane is orthogonal to p_i
        # grad_tangent = grad - (grad . p_i) * p_i
        dot = np.sum(grads * pts, axis=1, keepdims=True)
        grads_tangent = grads - dot * pts
        
        return grads_tangent

    # Main Optimization Loop
    current_score = 0.0
    best_points = points.copy()
    
    # We can estimate the covering radius roughly by the max of min-distances to neighbors
    # But calculating exact covering radius is expensive.
    # We will rely on the optimization metric to improve the configuration.
    
    # To monitor progress, we can compute the minimum distance to the nearest neighbor for each point
    # and take the maximum of those minima (this is a lower bound on covering radius, related to packing radius)
    # Actually, covering radius is about the *facets* of the convex hull.
    # A good proxy for improvement is maximizing the minimum pairwise distance (packing radius).
    
    def calc_packing_radius(pts):
        """Calculate the minimum distance between any pair of points (packing radius)."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dists = np.sqrt(np.min(dist_sq, axis=1))
        return np.min(min_dists)
    
    # Initial metric
    current_pack_radius = calc_packing_radius(points)
    best_pack_radius = current_pack_radius
    
    # Adaptive step size
    step = 0.05
    min_step = 1e-5
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        grads = compute_gradients(points)
        
        # Normalize gradients to ensure stable steps?
        # Or just scale by step size.
        # The magnitude of grads varies.
        
        # Update points
        new_points = points + step * grads
        
        # Renormalize to unit sphere
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        # Check if this improvement is valid (finite, etc.)
        if not np.all(np.isfinite(new_points)):
            break
            
        # Calculate new packing radius to check for improvement
        new_pack_radius = calc_packing_radius(new_points)
        
        if new_pack_radius > best_pack_radius:
            best_points = new_points
            best_pack_radius = new_pack_radius
            points = new_points
            # Slightly increase step if improving? Or keep constant.
            # Let's keep step constant but cool down slowly.
        else:
            # If no improvement, reduce step size to navigate local minima or plateau
            step *= cooling_rate
            if step < min_step:
                # Try a random kick to escape local minimum?
                # Simple: just stop if step is too small
                break
            points = new_points # Accept the move even if metric didn't improve globally? 
            # Actually, for gradient ascent on a smooth landscape, small steps usually help locally.
            # But if we want to guarantee best result, we should maybe backtrack?
            # Let's accept the move to keep exploring, but track best separately.
            
        # Cool down step size slightly every iteration regardless
        step *= 0.9995
        
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

# Keep the original function signature interface if required by the prompt structure
# The prompt says: "Keep the function names and signatures of the seed program."
# Seed program: def construct(): ... return array
# My construct() returns the array.

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
