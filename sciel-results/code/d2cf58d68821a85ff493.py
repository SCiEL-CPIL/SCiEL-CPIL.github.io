"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import cKDTree
import time

N = 200

# EVOLVE-BLOCK-START

def _score(pts):
    """Compute the packing radius for a set of points in [-0.5, 0.5]^2."""
    # Boundary distances
    dists = np.minimum(
        np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1])),
        0.5
    )
    min_boundary = np.min(dists)
    
    # Pairwise distances
    tree = cKDTree(pts)
    # Get all pairs within a reasonable distance to find min
    # For efficiency, just compute min of all pairwise distances
    # For N=200, this is 19900 pairs, manageable
    min_pair = np.inf
    for i in range(N):
        d, _ = tree.query(pts[i], k=N)
        # d[0] is distance to self (0), so start from index 1
        if len(d) > 1:
            min_pair = min(min_pair, np.min(d[1:]))
    
    return min(min_boundary, min_pair / 2.0)


def _gradient_ascent(pts, max_iters=5000, lr0=0.001, lr_decay=0.999, tol=1e-10, time_limit=None):
    """
    Gradient ascent on a smooth surrogate of the packing radius.
    Uses soft-min to approximate the min of pairwise and boundary distances.
    """
    pts = pts.copy()
    N = len(pts)
    
    # Precompute which pairs are "close" for efficiency
    # We'll use a dynamic approach: track nearest neighbors
    
    best_radius = _score(pts)
    best_pts = pts.copy()
    
    lr = lr0
    prev_obj = -np.inf
    
    for it in range(max_iters):
        if time_limit is not None and (time.time() - time_limit[0]) > time_limit[1]:
            break
        
        # Compute distances to boundaries
        boundary_dists = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
        
        # Compute pairwise distances - we only need the smallest few
        # For gradient, we need contributions from all pairs, but weighted by soft-min
        # Use a temperature parameter for soft-min
        beta = 50.0  # Higher = closer to true min
        
        # Pairwise distances
        tree = cKDTree(pts)
        # Query each point for all others
        # For N=200, this is fine
        dists_matrix = np.zeros((N, N))
        for i in range(N):
            d, _ = tree.query(pts[i], k=N)
            dists_matrix[i] = d
        
        # Set diagonal to inf so it doesn't affect soft-min
        np.fill_diagonal(dists_matrix, np.inf)
        
        # Soft-min of pairwise distances (divided by 2 for radius)
        # We want to maximize the minimum of {boundary_dists, pairwise_dists/2}
        # Combine all constraints: boundary_dists (N values) and pairwise_dists/2 (N*(N-1)/2 values)
        
        # For efficiency, compute gradient contributions
        # Soft-min: f(x) = -1/beta * log(sum(exp(-beta * x_i)))
        # grad f_i = exp(-beta * x_i) / sum(exp(-beta * x_j))
        
        # Scale pairwise distances by 0.5
        pair_dists_scaled = dists_matrix / 2.0
        
        # Combine all "radius" values
        # Boundary: N values
        # Pairs: N*(N-1)/2 values, but we have full matrix, so N*(N-1) values (each pair counted twice)
        # To avoid double counting, we can just use the matrix as is and adjust weights
        
        # Actually, let's just compute the objective and gradient numerically-ish
        # for stability. For N=200, we can afford to compute gradients analytically.
        
        # All radius contributions
        # Boundary contributions: for point i, dist to nearest boundary
        # Pair contributions: for pair (i,j), dist/2
        
        # Let's use a simpler approach: compute the actual min and its gradient
        # The min is either a boundary distance or a pairwise distance/2
        
        # Find the actual minimum
        min_boundary_idx = np.argmin(boundary_dists)
        min_boundary_val = boundary_dists[min_boundary_idx]
        
        min_pair_idx = np.unravel_index(np.argmin(pair_dists_scaled), (N, N))
        # Avoid diagonal
        if min_pair_idx[0] == min_pair_idx[1]:
            # Find second smallest
            flat = pair_dists_scaled.flatten()
            flat[np.diag_indices(N)] = np.inf
            min_pair_flat_idx = np.argmin(flat)
            min_pair_idx = np.unravel_index(min_pair_flat_idx, (N, N))
        min_pair_val = pair_dists_scaled[min_pair_idx]
        
        current_min = min(min_boundary_val, min_pair_val)
        
        # Compute gradient of the soft-min
        # We'll use a weighted combination based on how close each constraint is to the min
        # This is a smooth approximation
        
        # Boundary gradient
        grad = np.zeros_like(pts)
        
        # For boundary constraints
        # d/dx_i (0.5 - |x_i|) = -sign(x_i) if x_i != 0
        # d/dy_i (0.5 - |y_i|) = -sign(y_i) if y_i != 0
        
        # Weight by soft-min probability
        # For each boundary constraint, weight = exp(-beta * (boundary_dist - current_min))
        # This is expensive to compute for all, so let's just use the active ones
        
        # Simplified: move points away from boundaries and from each other
        # based on the gradient of the min
        
        # Boundary forces
        for i in range(N):
            # Push away from left boundary if close
            if pts[i, 0] < 0:
                grad[i, 0] += 1.0  # push right
            elif pts[i, 0] > 0:
                grad[i, 0] -= 1.0  # push left
            
            if pts[i, 1] < 0:
                grad[i, 1] += 1.0  # push up
            elif pts[i, 1] > 0:
                grad[i, 1] -= 1.0  # push down
        
        # Pairwise forces: push apart
        # Only consider pairs that are relatively close
        threshold = current_min * 2.0 * 1.5  # Consider pairs within 1.5x the current min radius
        for i in range(N):
            for j in range(i + 1, N):
                d = dists_matrix[i, j]
                if d < threshold:
                    # Force magnitude increases as distance decreases
                    diff = pts[i] - pts[j]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-12:
                        # Gradient contribution: push i away from j and j away from i
                        force = 1.0 / (norm + 1e-6)
                        grad[i] += force * diff
                        grad[j] -= force * diff
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad /= grad_norm
        
        # Update
        pts_new = pts + lr * grad
        
        # Project back into the square
        pts_new[:, 0] = np.clip(pts_new[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
        pts_new[:, 1] = np.clip(pts_new[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Check if we improved
        new_radius = _score(pts_new)
        
        if new_radius > best_radius + tol:
            best_radius = new_radius
            best_pts = pts_new.copy()
            pts = pts_new
            # Increase learning rate slightly if improving
            lr = min(lr * 1.01, 0.01)
        else:
            # Decrease learning rate
            lr *= lr_decay
        
        # Convergence check
        if abs(new_radius - prev_obj) < tol and it > 100:
            break
        prev_obj = new_radius
    
    return best_pts, best_radius


def construct():
    """Construct 200 equal circles in the unit square with maximum radius."""
    start_time = time.time()
    time_limit = (start_time, 140.0)  # 140s out of 150s budget
    
    # Initial configurations: grid-based with perturbations
    best_pts = None
    best_radius = -np.inf
    
    # Create initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Multiple restarts with different initial conditions
    rng = np.random.RandomState(42)
    
    n_restarts = 10
    for restart in range(n_restarts):
        if (time.time() - start_time) > 135.0:
            break
        
        if restart == 0:
            # Start with grid
            pts = grid.copy()
        else:
            # Perturbed grid or random
            if restart % 2 == 0:
                # Small perturbation of grid
                pts = grid + rng.randn(N, 2) * 0.01
            else:
                # Random initialization
                pts = rng.uniform(-0.45, 0.45, (N, 2))
        
        # Project into square
        pts[:, 0] = np.clip(pts[:, 0], -0.49, 0.49)
        pts[:, 1] = np.clip(pts[:, 1], -0.49, 0.49)
        
        # Run gradient ascent
        result_pts, result_radius = _gradient_ascent(pts, max_iters=3000, lr0=0.005, lr_decay=0.998, time_limit=time_limit)
        
        if result_radius > best_radius:
            best_radius = result_radius
            best_pts = result_pts
    
    # Final polish: run more iterations on the best solution
    if best_pts is not None and (time.time() - start_time) < 145.0:
        polished_pts, polished_radius = _gradient_ascent(best_pts, max_iters=5000, lr0=0.001, lr_decay=0.999, time_limit=time_limit)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_pts = polished_pts
    
    return best_pts

# EVOLVE-BLOCK-END
