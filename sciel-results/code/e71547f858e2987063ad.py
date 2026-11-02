"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np

def compute_pairwise_sq_dists(pts):
    """Compute all pairwise squared distances."""
    # (n, 1, d) - (1, n, d) -> (n, n, d)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    return sq_dists

def get_score(pts):
    """Calculate the squared ratio of max to min pairwise distance."""
    sq_dists = compute_pairwise_sq_dists(pts)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    valid_dists = sq_dists[mask]
    if np.any(valid_dists < 1e-10):
        return 1e10
    min_d = np.min(valid_dists)
    max_d = np.max(valid_dists)
    if min_d == 0:
        return 1e10
    return (max_d / min_d)

def compute_gradient(pts):
    """
    Compute the gradient of the objective function f = M/m.
    M = max pairwise sq dist, m = min pairwise sq dist.
    """
    sq_dists = compute_pairwise_sq_dists(pts)
    mask = ~np.eye(N, dtype=bool)
    
    # Find max and min values
    max_val = np.max(sq_dists[mask])
    min_val = np.min(sq_dists[mask])
    
    if min_val < 1e-10:
        return np.zeros_like(pts)
    
    # Find indices of max and min distances
    # We need to handle ties by averaging gradients (subgradient)
    # For simplicity and efficiency, we pick one if unique, or average if tied within tolerance
    # Let's just pick the first occurrence for now, or better, identify all close to max/min
    
    # Get flat indices
    flat_sq_dists = sq_dists[mask]
    
    # Threshold for "tied"
    eps = 1e-6
    
    # Max indices
    max_mask = (sq_dists > max_val - eps) & mask
    # Min indices
    min_mask = (sq_dists < min_val + eps) & mask
    
    # Gradient of M (max)
    # Sum of gradients of all max pairs, normalized by count? 
    # Actually, the subgradient of max is the convex hull of gradients of active constraints.
    # A standard choice is the average of the gradients of the active constraints.
    
    grad_M = np.zeros_like(pts)
    count_M = np.sum(max_mask)
    if count_M > 0:
        # For each point k, sum gradients from all max pairs involving k
        # d/dp_k ||p_i - p_j||^2 = 2(p_k - p_j) if k=i, 2(p_k - p_i) if k=j, 0 otherwise
        # We can compute this efficiently.
        # Let's just loop over the active pairs since N=14 is small
        for i in range(N):
            for j in range(i + 1, N):
                if max_mask[i, j]:
                    diff = pts[i] - pts[j]
                    grad_M[i] += 2 * diff
                    grad_M[j] -= 2 * diff
        grad_M /= count_M # Average subgradient
    
    # Gradient of m (min)
    grad_m = np.zeros_like(pts)
    count_m = np.sum(min_mask)
    if count_m > 0:
        for i in range(N):
            for j in range(i + 1, N):
                if min_mask[i, j]:
                    diff = pts[i] - pts[j]
                    grad_m[i] += 2 * diff
                    grad_m[j] -= 2 * diff
        grad_m /= count_m # Average subgradient
    
    # f = M / m
    # df = (1/m) dM - (M/m^2) dm
    grad_f = (1.0 / min_val) * grad_M - (max_val / (min_val ** 2)) * grad_m
    
    return grad_f

def gradient_descent(initial_pts, max_iter=20000, lr=0.01, momentum=0.9, rng=None):
    """
    Run Gradient Descent with Momentum.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    pts = initial_pts.copy()
    velocity = np.zeros_like(pts)
    
    # Adaptive learning rate
    current_lr = lr
    best_score = get_score(pts)
    best_pts = pts.copy()
    
    no_improve_count = 0
    
    for i in range(max_iter):
        grad = compute_gradient(pts)
        
        # Update velocity
        velocity = momentum * velocity - current_lr * grad
        
        # Update positions
        new_pts = pts + velocity
        
        new_score = get_score(new_pts)
        
        if new_score < best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Simple adaptive learning rate
        if no_improve_count > 100:
            current_lr *= 0.9
            no_improve_count = 0
            if current_lr < 1e-8:
                break
        
        pts = new_pts
        
    return best_pts, best_score

def construct():
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = float('inf')
    
    # Multiple restarts
    n_restarts = 10
    
    for r in range(n_restarts):
        # Random initialization on a sphere or Gaussian
        # Using random points on a sphere helps avoid points collapsing to center
        # Let's try a mix
        if r % 2 == 0:
            # Uniform on sphere
            pts = rng.normal(size=(N, D))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms * np.sqrt(N) # Scale appropriately
        else:
            # Gaussian
            pts = rng.normal(0, 1, size=(N, D))
            
        # Run GD
        opt_pts, score = gradient_descent(pts, max_iter=5000, lr=0.05, momentum=0.9)
        
        if score < best_score:
            best_score = score
            best_pts = opt_pts
            
    return best_pts

# EVOLVE-BLOCK-END
