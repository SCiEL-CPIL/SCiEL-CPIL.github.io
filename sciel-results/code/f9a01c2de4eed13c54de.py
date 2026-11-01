"""Packing 50 equal spheres in a unit cube [-0.5, 0.5]^3."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 50

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall constraint
    dists_to_wall = np.min(0.5 - np.abs(centers), axis=1)
    wall_radius = np.min(dists_to_wall)
    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    pair_radius = np.min(d) / 2.0
    return min(wall_radius, pair_radius)

def _softmin_radius_and_grad(centers, t=0.005):
    """
    Compute a smooth approximation of the minimum radius and its gradient.
    Uses soft-min over all constraints (pairwise and wall).
    Returns (radius_soft, grad) where grad has shape (N, 3).
    """
    centers = np.asarray(centers, dtype=np.float64)
    Npts = len(centers)
    
    # Collect all constraints: pairwise distances/2 and wall distances
    # For efficiency, compute all pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    
    # Wall distances
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)  # shape (N,)
    
    # We want soft-min of all constraints
    # Constraints: d[i,j]/2 for i<j, and wall_dists[i] for each i
    # To avoid double counting, only use i<j
    tri_idx = np.triu_indices(Npts, k=1)
    pair_vals = d[tri_idx] / 2.0  # shape (M,)
    
    # Combine all constraint values
    all_vals = np.concatenate([pair_vals, wall_dists])
    
    # Soft-min: -t * log(sum(exp(-x/t)))
    # Use log-sum-exp for numerical stability
    max_val = np.max(all_vals)
    exp_terms = np.exp(-(all_vals - max_val) / t)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = max_val - t * log_sum
    
    # Gradient with respect to centers
    # d(soft_min)/d(centers) = -sum_i (exp(-x_i/t) / sum_j(exp(-x_j/t))) * dx_i/d(centers)
    weights = exp_terms / np.sum(exp_terms)  # shape (M+N,)
    
    grad = np.zeros_like(centers)
    
    # Pairwise part: for each pair (i,j), contribution to grad[i] and grad[j]
    # x = d[i,j]/2, dx/d(centers[i]) = (centers[i]-centers[j])/d[i,j], dx/d(centers[j]) = -(centers[i]-centers[j])/d[i,j]
    # So d(soft_min)/d(centers[i]) += -w * (centers[i]-centers[j])/d[i,j]
    # But we want to maximize, so we use -grad for descent, or equivalently compute grad of -soft_min
    # Actually let's compute grad of soft_min, then we'll negate it for maximization
    
    for idx in range(len(pair_vals)):
        i, j = tri_idx[0][idx], tri_idx[1][idx]
        w = weights[idx]
        diff = centers[i] - centers[j]
        dist = d[i, j]
        if dist < 1e-12:
            continue
        # d(x)/d(centers[i]) = diff/dist, d(x)/d(centers[j]) = -diff/dist
        grad[i] += -w * diff / dist
        grad[j] += w * diff / dist
    
    # Wall part: x = min_k(0.5 - |c_k|) for center i
    # The min is over the 6 faces. For soft-min over faces too, but we already took min over faces.
    # Actually, wall_dists[i] = min(0.5-|cx|, 0.5-|cy|, 0.5-|cz|)
    # The gradient of min is the gradient of the active constraint.
    # For a smooth approximation, we need to handle this more carefully.
    # Let's use a different approach: for each center, the wall distance is min over 6 values.
    # We can approximate the gradient by using the direction toward the closest wall.
    
    for i in range(Npts):
        w = weights[len(pair_vals) + i]
        # Find which wall is closest
        cx, cy, cz = centers[i]
        # Six wall distances
        wall_vals = np.array([0.5 - cx, cx + 0.5, 0.5 - cy, cy + 0.5, 0.5 - cz, cz + 0.5])
        min_idx = np.argmin(wall_vals)
        # Gradient of the active wall constraint
        # If 0.5-cx is min: d/dcx = -1
        # If cx+0.5 is min: d/dcx = +1
        # etc.
        g = np.zeros(3)
        if min_idx == 0: g[0] = -1.0
        elif min_idx == 1: g[0] = 1.0
        elif min_idx == 2: g[1] = -1.0
        elif min_idx == 3: g[1] = 1.0
        elif min_idx == 4: g[2] = -1.0
        elif min_idx == 5: g[2] = 1.0
        grad[i] += -w * g
    
    return soft_min_val, grad

def _lloyd_step(centers, alpha=0.1):
    """
    Lloyd's algorithm step: move each center toward the centroid of its Voronoi cell.
    This naturally equalizes clearance.
    """
    from scipy.spatial import Voronoi
    
    # Add boundary points to ensure all Voronoi cells are bounded
    # Use a larger bounding box
    margin = 0.1
    boundary_pts = []
    for x in [-0.5 - margin, 0.5 + margin]:
        for y in [-0.5 - margin, 0.5 + margin]:
            for z in [-0.5 - margin, 0.5 + margin]:
                boundary_pts.append([x, y, z])
    # Add edge midpoints and face centers for better Voronoi
    for axis in range(3):
        for val in [-0.5 - margin, 0.5 + margin]:
            for other1 in [-0.5 - margin, 0.0, 0.5 + margin]:
                for other2 in [-0.5 - margin, 0.0, 0.5 + margin]:
                    pt = [0.0, 0.0, 0.0]
                    pt[axis] = val
                    pt[(axis+1)%3] = other1
                    pt[(axis+2)%3] = other2
                    if pt not in boundary_pts:
                        boundary_pts.append(pt)
    
    all_pts = np.vstack([centers, np.array(boundary_pts)])
    
    try:
        vor = Voronoi(all_pts)
    except Exception:
        # Fallback: just return centers unchanged
        return centers.copy()
    
    new_centers = centers.copy()
    for i in range(len(centers)):
        region = vor.regions[vor.point_region[i]]
        if -1 in region or len(region) == 0:
            continue
        vertices = vor.vertices[region]
        if len(vertices) == 0:
            continue
        centroid = np.mean(vertices, axis=0)
        # Clamp centroid to cube
        centroid = np.clip(centroid, -0.499, 0.499)
        new_centers[i] = (1 - alpha) * centers[i] + alpha * centroid
    
    return new_centers

def _gradient_descent(centers, max_iters=200, deadline=None):
    """
    Gradient descent on the soft-min radius to maximize it.
    """
    centers = centers.copy()
    t = 0.005
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        radius, grad = _softmin_radius_and_grad(centers, t)
        
        # Take a step in the direction of increasing radius
        # Use adaptive step size
        step_size = 0.01 / (1 + it * 0.01)
        centers_new = centers + step_size * grad
        
        # Clamp to cube
        centers_new = np.clip(centers_new, -0.4999, 0.4999)
        
        # Check if this improves
        radius_new, _ = _softmin_radius_and_grad(centers_new, t)
        if radius_new > radius - 1e-10:
            centers = centers_new
        else:
            # Try smaller step
            centers_new = centers + 0.5 * step_size * grad
            centers_new = np.clip(centers_new, -0.4999, 0.4999)
            radius_new, _ = _softmin_radius_and_grad(centers_new, t)
            if radius_new > radius - 1e-10:
                centers = centers_new
        
        # Decrease temperature
        t *= 0.99
    
    return centers

def _local_search(centers, max_iters=500, deadline=None):
    """
    Local search using a combination of gradient descent and direct perturbation.
    """
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try random perturbations
        for _ in range(10):
            idx = np.random.randint(len(centers))
            delta = np.random.randn(3) * 0.01
            centers_new = centers.copy()
            centers_new[idx] += delta
            centers_new = np.clip(centers_new, -0.4999, 0.4999)
            
            r_new = _compute_radius(centers_new)
            if r_new > best_radius:
                centers = centers_new
                best_radius = r_new
                break
        else:
            # No improvement, try gradient step
            radius, grad = _softmin_radius_and_grad(centers, t=0.003)
            step_size = 0.005
            centers_new = centers + step_size * grad
            centers_new = np.clip(centers_new, -0.4999, 0.4999)
            r_new = _compute_radius(centers_new)
            if r_new > best_radius:
                centers = centers_new
                best_radius = r_new
    
    return centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_centers = None
    best_radius = 0.0
    
    np.random.seed(42)
    
    # Strategy 1: Lloyd's algorithm with multiple restarts
    for restart in range(5):
        if time.time() > deadline - 30.0:
            break
        
        # Initialize with random points
        centers = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        # Run Lloyd's algorithm
        for it in range(100):
            if time.time() > deadline - 25.0:
                break
            centers = _lloyd_step(centers, alpha=0.1)
            centers = np.clip(centers, -0.4999, 0.4999)
        
        # Polish with gradient descent
        centers = _gradient_descent(centers, max_iters=100, deadline=deadline - 20.0)
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Local search
        centers, radius = _local_search(centers, max_iters=200, deadline=deadline - 15.0)
        centers = np.clip(centers, -0.4999, 0.4999)
        radius = _compute_radius(centers)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Strategy 2: Gradient descent from structured initializations
    for restart in range(3):
        if time.time() > deadline - 15.0:
            break
        
        # Structured initialization: grid-like
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        
        if len(grid) >= N:
            centers = grid[:N].copy()
            # Add small random perturbation
            centers += np.random.randn(N, 3) * 0.01
        else:
            centers = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        # Gradient descent
        centers = _gradient_descent(centers, max_iters=150, deadline=deadline - 10.0)
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Local search
        centers, radius = _local_search(centers, max_iters=300, deadline=deadline - 5.0)
        centers = np.clip(centers, -0.4999, 0.4999)
        radius = _compute_radius(centers)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Final polish on the best configuration
    if best_centers is not None and time.time() < deadline - 2.0:
        centers = best_centers.copy()
        centers = _gradient_descent(centers, max_iters=100, deadline=deadline - 1.0)
        centers = np.clip(centers, -0.4999, 0.4999)
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
