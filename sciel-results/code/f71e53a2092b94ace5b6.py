"""Optimized 130-point sphere covering using gradient-based repulsion."""
import numpy as np
import time

N = 130

def _soft_max(values, beta=50.0):
    """Soft approximation of max using log-sum-exp."""
    m = np.max(values)
    return m + np.log(np.sum(np.exp(beta * (values - m)))) / beta

def _covering_radius(pts):
    """Compute covering radius using convex hull circumradii (approximate via Delaunay)."""
    # For speed, use a grid-based approximation
    # Sample points on sphere and find max distance to nearest point
    # But this is expensive, so we'll use the repulsion energy as proxy
    
    # Actually, let's compute the actual covering radius by checking
    # the Voronoi cells (points farthest from any site)
    # These are the circumcenters of Delaunay triangles
    
    # For efficiency, use a dense grid
    n_theta = 60
    n_phi = 120
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian
    X = np.sin(T) * np.cos(P)
    Y = np.sin(T) * np.sin(P)
    Z = np.cos(T)
    grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # Compute distances from each grid point to all sites
    # This is O(n_grid * N), which is manageable
    # Use chunked computation for memory
    chunk_size = 5000
    max_dist = 0.0
    for start in range(0, len(grid), chunk_size):
        chunk = grid[start:start+chunk_size]
        # Compute angular distances: arccos(dot)
        dots = chunk @ pts.T  # (chunk_size, N)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)  # (chunk_size, N)
        min_dists = np.min(dists, axis=1)
        max_dist = max(max_dist, np.max(min_dists))
    return max_dist

def _repulsion_energy(pts, k=8.0):
    """Compute a repulsion-based energy that correlates with covering radius.
    Uses inverse power law repulsion between all pairs."""
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)  # Avoid division by zero
    
    # Energy: sum of 1/d^k for all pairs
    # This pushes points apart
    energy = np.sum(1.0 / (dists**k))
    return energy

def _gradient_repulsion(pts, k=8.0):
    """Compute gradient of repulsion energy w.r.t. point positions."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    
    # Gradient of sum 1/d^k = -k * sum (1/d^(k+2)) * diff/d
    # For point i: grad_i = -k * sum_j (1/d_ij^(k+2)) * (p_i - p_j) / d_ij
    #             = -k * sum_j (1/d_ij^(k+1)) * (p_i - p_j) / d_ij
    # Wait, let me be more careful.
    # E = sum_{i<j} 1/d_ij^k
    # dE/dp_i = sum_{j!=i} d/dp_i (1/d_ij^k)
    # d_ij = |p_i - p_j|
    # d/dp_i (1/d_ij^k) = -k * 1/d_ij^(k+2) * (p_i - p_j)
    
    # So grad_i = -k * sum_{j!=i} (p_i - p_j) / d_ij^(k+2)
    
    inv_d_pow = 1.0 / (dists**(k+2))
    np.fill_diagonal(inv_d_pow, 0.0)
    
    # grad_i = -k * sum_j inv_d_pow[i,j] * (p_i - p_j)
    # = -k * (sum_j inv_d_pow[i,j] * p_i - sum_j inv_d_pow[i,j] * p_j)
    # = -k * (p_i * sum_j inv_d_pow[i,j] - inv_d_pow[i,:] @ p_j)
    
    row_sums = np.sum(inv_d_pow, axis=1)
    grad = -k * (pts * row_sums[:, np.newaxis] - inv_d_pow @ pts)
    
    # Project gradient onto tangent plane of sphere
    # For point p, tangent projection: grad - (grad . p) * p
    dot = np.sum(grad * pts, axis=1)
    grad = grad - dot[:, np.newaxis] * pts
    
    return grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.sqrt(np.sum(pts**2, axis=1))
    norms = np.maximum(norms, 1e-10)
    return pts / norms[:, np.newaxis]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize using gradient descent with adaptive step size
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    k = 8.0  # Repulsion exponent
    lr = 0.01
    patience = 0
    max_patience = 50
    
    while time.time() < deadline:
        # Compute gradient
        grad = _gradient_repulsion(pts, k)
        
        # Take gradient step
        new_pts = pts + lr * grad
        new_pts = _project_to_sphere(new_pts)
        
        # Check if we improved
        new_radius = _covering_radius(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
            patience = 0
            # Slightly increase learning rate if improving
            lr = min(lr * 1.05, 0.05)
        else:
            patience += 1
            # Decrease learning rate
            lr *= 0.9
            if patience > max_patience:
                # Try a random restart with perturbation
                if time.time() > deadline - 20:
                    break
                noise = 0.01 * np.random.randn(N, 3)
                pts = _project_to_sphere(pts + noise)
                patience = 0
                lr = 0.01
        
        # Periodically try different k values
        if patience % 20 == 0 and patience > 0:
            k = np.random.choice([6.0, 8.0, 10.0, 12.0])
    
    # Final polish: use smaller steps for a few iterations
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        lr = 0.001
        for _ in range(min(100, int(remaining_time / 0.05))):
            if time.time() > deadline - 2:
                break
            grad = _gradient_repulsion(pts, k)
            new_pts = _project_to_sphere(pts + lr * grad)
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = new_pts
                pts = new_pts
    
    return _project_to_sphere(best_pts)
