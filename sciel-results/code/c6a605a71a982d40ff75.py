# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    best_points = None
    best_score = -1.0

    def compute_score(pts):
        """Calculate R = (d_min / d_max)^2"""
        if pts.shape != (16, 2):
            return -1.0
        dists = pdist(pts)
        if len(dists) == 0:
            return -1.0
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def objective(x):
        """Minimize negative sum of log distances to maximize spread."""
        pts = x.reshape(n, 2)
        # Add a small epsilon to avoid log(0)
        dists = pdist(pts)
        dists = np.maximum(dists, 1e-12)
        return -np.sum(np.log(dists))

    def gradient(x):
        """Gradient of the objective function."""
        pts = x.reshape(n, 2)
        # Compute pairwise distances
        dists = pdist(pts)
        # Reconstruct distance matrix
        dist_mat = np.zeros((n, n))
        iu = np.triu_indices(n, 1)
        dist_mat[iu] = dists
        dist_mat[iu[::-1]] = dists
        
        # Avoid division by zero
        dist_mat_safe = np.maximum(dist_mat, 1e-12)
        
        # Gradient of -sum(log(d_ij)) w.r.t x_k
        # d/dx_k sum(log(d_ij)) = sum_{i,j} (1/d_ij) * d(d_ij)/dx_k
        # d(d_ij)/dx_k = (x_k - x_i)/d_ij if k=i, (x_k - x_j)/d_ij if k=j, 0 otherwise
        # This is equivalent to -sum_{j != k} (x_k - x_j) / d_kj^2 * 2? 
        # Let's derive:
        # f = - sum_{i<j} log(||x_i - x_j||)
        # df/dx_k = - sum_{j != k} d/dx_k log(||x_k - x_j||)  (note: pairs involving k)
        # d/dx_k log(||x_k - x_j||) = (x_k - x_j) / ||x_k - x_j||^2
        # So df/dx_k = - sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
        
        grad = np.zeros_like(pts)
        # Efficient calculation
        # For each point k, sum over all j != k of (x_k - x_j) / d_kj^2
        # (x_k - x_j) / d_kj^2 is the unit vector from j to k divided by d_kj
        # Let v_kj = x_k - x_j
        # term = v_kj / ||v_kj||^2
        # Sum over j
        
        # Using broadcasting
        # diff: (n, n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # dist_sq: (n, n)
        dist_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0) # We will mask it out
        dist_sq_safe = np.maximum(dist_sq, 1e-12)
        
        # term[i, j] = (x_i - x_j) / dist_sq[i, j]
        # We want sum_j term[i, j] for each i
        # term shape (n, n, 2)
        # divide diff by dist_sq[..., np.newaxis]
        terms = diff / dist_sq_safe[..., np.newaxis]
        
        # Zero out diagonal
        np.fill_diagonal(terms[:, :, 0], 0.0)
        np.fill_diagonal(terms[:, :, 1], 0.0)
        
        # Sum over j (axis 1)
        grad = -np.sum(terms, axis=1)
        
        return grad.flatten()

    def normalize(pts):
        """Normalize points so that max distance is 1."""
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max > 1e-10:
            return pts / d_max
        return pts

    def run_optimization(initial_points):
        x0 = initial_points.flatten()
        # Use L-BFGS-B for speed and robustness
        res = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        opt_points = res.x.reshape(n, 2)
        # Normalize
        opt_points = normalize(opt_points)
        score = compute_score(opt_points)
        return opt_points, score

    # Generate initial configurations
    inits = []
    
    # 1. Grid 4x4
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
    grid_pts -= np.mean(grid_pts, axis=0)
    inits.append(grid_pts)

    # 2. Hexagonal packing
    # Approximate hex grid
    hex_pts = []
    # 4 rows, 4 cols roughly
    dx = 1.0
    dy = np.sqrt(3) * 0.5
    for i in range(4):
        for j in range(4):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts -= np.mean(hex_pts, axis=0)
    inits.append(hex_pts)

    # 3. Random points
    np.random.seed(42)
    for i in range(10):
        rand_pts = np.random.randn(n, 2)
        inits.append(rand_pts)

    # 4. Points on circles
    # 4 on inner circle, 12 on outer? Or 8 and 8?
    # Let's try 4 on r=0.5, 12 on r=1.5
    angles_inner = np.linspace(0, 2*np.pi, 4, endpoint=False)
    pts_inner = np.column_stack((0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)))
    angles_outer = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts_outer = np.column_stack((1.5 * np.cos(angles_outer), 1.5 * np.sin(angles_outer)))
    circ_pts = np.vstack((pts_inner, pts_outer))
    inits.append(circ_pts)
    
    # 5. 8 on inner, 8 on outer
    angles_inner2 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts_inner2 = np.column_stack((0.6 * np.cos(angles_inner2), 0.6 * np.sin(angles_inner2)))
    angles_outer2 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts_outer2 = np.column_stack((1.4 * np.cos(angles_outer2), 1.4 * np.sin(angles_outer2)))
    circ_pts2 = np.vstack((pts_inner2, pts_outer2))
    inits.append(circ_pts2)

    # Optimize each initial configuration
    for init in inits:
        if time.time() > deadline:
            break
        try:
            opt_pts, score = run_optimization(init)
            if score > best_score:
                best_score = score
                best_points = opt_pts
        except Exception:
            continue

    # If no valid point found (should not happen), return random
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, 2)
        
    return best_points

# EVOLVE-BLOCK-END
