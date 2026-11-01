import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12
DEADLINE = 145.0  # Safety margin before 150s kill

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_covering_radius(pts):
    """
    Compute the true covering radius using the convex hull.
    Returns the radius in radians.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        valid = norms.flatten() > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid] / norms[valid]
        
        # Ensure normals point outward
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1)
        normals = normals * np.sign(signs)[:, np.newaxis]
        
        # Circumradius is angular distance from pole to any vertex
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return np.max(radii)
        
    except Exception:
        return np.pi

def _soft_min_loss_and_grad(pts, temp=0.1):
    """
    Compute the soft-min of squared pairwise distances and its gradient.
    We want to maximize the minimum distance, so we minimize -soft_min(d^2).
    
    Let G be the Gram matrix of pts (N x N). G_ij = pts_i . pts_j.
    d_ij^2 = 2 - 2 G_ij.
    
    We want to maximize min_{i != j} d_ij^2.
    Equivalent to minimizing max_{i != j} G_ij (since d^2 decreases as G increases).
    
    Surrogate: L = temp * log(sum_{i != j} exp(G_ij / temp))
    Minimizing L encourages all G_ij to be small (points far apart).
    
    Gradient of L w.r.t. pts_k:
    dL/dG_ij = (1/sum_exp) * exp(G_ij/temp) * (1/temp)
    dG_ij/dpts_k = 0 if k != i,j
    dG_ii/dpts_k = 0 (we ignore i==j)
    dG_ij/dpts_i = pts_j
    dG_ij/dpts_j = pts_i
    
    So, grad_k = sum_{j != k} (w_kj * pts_j) + sum_{i != k} (w_ik * pts_i)
    where w_kj = exp(G_kj / temp) / sum_exp.
    Note that G is symmetric, so w_kj = w_jk.
    grad_k = 2 * sum_{j != k} w_kj * pts_j
    
    Wait, let's re-verify the sign.
    We want to MAXIMIZE min d^2.
    Minimize L = - soft_min(d^2) = - temp * log(sum exp(-d^2/temp))
    This is equivalent to minimizing temp * log(sum exp(G/temp)) because -d^2 = 2G - 2.
    exp((2G-2)/temp) = exp(-2/temp) * exp(2G/temp).
    So minimizing log(sum exp(G/temp)) is correct for maximizing min distance.
    
    Gradient direction:
    If we move pts_k in direction v, G_kj changes by v . pts_j.
    We want to decrease G_kj. So we should move pts_k in direction -pts_j.
    The gradient of the loss L w.r.t. pts_k is:
    dL/dpts_k = sum_{j != k} (dL/dG_kj) * (dG_kj/dpts_k)
    dL/dG_kj = (1/temp) * exp(G_kj/temp) / sum_exp = w_kj / temp
    dG_kj/dpts_k = pts_j
    So grad_k = (1/temp) * sum_{j != k} w_kj * pts_j
    
    We want to minimize L, so we move pts_k by -lr * grad_k.
    Move direction: - sum w_kj pts_j.
    This means moving away from the "weighted average" of neighbors.
    This is a repulsive force.
    """
    G = pts @ pts.T
    # Mask out diagonal
    mask = ~np.eye(N, dtype=bool)
    G_masked = np.where(mask, G, -np.inf)
    
    # Soft max of G (since we want to minimize max G)
    # L = temp * log(sum exp(G/temp))
    max_G = np.max(G_masked)
    # For numerical stability, subtract max
    exp_G = np.exp((G_masked - max_G) / temp)
    sum_exp = np.sum(exp_G)
    
    # Weights w_ij = exp((G_ij - max_G)/temp) / sum_exp
    # Note: This is proportional to exp(G_ij/temp)
    W = exp_G / sum_exp
    
    # Gradient for each point k:
    # grad_k = (1/temp) * sum_{j != k} W_kj * pts_j
    # grad = (1/temp) * (W @ pts)
    grad = (1.0 / temp) * (W @ pts)
    
    # Loss value (scaled for monitoring, not strictly needed for GD step direction)
    loss = temp * (np.log(sum_exp) + max_G / temp)
    
    return loss, grad

def _project_to_sphere(pts, grad, lr):
    """
    Perform a gradient descent step on the sphere.
    1. Compute Euclidean step: -lr * grad
    2. Project to tangent space? 
       Actually, for sphere constraints, a simple update followed by normalization
       is a valid Riemannian gradient step (retraction).
       pts_new = normalize(pts - lr * grad)
    """
    new_pts = pts - lr * grad
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < _EPS] = _EPS
    new_pts = new_pts / norms
    return new_pts

def _local_polish(pts, n_iters=100, temp_start=0.5, temp_end=0.05, lr_start=0.01, lr_end=0.001):
    """
    Optimize using Gradient Descent on the soft-min surrogate.
    Anneal temperature and learning rate.
    """
    best_pts = pts.copy()
    best_r = _compute_covering_radius(pts)
    
    for i in range(n_iters):
        # Anneal temperature: start high (smooth), end low (sharp max)
        progress = i / max(1, n_iters - 1)
        temp = temp_start * (temp_end / temp_start) ** progress
        lr = lr_start * (lr_end / lr_start) ** progress
        
        loss, grad = _soft_min_loss_and_grad(pts, temp)
        
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            break
            
        pts = _project_to_sphere(pts, grad, lr)
        
        # Check improvement periodically to save time
        if i % 5 == 0 or i == n_iters - 1:
            curr_r = _compute_covering_radius(pts)
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
                
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Multi-restart strategy
    # 1. Fibonacci start
    # 2. Random starts
    
    best_pts = _initial_fib()
    best_r = _compute_covering_radius(best_pts)
    
    n_restarts = 15
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
            
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = _random_init(rng)
            
        # Phase 1: Coarse optimization with higher temp
        # Phase 2: Fine optimization with lower temp
        
        # Time budget per restart
        phase_deadline = time.time() + 8.0
        
        # Coarse phase
        iters_coarse = 50
        while time.time() < phase_deadline and iters_coarse > 0:
            pts = _local_polish(pts, n_iters=min(10, iters_coarse), temp_start=0.5, temp_end=0.1, lr_start=0.02, lr_end=0.01)
            iters_coarse -= 10
            
        # Fine phase
        phase_deadline = time.time() + 5.0
        while time.time() < phase_deadline:
            pts = _local_polish(pts, n_iters=20, temp_start=0.1, temp_end=0.02, lr_start=0.005, lr_end=0.0005)
            
        curr_r = _compute_covering_radius(pts)
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
            
    # Final polish on the best configuration
    if time.time() < deadline - 5.0:
        phase_deadline = time.time() + 5.0
        while time.time() < phase_deadline:
            best_pts = _local_polish(best_pts, n_iters=10, temp_start=0.05, temp_end=0.01, lr_start=0.001, lr_end=0.0001)
            curr_r = _compute_covering_radius(best_pts)
            if curr_r < best_r:
                best_r = curr_r
            else:
                break
                
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
