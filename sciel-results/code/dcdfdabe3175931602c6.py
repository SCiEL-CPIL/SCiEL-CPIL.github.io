"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Constants
_DEADLINE = 145.0  # Safety margin before 150s kill
_N_SAMPLES = 50000  # Number of sample points for refinement

def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        norm = np.linalg.norm(pts)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return pts / norm
    else:
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

def _generate_random_samples(n_samples, rng):
    """Generate random points on the sphere."""
    pts = rng.normal(size=(n_samples, 3))
    return _normalize(pts)

def _min_pairwise_sq_dist(pts):
    """
    Compute the minimum squared Euclidean distance between all pairs.
    Returns (min_sq_dist, min_pair_indices).
    """
    # Efficiently compute squared distances
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    # Since pts are normalized, ||a||^2 = 1
    dots = pts @ pts.T
    sq_dists = 2.0 - 2.0 * dots
    
    # Mask diagonal to ignore self-distance
    np.fill_diagonal(sq_dists, np.inf)
    
    min_val = np.min(sq_dists)
    idx = np.unravel_index(np.argmin(sq_dists), sq_dists.shape)
    
    return min_val, idx

def _tammes_objective(flat_pts):
    """
    Objective function for L-BFGS-B: maximize min pairwise squared distance.
    We minimize the negative of this.
    """
    pts = flat_pts.reshape(N, 3)
    # Normalize to ensure we stay on sphere during optimization steps
    pts = _normalize(pts)
    
    min_sq_d, _ = _min_pairwise_sq_dist(pts)
    
    # Return negative to minimize
    return -min_sq_d

def _tammes_gradient(flat_pts):
    """
    Approximate gradient for L-BFGS-B.
    Since the objective is non-smooth (min), we use a subgradient approach 
    or rely on L-BFGS-B's numerical gradient if we don't provide one.
    Providing a custom gradient for 'min' is tricky. 
    However, for the *pair* that achieves the minimum, the gradient of the objective 
    w.r.t. the points in that pair is well-defined.
    
    Let i, j be the indices of the closest pair.
    f = ||p_i - p_j||^2 = 2 - 2 p_i . p_j
    df/dp_i = -2 p_j
    df/dp_j = -2 p_i
    
    We want to maximize f, so gradient of -f is:
    d(-f)/dp_i = 2 p_j
    d(-f)/dp_j = 2 p_i
    
    For all other points, the gradient is 0 (since they don't define the min).
    This is a valid subgradient.
    """
    pts = flat_pts.reshape(N, 3)
    pts = _normalize(pts)
    
    min_sq_d, (i, j) = _min_pairwise_sq_dist(pts)
    
    grad = np.zeros_like(flat_pts)
    
    # Gradient of - (2 - 2 p_i . p_j)
    # d/dp_i (-2 + 2 p_i . p_j) = 2 p_j
    # d/dp_j (-2 + 2 p_i . p_j) = 2 p_i
    
    grad[i*3:(i+1)*3] = 2.0 * pts[j]
    grad[j*3:(j+1)*3] = 2.0 * pts[i]
    
    # Project gradient to tangent space of the sphere
    # For point k, tangent space is orthogonal to p_k.
    # grad_proj = grad - (grad . p_k) p_k
    for k in range(N):
        p_k = pts[k]
        g_k = grad[k*3:(k+1)*3]
        dot = np.dot(g_k, p_k)
        grad[k*3:(k+1)*3] = g_k - dot * p_k
        
    return grad

def _refine_covering_radius(pts, samples, time_limit, start_time):
    """
    Refine points to minimize covering radius using farthest-point sampling logic.
    This directly targets the metric we care about.
    """
    pts = _normalize(pts).copy()
    best_pts = pts.copy()
    
    # Calculate initial covering radius estimate
    def _get_covering_radius(pts):
        dots = samples @ pts.T
        max_dots = dots.max(axis=1)
        # Covering radius is max over y of min_i d(y, p_i)
        # min_i d(y, p_i) = arccos(max_i y.p_i)
        # Max of arccos(x) is arccos(min x)
        min_max_dot = np.min(max_dots)
        return np.degrees(np.arccos(np.clip(min_max_dot, -1.0, 1.0)))
    
    best_radius = _get_covering_radius(pts)
    
    lr = 0.05
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        # Assign samples to nearest point
        dots = samples @ pts.T
        nearest_idx = dots.argmax(axis=1)
        
        new_pts = pts.copy()
        
        # For each point, find the farthest assigned sample
        # Move point towards that sample
        for j in range(N):
            mask = (nearest_idx == j)
            if not np.any(mask):
                continue
            
            assigned_dots = dots[mask, j]
            # Farthest sample has minimum dot product
            far_idx_in_mask = np.argmin(assigned_dots)
            
            # Get the actual index in samples
            indices = np.where(mask)[0]
            far_sample_idx = indices[far_idx_in_mask]
            far_sample = samples[far_sample_idx]
            
            # Move towards far sample
            diff = far_sample - pts[j]
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-10:
                new_pts[j] = pts[j] + lr * diff / diff_norm
        
        new_pts = _normalize(new_pts)
        
        # Evaluate
        new_radius = _get_covering_radius(new_pts)
        
        if new_radius < best_radius - 1e-6:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            no_improve_count = 0
            lr = max(0.01, lr * 0.98)
        else:
            no_improve_count += 1
            lr = max(0.005, lr * 0.9)
            if no_improve_count > 20:
                break
                
    return best_pts, best_radius

def construct():
    """
    Construct 50 points on unit sphere minimizing covering radius.
    Strategy:
    1. Multi-start Tammes optimization (max min pair dist) using L-BFGS-B.
    2. Refine using farthest-point covering radius minimization.
    """
    start_time = time.time()
    
    best_pts = None
    best_radius = np.inf
    
    # Number of restarts for Tammes optimization
    n_restarts = 5
    
    for i in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 30:
            break
        
        # 1. Initialization: Fibonacci spiral + random perturbation
        rng = np.random.default_rng(42 + i)
        
        # Fibonacci spiral base
        idx = np.arange(N) + 0.5
        z = 1.0 - 2.0 * idx / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * idx
        r = np.sqrt(1.0 - z * z)
        base_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Add small random noise to break symmetry
        noise = rng.normal(scale=0.1, size=(N, 3))
        init_pts = _normalize(base_pts + noise)
        
        # 2. Tammes Optimization (Max Min Pair Dist)
        flat_init = init_pts.flatten()
        
        # L-BFGS-B settings
        # We use the subgradient for the closest pair.
        # This can be noisy, so we limit iterations.
        try:
            res = minimize(
                _tammes_objective,
                flat_init,
                jac=_tammes_gradient,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            tammes_pts = _normalize(res.x.reshape(N, 3))
        except Exception:
            tammes_pts = init_pts
        
        # 3. Refine Covering Radius
        # Generate samples for refinement
        samples = _generate_random_samples(_N_SAMPLES, rng)
        
        remaining_time = _DEADLINE - (time.time() - start_time) - 5
        if remaining_time < 5:
            # Just evaluate tammes_pts if no time for refinement
            dots = samples @ tammes_pts.T
            max_dots = dots.max(axis=1)
            min_max_dot = np.min(max_dots)
            current_radius = np.degrees(np.arccos(np.clip(min_max_dot, -1.0, 1.0)))
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = tammes_pts
            continue
            
        refined_pts, refined_radius = _refine_covering_radius(
            tammes_pts, samples, time_limit=remaining_time, start_time=start_time
        )
        
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_pts = refined_pts.copy()
            
    if best_pts is None:
        # Fallback: Fibonacci spiral
        idx = np.arange(N) + 0.5
        z = 1.0 - 2.0 * idx / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * idx
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _normalize(best_pts)
        
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
