"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import itertools

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls
    min_dist_to_wall = np.min(0.5 - np.abs(centers))

    # Minimum pairwise distance / 2
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=1.0)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d

    return min(min_dist_to_wall, min_pair_dist / 2.0)

def _get_initial_centers():
    """
    Generate a good initial configuration.
    We use a jittered grid to avoid symmetries that trap local optimizers.
    50 spheres. 4x4x4 = 64 grid points.
    We pick 50 points.
    """
    # 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    # Add small deterministic jitter based on index to break symmetry
    # But we want to select 50 out of 64.
    # Let's just use the 50 points closest to the center of the cube?
    # Or just the first 50?
    # A better initialization:
    # Use a 3x4x5 grid = 60 points. Remove 10.
    # Or simply random start? Random start with L-BFGS-B might get stuck in bad local minima.
    # Let's use the 4x4x4 grid, but remove the 14 points with the largest distance to the center?
    # No, corners are good for packing?
    # Actually, for sphere packing, corners are not always optimal for the *minimum* distance if interior points are closer.
    # Let's try a standard grid initialization with slight random perturbation.
    
    # 4x4x4 grid
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # We need 50 points.
    # Let's sort by distance to origin and take the first 50?
    # This creates a cluster in the middle, which might be bad.
    # Let's take the first 50 in row-major order? That leaves the "back" of the cube empty.
    
    # Better: Interleave.
    # Or just use a random permutation of the 64 points and take 50?
    # Deterministic is safer for reproducibility, but random with fixed seed is okay.
    # Let's use a fixed seed random selection from the 64 grid points.
    rng = np.random.default_rng(42)
    indices = rng.permutation(64)[:50]
    centers = grid[indices]
    
    # Add small jitter to break exact grid symmetry
    jitter = rng.uniform(-0.01, 0.01, size=centers.shape)
    centers += jitter
    
    # Clip to ensure inside
    centers = np.clip(centers, -0.49, 0.49)
    
    return centers

def _objective_and_grad(x, r_target, beta):
    """
    Objective: Sum of soft-violations for all constraints.
    Constraints:
    1. Wall distance: 0.5 - |x_i| >= r_target
    2. Pairwise distance: ||x_i - x_j|| >= 2 * r_target
    
    We use a log-sum-exp approximation for the max violation to make it smooth.
    Actually, sum of exp(violation / beta) is a good barrier.
    Violation = max(0, target - actual).
    To be smooth everywhere, we can use:
    f(d, t) = log(1 + exp((t - d)/beta)) * beta
    This approximates max(0, t-d).
    
    We want to minimize the sum of these penalties.
    If sum is close to 0, all constraints are satisfied.
    """
    centers = x.reshape(N, 3)
    
    # 1. Wall constraints
    # d_wall = 0.5 - |centers|
    # We need d_wall >= r_target
    # violation = r_target - d_wall
    # penalty = beta * log(1 + exp(violation/beta))
    
    abs_centers = np.abs(centers)
    d_walls = 0.5 - abs_centers
    v_walls = r_target - d_walls
    # Soft plus: beta * log(1 + exp(v/beta))
    # Use stable log-sum-exp: max(v, 0) + log(1 + exp(-|v|))
    # Actually, for large v, exp(v/beta) overflows.
    # Let's use:
    # f(v) = beta * log(1 + exp(v/beta))
    # If v/beta > 50, f(v) ~ v.
    # If v/beta < -50, f(v) ~ 0.
    
    v_walls_scaled = v_walls / beta
    # Clip to avoid overflow
    v_walls_scaled = np.clip(v_walls_scaled, -50, 50)
    p_walls = beta * np.log1p(np.exp(v_walls_scaled))
    obj = np.sum(p_walls)
    
    grad = np.zeros_like(x)
    
    # Gradient of p_walls w.r.t centers
    # d/dx [ beta * log(1 + exp((t - (0.5 - |x|))/beta)) ]
    # = beta * ( exp((t - d)/beta) / (1 + exp((t - d)/beta)) ) * (1/beta) * d/dx(t - d)
    # = sigmoid(v/beta) * d/dx(r_target - (0.5 - |x|))
    # d/dx(r_target - 0.5 + |x|) = sign(x)
    
    sig_walls = 1.0 / (1.0 + np.exp(-v_walls_scaled))
    # grad_p_walls = sig_walls * sign(centers)
    grad_p_walls = sig_walls * np.sign(centers)
    # Handle 0 sign? If center is 0, derivative of |x| is undefined, but subgradient 0 is fine.
    grad_p_walls[centers == 0] = 0.0
    
    grad += grad_p_walls.reshape(N, 3)
    
    # 2. Pairwise constraints
    # We need to check all pairs. 50 points -> 1225 pairs.
    # This is small enough to compute fully in vectorized numpy without KDTree for speed in gradient?
    # KDTree is good for finding *which* pairs are close, but for gradient we need all pairs that might violate.
    # With r_target ~ 0.13, and cube side 1, most pairs are far.
    # But to be safe, let's just compute all pairs. 1225 pairs * 3 dims is tiny.
    
    # i < j
    # Create indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    # centers[i_idx] - centers[j_idx]
    diffs = centers[i_idx] - centers[j_idx]
    dists = np.linalg.norm(diffs, axis=1)
    
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-8, 1e-8, dists)
    
    # violation = 2 * r_target - dist
    v_pairs = 2 * r_target - dists
    v_pairs_scaled = v_pairs / beta
    v_pairs_scaled = np.clip(v_pairs_scaled, -50, 50)
    
    p_pairs = beta * np.log1p(np.exp(v_pairs_scaled))
    obj += np.sum(p_pairs)
    
    # Gradient
    # f(v) = beta log(1+exp(v/beta))
    # df/dv = sigmoid(v/beta)
    # dv/ddist = -1
    # ddist/dx_i = (x_i - x_j) / dist
    # ddist/dx_j = (x_j - x_i) / dist = - (x_i - x_j) / dist
    
    sig_pairs = 1.0 / (1.0 + np.exp(-v_pairs_scaled))
    
    # grad_diffs = sig_pairs * (-1) * (diffs / dists_safe[:, None])
    # Note: diffs = x_i - x_j
    # d(dist)/dx_i = diffs / dist
    # d(dist)/dx_j = -diffs / dist
    
    # term = sig_pairs[:, None] * (diffs / dists_safe[:, None])
    # grad_x_i += term
    # grad_x_j -= term
    
    term = sig_pairs[:, None] * (diffs / dists_safe[:, None])
    
    # Scatter add to grad
    # grad[i_idx] += term
    # grad[j_idx] -= term
    np.add.at(grad, i_idx, term)
    np.add.at(grad, j_idx, -term)
    
    return obj, grad

def construct():
    """Construct 50 sphere centers in unit cube to maximize minimum radius."""
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Initial guess
    centers = _get_initial_centers()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Start with a small r_target and increase
    # The optimal r is ~ 0.136
    # Let's start from r=0.1 and go up to 0.14
    
    r_target = 0.10
    beta = 0.01
    
    x = centers.flatten()
    
    # Iteratively increase r_target
    # We will try to satisfy constraints for r_target.
    # If we succeed (obj ~ 0), we increase r_target.
    
    max_r = 0.140
    r_step = 0.001
    fail_count = 0
    
    while r_target < max_r and time.time() < deadline - 5:
        # Try to optimize for current r_target
        try:
            res = minimize(
                _objective_and_grad, 
                x, 
                args=(r_target, beta), 
                method='L-BFGS-B', 
                jac=True,
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False}
            )
            
            if res.success:
                x = res.x
                centers = x.reshape(N, 3)
                # Update best if valid
                r_curr = _compute_radius(centers)
                if r_curr > best_radius:
                    best_radius = r_curr
                    best_centers = centers.copy()
                
                # If objective is small, we likely satisfied the constraint
                if res.fun < 1e-4:
                    r_target += r_step
                    fail_count = 0
                else:
                    # Failed to satisfy, try smaller beta? Or break?
                    # If we can't satisfy r_target, we should stop increasing.
                    # But maybe we need more iterations?
                    # For now, break if we fail significantly.
                    if res.fun > 1.0:
                        break
                    else:
                        # Try again with same r_target?
                        # Or reduce step?
                        # Let's just break to be safe with time.
                        break
            else:
                # Optimization failed
                break
                
        except Exception:
            break
            
        # Decrease beta slightly to sharpen the barrier as we progress?
        # Or keep it constant.
        # beta = max(0.001, beta * 0.9)
        
    # Final polish: use the best found radius as a target for one last optimization?
    # Or just return best_centers.
    
    # Ensure return shape and type
    return best_centers

# EVOLVE-BLOCK-END
