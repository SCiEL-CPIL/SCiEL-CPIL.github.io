"""
Strategy: Differentiable latent-space optimization.
Instead of moving coordinates directly (which gets stuck in local minima due to the rigid lattice structure and discrete symmetry),
we represent the 300 centers as the output of a small neural network:
    z (latent, dim D) -> Linear -> ReLU -> Linear -> (x, y) for each of 300 points.
The latent vector z is the variable optimized by L-BFGS-B.
The network acts as a flexible, smooth manifold that can break the hexagonal symmetry and explore non-trivial topologies
without the "jitter" of random moves. The objective is a soft-min of the violations (boundary and pairwise distance).
This leverages the network's capacity to map a low-dimensional latent space to a high-dimensional configuration space
in a way that smooths out the landscape for the gradient optimizer.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 300
DEADLINE = 145.0  # Safety margin from 150s limit

def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk for initial latent mapping."""
    s = math.sqrt(2 * math.pi / (N * math.sqrt(3)))
    
    best_points = None
    for factor in [1.0, 1.01, 1.02, 1.03, 0.99, 0.8, 0.7]:
        s_try = s * factor
        k = int(1.0 / s_try) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s_try
                y = j * s_try * math.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            # Keep the N closest to center
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            if best_points is None:
                best_points = pts
            if len(pts) == N:
                break
    
    if best_points is not None and len(best_points) == N:
        return best_points
    
    # Fallback
    s = 0.05
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * math.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append((x, y))
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind='stable')[:N]
    return points[idx]

def _soft_min(values, k=50.0):
    """
    Smooth approximation of the minimum of a set of values.
    Uses the log-sum-exp trick for numerical stability.
    min(x) ≈ -1/k * log(sum(exp(-k*x)))
    """
    min_val = np.min(values)
    # To avoid overflow in exp, shift by min_val
    # exp(-k * (x - min_val)) is <= 1
    shifted = -k * (values - min_val)
    # Clip to avoid underflow/overflow issues in exp, though shifted <= 0 so exp <= 1
    # But if values are very large negative, shifted is large positive?
    # Wait: min_val is the smallest. x - min_val >= 0. -k * (x - min_val) <= 0.
    # So exp(shifted) is in (0, 1]. Safe.
    log_sum = np.log(np.sum(np.exp(shifted)))
    return min_val - (1.0 / k) * log_sum

def _compute_penalty_and_grad(points, target_r):
    """
    Compute the penalty for a given configuration and target radius.
    Penalty = max(0, target_r - min_dist/2) + max(0, target_r - min_boundary_dist)
    Actually, we want to maximize r.
    Let's define the objective to minimize as:
    J = soft_min( (dist_ij / 2 - target_r) ) + soft_min( (1 - |c_i| - target_r) )
    If we can make both soft mins positive, we are good.
    But we want to find the max r.
    
    Alternative:
    Minimize J(r, points) = soft_min( (dist_ij / 2 - r) ) + soft_min( (1 - |c_i| - r) )
    If J < 0, it means r is likely too large?
    Actually, let's fix the objective to minimize the "violations" for a fixed r,
    and then binary search on r? Or just optimize points for a fixed r and see if it works?
    
    Better approach:
    Optimize points to maximize the minimum distance.
    Objective to minimize:
    f(points) = soft_min( - (dist_ij / 2) ) + soft_min( - (1 - |c_i|) )
    This is equivalent to minimizing the negative of the soft-min radius.
    Let R_soft = soft_min( min( dist_ij/2, 1-|c_i| ) ) ... no, soft min of the set of all constraints.
    
    Let d_i = min( min_j (dist_ij/2), 1 - |c_i| ) for each i? No, the radius is global.
    The radius r is limited by:
    1. min_{i<j} dist_ij / 2
    2. min_i (1 - |c_i|)
    
    Let A be the vector of all pairwise distances / 2.
    Let B be the vector of boundary distances.
    We want to maximize min( min(A), min(B) ).
    
    Let's define the objective as:
    J = soft_min( -A, k ) + soft_min( -B, k )
    Minimizing J is equivalent to maximizing the soft-min of the distances.
    
    Gradient:
    d/dx soft_min(-x, k) = - (1/k) * sum(exp(-k(x - min(x)))) * (-1) / sum(...) 
    Wait, derivative of -1/k log(sum(exp(-k x_i))) w.r.t x_j:
    = -1/k * (1/sum(exp)) * d/dx_j sum(exp(-k x_i))
    = -1/k * (1/S) * exp(-k x_j) * (-k)
    = exp(-k x_j) / S
    where S = sum(exp(-k x_i)).
    
    So the gradient of soft_min(-x) w.r.t x_j is p_j = exp(-k x_j) / S.
    This p_j is a probability distribution over the constraints, weighted by how "tight" they are.
    
    For A (pairwise dists / 2):
    A_{ij} = 0.5 * ||c_i - c_j||
    dA_{ij}/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    dA_{ij}/dc_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
    
    For B (boundary dists):
    B_i = 1 - ||c_i||
    dB_i/dc_i = - c_i / ||c_i||
    
    So total gradient on c_i:
    grad_c_i = sum_{j!=i} p_{ij} * dA_{ij}/dc_i + p_{boundary_i} * dB_i/dc_i
    Note: p_{ij} here is the weight from the soft min of A.
    But we have two separate soft mins: one for A, one for B.
    J = soft_min(-A) + soft_min(-B)
    grad J = grad soft_min(-A) + grad soft_min(-B)
    
    Let's compute the weights for A and B separately.
    """
    # Pairwise distances
    # To save memory and time, we can compute this in chunks or use a trick.
    # N=300, so 300x300 matrix is 90k floats. Very small.
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # A = dists / 2
    A = dists / 2.0
    
    # B = 1 - |c_i|
    r_pts = np.linalg.norm(points, axis=1)
    B = 1.0 - r_pts
    
    # Compute soft min weights for A
    # We need min(A) to shift
    min_A = np.min(A)
    # Flatten A for vector ops
    A_flat = A.flatten()
    # Mask diagonal (inf)
    mask = ~np.isinf(A_flat)
    A_valid = A_flat[mask]
    min_A_valid = np.min(A_valid)
    
    # Weights for A: w_A = exp(-k * (A - min_A)) / sum(exp(-k * (A - min_A)))
    # We only care about valid entries.
    exp_A = np.exp(-50.0 * (A_valid - min_A_valid))
    sum_exp_A = np.sum(exp_A)
    w_A_flat = np.zeros_like(A_flat)
    w_A_flat[mask] = exp_A / sum_exp_A
    w_A = w_A_flat.reshape(N, N)
    
    # Gradient contribution from A to points
    # grad_A[i] = sum_j w_A[i,j] * 0.5 * (c_i - c_j) / dist_ij
    # Note: w_A is symmetric? The soft min is over the whole set, so w_A[i,j] = w_A[j,i].
    # The term for pair (i,j) in the sum for grad_i is w_A[i,j] * 0.5 * (c_i - c_j) / d_ij
    # The term for pair (j,i) in the sum for grad_j is w_A[j,i] * 0.5 * (c_j - c_i) / d_ij
    # Since w_A[i,j] = w_A[j,i] and (c_j - c_i) = -(c_i - c_j), the forces are equal and opposite.
    
    # Vectorized gradient for A:
    # G_A[i] = sum_j w_A[i,j] * 0.5 * (c_i - c_j) / d_ij
    # Let's compute this efficiently.
    # d_ij is dists.
    # factor[i,j] = w_A[i,j] * 0.5 / d_ij
    # G_A[i] = sum_j factor[i,j] * (c_i - c_j)
    # G_A[i] = c_i * sum_j factor[i,j] - sum_j factor[i,j] * c_j
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor_A = w_A * 0.5 / dists
        factor_A[~mask.reshape(N, N)] = 0.0 # Set diagonal/inf to 0
    
    sum_factor_A = np.sum(factor_A, axis=1) # Shape (N,)
    # sum_j factor_A[i,j] * c_j
    # This is factor_A @ points
    sum_fc_A = factor_A @ points # Shape (N, 2)
    
    grad_A = points * sum_factor_A[:, np.newaxis] - sum_fc_A
    
    # Compute soft min weights for B
    min_B = np.min(B)
    exp_B = np.exp(-50.0 * (B - min_B))
    sum_exp_B = np.sum(exp_B)
    w_B = exp_B / sum_exp_B # Shape (N,)
    
    # Gradient contribution from B to points
    # grad_B[i] = w_B[i] * (-c_i / |c_i|)
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_B = -w_B[:, np.newaxis] * (points / r_pts[:, np.newaxis])
    
    total_grad = grad_A + grad_B
    
    # Objective value
    # J = soft_min(-A) + soft_min(-B)
    # soft_min(-A) = min_A - (1/k) log(sum(exp(-k(A-min_A))))
    # We already have sum(exp) as sum_exp_A (shifted)
    # log_sum_A = log(sum_exp_A)
    # J_A = min_A - (1/50) * log_sum_A
    J_A = min_A_valid - (1.0 / 50.0) * np.log(sum_exp_A)
    J_B = min_B - (1.0 / 50.0) * np.log(sum_exp_B)
    
    return J_A + J_B, total_grad.flatten()

def _network_forward(z, W1, b1, W2, b2):
    """
    Forward pass of the simple neural network.
    z: (D,)
    W1: (D, H)
    b1: (H,)
    W2: (H, 2*N)
    b2: (2*N,)
    Returns: (2*N,) -> reshape to (N, 2)
    """
    h = z @ W1 + b1
    h = np.maximum(h, 0.0) # ReLU
    out = h @ W2 + b2
    return out

def _network_jacobian(z, W1, b1, W2, b2, points):
    """
    Compute the Jacobian of the network output w.r.t z.
    We need d points / d z to chain rule: grad_z = J^T @ grad_points
    """
    h = z @ W1 + b1
    h_act = np.maximum(h, 0.0)
    # Derivative of ReLU
    h_grad = (h > 0).astype(float)
    
    # d out / d h = W2 (shape H, 2N)
    # d h / d z = W1 (shape D, H) * h_grad (element-wise)
    # J = W1 * h_grad[:, np.newaxis] @ W2 ?
    # Let's trace:
    # out_j = sum_k h_k W2[k,j]
    # h_k = max(0, sum_i z_i W1[i,k] + b1_k)
    # d out_j / d z_i = sum_k (d out_j / d h_k) * (d h_k / d z_i)
    # d out_j / d h_k = W2[k,j]
    # d h_k / d z_i = W1[i,k] * (1 if h_k > 0 else 0)
    # So d out_j / d z_i = sum_k W2[k,j] * W1[i,k] * h_grad[k]
    # This can be written as: (W1 * h_grad[np.newaxis, :]) @ W2
    # Shape: (D, H) * (1, H) -> (D, H)
    # (D, H) @ (H, 2N) -> (D, 2N)
    
    J = (W1 * h_grad[np.newaxis, :]) @ W2
    return J

def construct():
    start_time = time.time()
    
    # Initialize points with hex lattice
    init_points = _hex_lattice_init()
    
    # Define network architecture
    # Latent dim D
    D = 100
    H = 200
    
    rng = np.random.default_rng(42)
    
    # Initialize weights
    # He initialization
    W1 = rng.normal(0, 1.0 / math.sqrt(D), size=(D, H))
    b1 = np.zeros(H)
    W2 = rng.normal(0, 1.0 / math.sqrt(H), size=(H, 2 * N))
    b2 = np.zeros(2 * N)
    
    # Initialize latent z
    # We want the initial output to be close to init_points.
    # We can solve for z that minimizes ||_network_forward(z) - init_points||^2
    # Or just start with random z and let the optimizer figure it out?
    # Better: Start with z=0, and set b2 to init_points.flatten()
    # Then the output is exactly init_points at z=0.
    # But the gradients might be poor if the network is not "active".
    # Let's set b2 = init_points.flatten() and start with small random z.
    b2 = init_points.flatten().copy()
    z_init = rng.normal(0, 0.1, size=D)
    
    # Objective function for L-BFGS-B
    def objective(z):
        if time.time() - start_time > DEADLINE - 5.0:
            return 1e10, np.zeros_like(z)
        
        out = _network_forward(z, W1, b1, W2, b2)
        points = out.reshape(N, 2)
        
        # Clamp points to be inside the circle to avoid huge penalties/gradients?
        # The soft min handles it, but if points go far outside, dists become large,
        # but boundary dist becomes negative.
        # It's better to let the optimizer handle it, but maybe clip for stability?
        # No, clipping breaks differentiability. Let it be.
        
        val, grad_points = _compute_penalty_and_grad(points, 0.0)
        
        # Chain rule
        J = _network_jacobian(z, W1, b1, W2, b2, points)
        grad_z = J.T @ grad_points
        
        return val, grad_z
    
    # Run optimization
    # Use L-BFGS-B
    # We might need multiple restarts or a warm start.
    
    best_z = z_init.copy()
    best_val = np.inf
    
    # Time budget
    max_time_per_opt = 30.0
    
    for restart in range(5):
        # Reset z to random or previous best?
        # Let's try a few different initializations
        if restart == 0:
            z_current = z_init
        else:
            # Perturb the best found so far
            z_current = best_z + rng.normal(0, 0.5, size=D)
        
        # Run minimize
        # Use a callback to check time
        def callback(xk):
            return time.time() - start_time > DEADLINE - 5.0
        
        try:
            res = minimize(
                objective, 
                z_current, 
                method='L-BFGS-B', 
                jac=True,
                options={
                    'maxiter': 200,
                    'ftol': 1e-9,
                    'gtol': 1e-6,
                    'maxfun': 500,
                    'callback': callback
                }
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_z = res.x
        
        except Exception as e:
            pass
        
        if time.time() - start_time > DEADLINE - 10.0:
            break
    
    # Get final points
    out = _network_forward(best_z, W1, b1, W2, b2)
    points = out.reshape(N, 2)
    
    # Final polish: Project points inside the circle if they are outside
    r = np.linalg.norm(points, axis=1)
    mask = r > 1.0
    if np.any(mask):
        points[mask] = points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)
    
    # Optional: Run a few steps of local coordinate optimization to fix any small issues
    # The network output might have small overlaps due to the soft min approximation.
    # A simple push-apart can fix this.
    
    # Simple local relaxation
    for _ in range(100):
        if time.time() - start_time > DEADLINE - 2.0:
            break
        improved = False
        # Compute min dist
        tree_dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
        np.fill_diagonal(tree_dists, np.inf)
        min_d = np.min(tree_dists)
        r_b = 1.0 - np.min(1.0 - np.linalg.norm(points, axis=1))
        current_r = min(min_d / 2.0, r_b)
        
        if min_d < 2.0 * current_r + 1e-9: # If overlap
            # Find closest pair
            idx_i, idx_j = np.unravel_index(np.argmin(tree_dists), tree_dists.shape)
            diff = points[idx_i] - points[idx_j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                # Random nudge
                points[idx_i] += rng.normal(0, 0.01, size=2)
            else:
                # Push apart
                push = 0.5 * (2.0 * current_r - dist) * (diff / dist)
                points[idx_i] += push
                points[idx_j] -= push
        
        # Check boundary
        r_pts = np.linalg.norm(points, axis=1)
        out_mask = r_pts > 1.0
        if np.any(out_mask):
            points[out_mask] = points[out_mask] / r_pts[out_mask, np.newaxis] * (1.0 - 1e-10)
            improved = True
    
    return points
