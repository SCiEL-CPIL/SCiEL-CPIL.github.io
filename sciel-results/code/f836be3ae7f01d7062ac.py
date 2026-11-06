"""
Free-point smooth optimization strategy for packing 50 equal circles in a unit square.

Instead of restricting to a 6-parameter hexagonal lattice, we treat all 100 coordinates
as free variables. We define a smooth surrogate objective using log-sum-exp (soft-min)
of the "constraint distances" (half pairwise distances and boundary distances).
This avoids the kinks of the true min-function and allows L-BFGS-B to operate effectively.
We use multiple restarts from jittered hexagonal grids to escape local minima.
"""
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 50
SEED = 42
DEADLINE_FRACTION = 0.85

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    
    # Boundary distances
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    
    if min_boundary <= 0:
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _soft_min_obj(x, beta):
    """
    Objective function for L-BFGS-B minimization.
    We want to maximize r = min(constraints).
    So we minimize -r.
    Using soft-min: soft_min(C) = -1/beta * log(sum(exp(-beta * c)))
    We want to maximize soft_min, so we minimize -soft_min.
    
    Constraints:
    1. Boundary distances: 0.5 - |x_i|, 0.5 - |y_i| (Total 4N values)
    2. Pairwise half-distances: 0.5 * |x_i - x_j| (Total N*(N-1)/2 values)
    
    x: flat array of shape (2N,)
    """
    centers = x.reshape(N, 2)
    
    # 1. Boundary constraints
    # d_i = 0.5 - |x_i|, 0.5 - |y_i|
    # Note: |x| is not differentiable at 0, but L-BFGS-B usually handles it okay or we can smooth it.
    # For now, standard abs is often fine for this scale, but let's use a smooth approximation if needed.
    # Actually, scipy minimize with L-BFGS-B uses numerical gradients or analytic if provided.
    # We will provide analytic gradients for speed and robustness.
    
    # Let's compute the values first.
    bx = 0.5 - np.abs(centers[:, 0])
    by = 0.5 - np.abs(centers[:, 1])
    bd = np.concatenate([bx, by])
    
    # 2. Pairwise constraints
    # To save memory and time, we compute upper triangle.
    # diff = centers[:, None, :] - centers[None, :, :]
    # This is 50x50x2 = 5000 floats. Very small.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    pd = dists / 2.0
    # Extract upper triangle
    iu = np.triu_indices(N, k=1)
    pd_vals = pd[iu]
    
    # Combine all constraints into one vector C
    # C = [bd, pd_vals]
    # Size: 100 + 1225 = 1325
    
    # Soft min
    # To avoid overflow in exp, subtract max
    C = np.concatenate([bd, pd_vals])
    max_C = np.max(C)
    # If max_C is very negative, exp(-beta*C) might underflow/overflow?
    # exp(-beta * (C - max_C)) is safe.
    exp_terms = np.exp(-beta * (C - max_C))
    sum_exp = np.sum(exp_terms)
    
    soft_min_val = max_C - np.log(sum_exp) / beta
    
    # We want to MINIMIZE -soft_min_val
    # Gradient of soft_min_val w.r.t x:
    # d(soft_min)/dx = sum_j w_j * (dc_j/dx)
    # where w_j = exp(-beta * c_j) / sum(exp(-beta * c_k))
    
    weights = exp_terms / sum_exp
    
    grad = np.zeros(2 * N)
    
    # Gradient for boundary constraints
    # w_bd corresponds to first 2N weights
    w_bd = weights[:2 * N]
    
    # d/dx_i (0.5 - |x_i|) = -sign(x_i)
    # d/dy_i (0.5 - |y_i|) = -sign(y_i)
    # Note: sign(0) is 0.
    
    # bx corresponds to x coords, by to y coords
    # w_bd[0:N] are weights for bx
    # w_bd[N:2N] are weights for by
    
    # grad_x += sum_i w_bx_i * (-sign(x_i))
    # grad_y += sum_i w_by_i * (-sign(y_i))
    
    signs_x = np.sign(centers[:, 0])
    signs_y = np.sign(centers[:, 1])
    
    grad[0::2] += -w_bd[0:N] * signs_x
    grad[1::2] += -w_bd[N:2 * N] * signs_y
    
    # Gradient for pairwise constraints
    # w_pd corresponds to last N*(N-1)/2 weights
    w_pd = weights[2 * N:]
    
    # c_ij = 0.5 * dist_ij
    # d/dx_i (0.5 * dist_ij) = 0.5 * (x_i - x_j) / dist_ij
    # We need to accumulate these into grad.
    
    # This loop is O(N^2). For N=50, 2500 iterations. Acceptable?
    # 2500 * (few ops) is fast.
    # But we are calling this many times.
    # Vectorization is better.
    
    # Let's try to vectorize the pairwise gradient accumulation.
    # We have a list of pairs (i, j) with weight w_k.
    # Contribution to grad[i] += w_k * 0.5 * (x_i - x_j) / dist_ij
    # Contribution to grad[j] += w_k * 0.5 * (x_j - x_i) / dist_ij
    
    # iu[0] is array of i indices, iu[1] is array of j indices.
    i_idx = iu[0]
    j_idx = iu[1]
    
    # dists[i_idx, j_idx] gives distances for these pairs
    d_ij = dists[i_idx, j_idx]
    # Avoid division by zero (shouldn't happen if centers distinct)
    d_ij_safe = np.where(d_ij > 1e-12, d_ij, 1.0)
    
    # diff_vec[i] = x_i - x_j
    # x_i = centers[i_idx]
    # x_j = centers[j_idx]
    diff_vec = centers[i_idx] - centers[j_idx]
    
    # factor_k = 0.5 * w_k / d_k
    factors = 0.5 * w_pd / d_ij_safe
    
    # grad[i] += sum_k factor_k * diff_vec_k
    # grad[j] -= sum_k factor_k * diff_vec_k
    
    # Use np.add.at for unbuffered addition (scatter)
    np.add.at(grad[0::2], i_idx, factors * diff_vec[:, 0])
    np.add.at(grad[1::2], i_idx, factors * diff_vec[:, 1])
    
    np.add.at(grad[0::2], j_idx, -factors * diff_vec[:, 0])
    np.add.at(grad[1::2], j_idx, -factors * diff_vec[:, 1])
    
    # Objective is -soft_min_val
    val = -soft_min_val
    grad_out = -grad
    
    return val, grad_out

def _generate_hex_initial(jitter_scale=0.0):
    """Generate a hexagonal grid with optional jitter."""
    # Rows: 7, 6, 7, 6, 7, 6, 7, 4
    rows_info = [
        (7, 0),
        (6, 1),
        (7, 0),
        (6, 1),
        (7, 0),
        (6, 1),
        (7, 0),
        (4, 1),
    ]
    
    # Approximate optimal spacing for r ~ 0.07
    # r = 0.0713 -> d = 0.1427
    # Hex vertical spacing = d * sqrt(3)/2 ~ 0.123
    # Let's use a standard hex layout scaled to fit.
    dx = 0.143
    dy = dx * np.sqrt(3) / 2.0
    
    centers = []
    for i, (count, parity) in enumerate(rows_info):
        y = -0.5 + dy * (i + 0.5)
        row_width = (count - 1) * dx
        start_x = -row_width / 2.0
        
        if parity == 1:
            start_x += dx / 2.0
            
        for j in range(count):
            x = start_x + j * dx
            centers.append([x, y])
            
    centers = np.array(centers[:N])
    
    if jitter_scale > 0:
        jitter = np.random.normal(0, jitter_scale, size=centers.shape)
        centers += jitter
        # Clip to stay inside? No, let optimizer handle it, but keep close.
        # Actually, if we clip, we might introduce kinks. Let's just clip to [-0.49, 0.49] to be safe.
        np.clip(centers, -0.49, 0.49, out=centers)
        
    return centers

def _polish(centers, deadline):
    """
    Final polishing using direct gradient ascent on the exact min function.
    This corrects any bias from the soft-min approximation.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    start_time = time.time()
    
    lr = 0.01
    # Use a simple steepest ascent with line search or fixed step?
    # Fixed step with backtracking is safer.
    
    for it in range(2000):
        if time.time() > deadline:
            break
            
        # Compute gradients of the exact min?
        # The exact min is non-differentiable.
        # We can use the gradient of the soft-min with high beta as a proxy,
        # or use a subgradient.
        # Let's use the soft-min gradient with very high beta (sharp).
        beta = 100.0
        val, grad = _soft_min_obj(best_centers.flatten(), beta)
        # grad is for minimization, so we want to go in direction -grad
        direction = -grad.reshape(N, 2)
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            break
        direction /= norm
        
        # Line search: try a few steps
        improved = False
        for step_mult in [1.0, 0.5, 0.25, 0.1]:
            step = lr * step_mult
            candidate = best_centers + step * direction
            # Clip to square
            np.clip(candidate, -0.5, 0.5, out=candidate)
            
            new_r = _compute_radius(candidate)
            if new_r > best_radius + 1e-9:
                best_centers = candidate
                best_radius = new_r
                improved = True
                break
        if not improved:
            lr *= 0.8
            if lr < 1e-6:
                break
        else:
            lr *= 1.05
            
    return best_centers, best_radius

def construct():
    """Main entry point."""
    start_time = time.time()
    total_time_budget = 150.0
    deadline = start_time + total_time_budget * DEADLINE_FRACTION
    
    np.random.seed(SEED)
    random.seed(SEED)
    
    best_centers = None
    best_radius = -np.inf
    
    # Try multiple restarts
    # 1. Perfect hex
    # 2. Jittered hex (small)
    # 3. Jittered hex (medium)
    # 4. Random (if time permits)
    
    initializations = [
        _generate_hex_initial(0.0),
        _generate_hex_initial(0.005),
        _generate_hex_initial(0.01),
        _generate_hex_initial(0.02),
    ]
    
    # Maybe add a random one if time allows
    # Let's stick to 4 restarts for now to ensure time for optimization.
    
    for i, init_centers in enumerate(initializations):
        if time.time() > deadline:
            break
            
        x0 = init_centers.flatten()
        
        # We want to minimize -soft_min.
        # Beta starts moderate and increases?
        # Or just use a fixed high beta?
        # L-BFGS-B is a quasi-Newton method. It works best with smooth functions.
        # A moderate beta (e.g., 20-50) keeps it smooth enough.
        # If beta is too high, it becomes kinked.
        # Let's use beta=30.
        
        def obj_func(x):
            val, _ = _soft_min_obj(x, 30.0)
            return val
        
        def grad_func(x):
            _, grad = _soft_min_obj(x, 30.0)
            return grad
            
        try:
            result = minimize(
                obj_func,
                x0,
                jac=grad_func,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success:
                res_centers = result.x.reshape(N, 2)
                res_radius = _compute_radius(res_centers)
                
                if res_radius > best_radius:
                    best_radius = res_radius
                    best_centers = res_centers
        except Exception:
            pass
            
        # Polishing step on the best candidate found so far?
        # Or polish each candidate and keep the best?
        # Polishing is fast. Let's polish the result of this restart.
        if time.time() < deadline - 5.0: # Leave 5s for safety
            polished_centers, polished_radius = _polish(result.x.reshape(N, 2), deadline - 2.0)
            if polished_radius > best_radius:
                best_radius = polished_radius
                best_centers = polished_centers
    
    # Fallback if no valid result
    if best_centers is None:
        best_centers = _generate_hex_initial(0.0)
        best_radius = _compute_radius(best_centers)
        
    return best_centers

# EVOLVE-BLOCK-END
