"""Improved packing of 50 equal circles in a unit square."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the packing radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distances
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    min_bound = np.min(np.minimum(dist_x, dist_y))
    
    return min(min_pair, min_bound)

def _soft_min_obj_and_grad(x, n, beta):
    """Soft-min objective and gradient. x is flattened centers (2n values)."""
    centers = x.reshape(n, 2)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Boundary distances
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    bound_vals = np.minimum(dist_x, dist_y)
    
    # Combine all values
    m = n * (n - 1)  # number of pairwise distances
    all_vals = np.concatenate([pair_vals[~np.eye(n, dtype=bool)], bound_vals])
    total = len(all_vals)
    
    # Soft min using log-sum-exp for stability
    # min(x) ≈ -1/beta * log(sum(exp(-beta * x)))
    m_shift = np.max(all_vals)
    exp_terms = np.exp(-beta * (all_vals - m_shift))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = m_shift - log_sum / beta
    
    # Gradient computation
    # d(soft_min)/d(all_vals) = -exp(-beta*(all_vals - m_shift)) / sum(exp(-beta*(all_vals - m_shift)))
    weights = exp_terms / np.sum(exp_terms)
    
    grad_centers = np.zeros_like(centers)
    
    # Pairwise gradient contributions
    # For pair (i,j), all_vals index = i*(n-1) + j - (1 if j > i else 0)
    # We need to accumulate gradients for each center
    for i in range(n):
        for j in range(i + 1, n):
            # Index in pair_vals
            # pair_vals is n x n, flattened with diagonal removed
            # Let's compute index differently
            # Actually, let's just compute the gradient directly
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-15:
                continue
            # The value is dist/2
            # d(value)/d(centers[i]) = d/(2*dist)
            # d(value)/d(centers[j]) = -d/(2*dist)
            w = weights[i * (n - 1) + j - 1] if j > i else weights[j * (n - 1) + i]
            # Actually the indexing is tricky. Let me redo this.
            pass
    
    # Let me use a cleaner approach: compute gradients via the chain rule
    # all_vals[k] for k < m corresponds to pair (i,j)
    # all_vals[k] for k >= m corresponds to boundary constraint for center (k-m)
    
    # For pairwise: value = dist(i,j)/2
    # d(dist/2)/d(c_i) = (c_i - c_j) / (2 * dist)
    # d(dist/2)/d(c_j) = (c_j - c_i) / (2 * dist) = -(c_i - c_j) / (2 * dist)
    
    # For boundary: value = min(0.5 - |c_x|, 0.5 - |c_y|)
    # This is nonsmooth, but we can use the subgradient
    # If 0.5 - |c_x| < 0.5 - |c_y|, value = 0.5 - |c_x|
    #   d/d(c_x) = -sign(c_x), d/d(c_y) = 0
    # If 0.5 - |c_x| > 0.5 - |c_y|, value = 0.5 - |c_y|
    #   d/d(c_x) = 0, d/d(c_y) = -sign(c_y)
    # If equal, we can average: d/d(c_x) = -0.5*sign(c_x), d/d(c_y) = -0.5*sign(c_y)
    
    # Let's recompute weights properly
    # Build the mapping from all_vals indices to (type, i, j) or (type, i)
    
    # For efficiency, let's compute the gradient in a vectorized way
    # This is complex, so let's use a simpler approach:
    # Use finite differences for the gradient if needed, or
    # Use a simpler gradient approximation
    
    # Actually, let's just return the objective and let L-BFGS-B use its own
    # gradient approximation, or compute the gradient more carefully.
    
    # For now, let's compute the gradient properly but in a loop (slower but correct)
    grad_x = np.zeros_like(x)
    
    # Pairwise contributions
    for i in range(n):
        for j in range(i + 1, n):
            # Find the index in all_vals
            # all_vals = [pair_vals[0,1], pair_vals[0,2], ..., pair_vals[0,n-1],
            #             pair_vals[1,2], ..., pair_vals[1,n-1],
            #             ..., pair_vals[n-2,n-1],
            #             bound_vals[0], ..., bound_vals[n-1]]
            # Index for pair (i,j) where i < j:
            # Count pairs before row i: i*(n-1) - i*(i-1)/2 ... this is getting complex
            # Let me just iterate and track the index
            pass
    
    # This is getting too complex. Let me use a different approach:
    # Compute the gradient using automatic differentiation-like approach
    # or just use the objective without gradient and let L-BFGS-B approximate it.
    
    # For simplicity and correctness, let's just return the objective
    # and let the optimizer use finite differences for gradient.
    # This is slower but more reliable.
    
    obj = -soft_min_val
    return obj

def _soft_min_obj(x, n, beta):
    """Soft-min objective for optimization. x is flattened centers."""
    centers = x.reshape(n, 2)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Boundary distances
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    bound_vals = np.minimum(dist_x, dist_y)
    
    # Combine all values
    all_vals = np.concatenate([pair_vals[~np.eye(n, dtype=bool)], bound_vals])
    
    # Soft min using log-sum-exp for stability
    m = np.max(all_vals)
    exp_terms = np.exp(-beta * (all_vals - m))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = m - log_sum / beta
    
    return -soft_min_val

def _grad_soft_min(x, n, beta):
    """Gradient of the soft-min objective."""
    centers = x.reshape(n, 2)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Boundary distances
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    bound_vals = np.minimum(dist_x, dist_y)
    
    # Combine all values
    all_vals = np.concatenate([pair_vals[~np.eye(n, dtype=bool)], bound_vals])
    total = len(all_vals)
    n_pairs = n * (n - 1) // 2
    
    # Soft min weights
    m_shift = np.max(all_vals)
    exp_terms = np.exp(-beta * (all_vals - m_shift))
    weights = exp_terms / np.sum(exp_terms)
    
    grad_centers = np.zeros_like(centers)
    
    # Pairwise gradient contributions
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = weights[idx]
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist > 1e-15:
                grad_centers[i] += w * d / (2 * dist)
                grad_centers[j] -= w * d / (2 * dist)
            idx += 1
    
    # Boundary gradient contributions
    for i in range(n):
        w = weights[n_pairs + i]
        dx = 0.5 - abs(centers[i, 0])
        dy = 0.5 - abs(centers[i, 1])
        if dx < dy:
            grad_centers[i, 0] += w * (-np.sign(centers[i, 0]))
        elif dy < dx:
            grad_centers[i, 1] += w * (-np.sign(centers[i, 1]))
        else:
            grad_centers[i, 0] += w * (-0.5 * np.sign(centers[i, 0]))
            grad_centers[i, 1] += w * (-0.5 * np.sign(centers[i, 1]))
    
    # The objective is -soft_min, so gradient is -grad(soft_min)
    # grad(soft_min)/d(centers) = sum_k weights[k] * d(all_vals[k])/d(centers)
    # So grad(objective) = -grad(soft_min) = -sum_k weights[k] * d(all_vals[k])/d(centers)
    
    return -grad_centers.flatten()

def _simulated_annealing(centers, n, time_budget, start_temp=0.05, end_temp=0.001, cooling_rate=0.9995):
    """Simulated annealing to improve the packing."""
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    current_centers = centers.copy()
    current_r = best_r
    
    temp = start_temp
    step_size = 0.02
    
    while time_budget > 0:
        # Propose a random move
        move_idx = np.random.randint(0, n)
        delta = np.random.normal(0, step_size, size=2)
        new_centers = current_centers.copy()
        new_centers[move_idx] += delta
        
        # Ensure within bounds
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        new_r = _compute_radius(new_centers)
        delta_r = new_r - current_r
        
        # Accept or reject
        if delta_r > 0 or np.random.random() < np.exp(delta_r / temp):
            current_centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
        
        # Cool down
        temp *= cooling_rate
        if temp < end_temp:
            temp = end_temp
        
        time_budget -= 0.001  # Approximate time per iteration
    
    return best_centers, best_r

def _lbfgs_polish(centers, n, time_budget):
    """L-BFGS-B polish with increasing beta."""
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    betas = [50, 100, 200, 500, 1000]
    
    for beta in betas:
        if time_budget < 5:
            break
        
        x0 = best_centers.flatten()
        
        try:
            result = minimize(
                _soft_min_obj,
                x0,
                args=(n, beta),
                jac=_grad_soft_min,
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * (2 * n),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            refined_centers = result.x.reshape(n, 2)
            refined_r = _compute_radius(refined_centers)
            
            if refined_r > best_r:
                best_r = refined_r
                best_centers = refined_centers.copy()
                
        except Exception:
            pass
        
        time_budget -= 5
    
    return best_centers, best_r

def _grid_init(n):
    """Create a grid initialization."""
    # For 50 circles, use a 7x7 grid (49 points) plus one extra
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_49 = np.array([(x, y) for y in ticks for x in ticks])
    
    if n == 49:
        return grid_49
    
    # Find best spot for the 50th point
    sample_res = 200
    xs = np.linspace(-0.5, 0.5, sample_res)
    ys = np.linspace(-0.5, 0.5, sample_res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Distance to existing grid points
    diff = pts[:, np.newaxis, :] - grid_49[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    min_dists = np.min(dists, axis=1)
    
    # Distance to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    d_b = np.minimum(d_x, d_y)
    
    # Maximize the minimum clearance
    clearance = np.minimum(min_dists, d_b)
    idx = np.argmax(clearance)
    new_pt = pts[idx]
    
    centers = np.vstack([grid_49, new_pt])
    return centers[:n]

def _hex_init(n):
    """Create a hexagonal lattice initialization."""
    # Hexagonal packing
    r = 1.0 / (2 * np.sqrt(3))  # Approximate radius for hex packing
    dx = 2 * r
    dy = np.sqrt(3) * r
    
    centers = []
    row = 0
    while len(centers) < n:
        for col in range(10):
            x = (col + 0.5 * (row % 2)) * dx - 0.5
            y = row * dy - 0.5
            if -0.49 <= x <= 0.49 and -0.49 <= y <= 0.49:
                centers.append([x, y])
            if len(centers) >= n:
                break
        row += 1
    
    centers = np.array(centers[:n])
    
    # Center the configuration
    centers -= np.mean(centers, axis=0)
    
    return centers

def _random_init(n, seed=42):
    """Create a random initialization."""
    np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, size=(n, 2))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_centers = None
    best_r = 0.0
    
    # Multiple initializations
    inits = [
        _grid_init(N),
        _hex_init(N),
        _random_init(N, 42),
        _random_init(N, 123),
    ]
    
    for init_centers in inits:
        if time.time() > deadline - 20:
            break
        
        centers = init_centers.copy()
        r = _compute_radius(centers)
        
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
        
        # Simulated annealing
        sa_budget = min(30, deadline - time.time() - 10)
        if sa_budget > 5:
            sa_centers, sa_r = _simulated_annealing(centers, N, sa_budget)
            if sa_r > best_r:
                best_r = sa_r
                best_centers = sa_centers.copy()
        
        # L-BFGS polish
        lbfgs_budget = min(30, deadline - time.time() - 5)
        if lbfgs_budget > 5:
            lbfgs_centers, lbfgs_r = _lbfgs_polish(best_centers, N, lbfgs_budget)
            if lbfgs_r > best_r:
                best_r = lbfgs_r
                best_centers = lbfgs_centers.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 2 and best_centers is not None:
        try:
            x0 = best_centers.flatten()
            result = minimize(
                _soft_min_obj,
                x0,
                args=(N, 2000),
                jac=_grad_soft_min,
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            refined_centers = result.x.reshape(N, 2)
            refined_r = _compute_radius(refined_centers)
            if refined_r > best_r:
                best_r = refined_r
                best_centers = refined_centers
        except Exception:
            pass
    
    # Ensure validity
    if best_centers is not None:
        eps = 1e-9
        best_centers = np.clip(best_centers, -0.5 + eps, 0.5 - eps)
    else:
        best_centers = _grid_init(N)
        best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers

# EVOLVE-BLOCK-END
