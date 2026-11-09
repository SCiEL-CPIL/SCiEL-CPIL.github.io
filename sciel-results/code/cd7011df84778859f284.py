"""Direct non-smooth optimization for 100 equal spheres in a unit cube."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.optimize as scipy_opt

def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    n = len(centres)
    if n == 0:
        return 0.0
    # Boundary clearance
    d_boundary = np.min(0.5 - np.abs(centres), axis=1)
    min_boundary = np.min(d_boundary)
    # Pairwise distances (vectorized)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_boundary)

def _soft_min_value_and_grad(centres, beta=200.0):
    """
    Compute soft-min of all constraint values (pairwise distances/2 and boundary clearances)
    and its gradient. This provides a smooth surrogate for the hard min.
    
    Returns:
        soft_min_val: the soft-min value
        grad: gradient of the soft-min w.r.t. centres (shape (N, 3))
    """
    n = len(centres)
    
    # Collect all constraint values and their gradients
    # Boundary constraints: g_ik = 0.5 - |c_ik|
    # Pairwise constraints: g_ij = |c_i - c_j| / 2
    
    # For efficiency, compute soft-min using log-sum-exp trick on negatives
    # soft_min(x) = -logsumexp(-beta * x) / beta
    
    # Compute all constraint values
    # Boundary: for each point i and coordinate k
    d_boundary = 0.5 - np.abs(centres)  # shape (n, 3)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We'll build a list of (value, gradient_contribution)
    # But for memory efficiency with N=100, let's use a different approach
    
    # Use the fact that soft_min is dominated by the smallest few values
    # First, find approximate minimum
    min_boundary = np.min(d_boundary)
    min_pair = np.min(dists)
    r_approx = min(min_pair / 2.0, min_boundary)
    
    # Only consider constraints that are close to the minimum
    tol = 0.05  # tolerance for "active" constraints
    
    # Build gradient
    grad = np.zeros_like(centres)
    
    # Collect active constraint weights
    # For soft-min: weight_i = exp(-beta * (x_i - x_min)) / sum(exp(-beta * (x_j - x_min)))
    
    # To avoid underflow/overflow, shift by the minimum
    # We'll compute weights for boundary and pairwise separately, then combine
    
    # Boundary constraints
    # For each (i, k), value = d_boundary[i, k]
    # Gradient contribution to c_ik: -sign(c_ik) if c_ik != 0, else 0
    # Actually d/dc_ik (0.5 - |c_ik|) = -sign(c_ik) for c_ik != 0
    
    b_vals = d_boundary.ravel()  # shape (3n,)
    b_idx = np.argmin(b_vals)
    b_min = b_vals[b_idx]
    
    # Pairwise constraints
    p_vals = (dists / 2.0)
    np.fill_diagonal(p_vals, np.inf)
    p_idx_flat = np.argmin(p_vals)
    p_min = p_vals.ravel()[p_idx_flat]
    
    # Combine: we have two sets of constraints
    # Let's compute the soft-min over all constraints together
    
    # For numerical stability, find global min
    all_min = min(b_min, p_min)
    
    # Compute weights for boundary constraints
    b_shifted = b_vals - all_min
    b_weights = np.exp(-beta * b_shifted)
    
    # Compute weights for pairwise constraints
    p_vals_flat = p_vals.ravel()
    p_shifted = p_vals_flat - all_min
    p_weights = np.exp(-beta * p_shifted)
    
    # Total weight
    total_weight = b_weights.sum() + p_weights.sum()
    
    if total_weight < 1e-300:
        # Degenerate case, fall back to hard min
        return r_approx, grad
    
    # Soft-min value
    soft_min_val = all_min + np.log(total_weight) / beta
    
    # Compute gradient from boundary constraints
    # For boundary constraint (i, k): value = 0.5 - |c_ik|
    # d(value)/d(c_ik) = -sign(c_ik) if c_ik != 0, 0.5 if c_ik == 0 (subgradient)
    # Weight for this constraint: b_weights[i*3+k] / total_weight
    # Contribution to grad[i, k]: weight * (-sign(c_ik))
    
    b_weights_reshaped = b_weights.reshape(n, 3)
    signs = np.sign(centres)
    # Handle c_ik = 0: use 0 as subgradient (or random, but 0 is safer)
    grad_boundary = -b_weights_reshaped * signs
    grad_boundary /= total_weight
    
    # Compute gradient from pairwise constraints
    # For pairwise constraint (i, j): value = |c_i - c_j| / 2
    # d(value)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|) if |c_i - c_j| > 0
    # d(value)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|) = -(c_i - c_j) / (2 * |c_i - c_j|)
    
    # p_weights is over the flattened upper triangle (or full matrix with diagonal inf)
    # Let's iterate over pairs
    grad_pair = np.zeros_like(centres)
    for i in range(n):
        for j in range(i + 1, n):
            w = p_weights[i * n + j] / total_weight
            if w < 1e-15:
                continue
            d = dists[i, j]
            if d < 1e-12:
                continue
            unit = (centres[i] - centres[j]) / d
            grad_pair[i] += w * unit / 2.0
            grad_pair[j] -= w * unit / 2.0
    
    grad = grad_boundary + grad_pair
    
    return soft_min_val, grad


def _fcc_lattice_init():
    """
    Generate FCC lattice points within the cube and select the best 100
    by maximum minimum clearance (both pairwise and boundary).
    """
    # FCC lattice basis vectors
    a1 = np.array([0.5, 0.5, 0.0])
    a2 = np.array([0.5, 0.0, 0.5])
    a3 = np.array([0.0, 0.5, 0.5])
    
    # FCC offset points within unit cell
    offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.0, 0.5, 0.5])
    ]
    
    points = []
    # Generate a range of lattice points that cover the cube
    for offset in offsets:
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    p = i * a1 + j * a2 + k * a3 + offset
                    if np.all(p >= -0.5 - 1e-9) and np.all(p <= 0.5 + 1e-9):
                        p_clamped = np.clip(p, -0.49, 0.49)
                        points.append(p_clamped)
    
    points = np.array(points)
    
    if len(points) == 0:
        # Fallback: simple cubic grid
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    if len(points) < N:
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # For each candidate point, compute its "clearance": 
    # min over all other points of distance/2, and min over boundaries
    # Select N points that maximize the minimum clearance
    # We'll do a greedy selection: pick the point with highest min clearance, 
    # then remove it, repeat.
    
    # Actually, a simpler approach: compute for each point its distance to 
    # boundary and its nearest neighbor distance, then rank.
    # But the true objective depends on all 100 together.
    # Let's just pick the N points that are most "interior" and well-separated.
    
    # Compute boundary distance for each point
    d_boundary = np.min(0.5 - np.abs(points), axis=1)
    
    # Compute minimum pairwise distance for each point
    n_pts = len(points)
    min_pair_dist = np.full(n_pts, np.inf)
    for i in range(n_pts):
        if n_pts - i - 1 > 0:
            diffs = points[i+1:] - points[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            if len(dists) > 0:
                min_pair_dist[i] = np.min(dists)
                min_pair_dist[i+1:] = np.minimum(min_pair_dist[i+1:], dists)
    
    # Clearance for each point: min(boundary_dist, min_pair_dist / 2)
    clearance = np.minimum(d_boundary, min_pair_dist / 2.0)
    
    # Select top N points by clearance
    if n_pts > N:
        idx = np.argsort(-clearance)[:N]
        points = points[idx]
    
    return points[:N].astype(np.float64)


def _generalized_gradient_step(centres, step_size, beta=200.0):
    """
    One step of gradient ascent on the soft-min surrogate objective.
    Uses the smooth soft-min gradient for better navigation.
    Returns new centres and the current radius.
    """
    n = len(centres)
    
    # Compute the soft-min value and gradient
    soft_val, grad = _soft_min_value_and_grad(centres, beta=beta)
    
    # Also compute the hard radius for tracking
    d_boundary = 0.5 - np.abs(centres)
    min_boundary_val = np.min(d_boundary)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_val = np.min(dists)
    r = min(min_pair_val / 2.0, min_boundary_val)
    
    # Normalize the gradient to avoid steps that are too large
    norm_grad = np.linalg.norm(grad)
    if norm_grad < 1e-12:
        return centres, r
    
    # Adaptive step: use line search-like scaling
    # Scale step so that we don't overshoot the minimum constraint
    max_move = 0.01  # Max displacement per step
    scaled_step = min(step_size, max_move / norm_grad)
    
    # Step in the ascent direction
    new_centres = centres + scaled_step * grad
    # Clip to cube with small margin to stay strictly inside
    new_centres = np.clip(new_centres, -0.4999, 0.4999)
    
    return new_centres, r


def _active_set_polish(centres, max_iter=500, tol=1e-14):
    """
    High-precision polish using active-set method.
    Identifies the most critical constraints and optimizes only those.
    """
    n = len(centres)
    centres = centres.copy()
    
    for iteration in range(max_iter):
        # Compute all constraint values
        d_boundary = 0.5 - np.abs(centres)  # shape (n, 3)
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        p_vals = dists / 2.0
        
        # Find the minimum constraint
        b_min = np.min(d_boundary)
        p_min = np.min(p_vals)
        
        if b_min < p_min:
            # Boundary is binding
            min_val = b_min
            # Find which boundary constraints are near-minimal
            active_b = np.argwhere(d_boundary < b_min + 1e-10)
        else:
            # Pairwise is binding
            min_val = p_min
            active_b = np.array([[]])
            # Find which pairwise constraints are near-minimal
            active_p = np.argwhere(p_vals < p_min + 1e-10)
            # Only upper triangle
            active_p = active_p[active_p[:, 0] < active_p[:, 1]]
        
        # If we've converged (min is very large or no improvement possible)
        if min_val > 0.1113:  # Better than known best
            break
        
        # Build a small QP or use gradient step on active constraints only
        # For simplicity, do a small gradient step focusing on active constraints
        
        # Compute gradient contribution from active constraints only
        grad = np.zeros_like(centres)
        
        if b_min < p_min:
            # Active boundary constraints
            for i, k in active_b:
                sign = np.sign(centres[i, k])
                if sign == 0:
                    sign = 1.0
                grad[i, k] += -sign
            # Normalize
            norm = np.linalg.norm(grad)
            if norm > 1e-15:
                step = 1e-5
                centres += step * grad / norm
        else:
            # Active pairwise constraints
            for i, j in active_p:
                d = dists[i, j]
                if d < 1e-12:
                    continue
                unit = (centres[i] - centres[j]) / d
                grad[i] += unit / 2.0
                grad[j] -= unit / 2.0
            # Normalize
            norm = np.linalg.norm(grad)
            if norm > 1e-15:
                step = 1e-5
                centres += step * grad / norm
        
        # Clip to cube
        centres = np.clip(centres, -0.4999999, 0.4999999)
        
        # Check convergence
        if iteration > 10:
            new_r = _compute_radius(centres)
            if abs(new_r - min_val) < tol:
                break
    
    return centres


def _multi_start_refine(best_centres, best_radius, deadline, n_starts=3):
    """
    Try multiple random restarts from perturbed versions of the best solution.
    """
    for start_idx in range(n_starts):
        if time.time() > deadline - 10.0:
            break
        
        # Create a perturbed version
        kick = 0.03 * (1.0 - start_idx * 0.3)
        new_centres = best_centres + kick * np.random.randn(*best_centres.shape)
        new_centres = np.clip(new_centres, -0.4999, 0.4999)
        
        # Run a short optimization
        step_size = 0.005
        beta = 150.0
        for _ in range(100):
            if time.time() > deadline - 10.0:
                break
            new_centres, _ = _generalized_gradient_step(new_centres, step_size, beta=beta)
        
        r = _compute_radius(new_centres)
        if r > best_radius:
            best_centres = new_centres.copy()
            best_radius = r
    
    return best_centres, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initialize with FCC lattice
    centres = _fcc_lattice_init()
    
    # Ensure valid
    if centres.shape != (N, 3):
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Phase 1: Coarse optimization with soft-min gradient
    # Start with moderate beta (smoother), increase over time
    step_size = 0.03
    beta = 30.0  # Start smoother
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() < deadline - 70.0:
        new_centres, r = _generalized_gradient_step(centres, step_size, beta=beta)
        new_r = _compute_radius(new_centres)
        
        if new_r > best_radius + 1e-12:
            best_radius = new_r
            best_centres = new_centres.copy()
            centres = new_centres
            no_improve_count = 0
            # Slowly decrease step size and increase beta (sharper)
            step_size *= 0.9995
            beta *= 1.003
        else:
            centres = new_centres  # Still move, might escape local min
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        if step_size < 1e-5:
            break
    
    # Phase 2: Fine optimization with sharper soft-min
    step_size = 0.005
    beta = 100.0
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline - 40.0:
        new_centres, r = _generalized_gradient_step(centres, step_size, beta=beta)
        new_r = _compute_radius(new_centres)
        
        if new_r > best_radius + 1e-12:
            best_radius = new_r
            best_centres = new_centres.copy()
            centres = new_centres
            no_improve_count = 0
            step_size *= 0.999
            beta *= 1.005
        else:
            centres = new_centres
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        if step_size < 1e-6:
            break
    
    # Phase 3: Multi-start refinement to escape local optima
    best_centres, best_radius = _multi_start_refine(best_centres, best_radius, deadline, n_starts=5)
    
    # Phase 4: Active-set polish for high precision
    if time.time() < deadline - 5.0:
        polished = _active_set_polish(best_centres, max_iter=200, tol=1e-15)
        polished_r = _compute_radius(polished)
        if polished_r > best_radius:
            best_centres = polished
            best_radius = polished_r
    
    # Final validation: ensure all points are strictly inside the cube
    best_centres = np.clip(best_centres, -0.4999999, 0.4999999)
    
    # Final radius check
    final_radius = _compute_radius(best_centres)
    
    return best_centres
# EVOLVE-BLOCK-END
