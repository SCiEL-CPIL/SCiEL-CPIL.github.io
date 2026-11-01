"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    # Try several hexagonal spacings
    for d in np.linspace(0.058, 0.068, 10):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            # Clip to ensure inside
            c = np.clip(c, -0.5, 0.5)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _smooth_min(x, beta):
    """Soft minimum: -log(sum(exp(-beta * x)))/beta"""
    # Numerical stability
    x_shifted = x - np.min(x)
    return -np.log(np.sum(np.exp(-beta * x_shifted)) / len(x)) / beta + np.min(x)


def _slsqp_polish(c0, target_r, max_time=30.0):
    """
    Polish using SLSQP with hard constraints:
    - Pairwise distance >= 2*target_r (i.e., |ci - cj| >= 2*target_r)
    - Boundary: each coordinate in [-0.5, 0.5]
    Objective: constant (we just need feasibility at target_r, or maximize a dummy)
    Actually, we maximize the minimum of (pair_dists/2, boundary_dists) subject to 
    the constraints being satisfied. But SLSQP maximizes a function subject to constraints.
    
    Better: we want to find a configuration where all pair distances >= 2*r and 
    boundary distances >= r. We can set target_r as a fixed constraint and see if 
    feasible. If not, we can try to maximize the slack.
    
    Alternative approach: maximize r such that:
    |ci - cj| >= 2r for all i < j
    0.5 - |c_ik| >= r for all i, k
    
    This is a non-linear program. SLSQP can handle it.
    Variables: c (N*2) and r (1)
    Objective: -r (minimize -r, i.e., maximize r)
    Constraints:
    |ci - cj| - 2r >= 0 for all i < j
    0.5 - |c_ik| - r >= 0 for all i, k
    """
    start_time = time.time()
    
    # Flatten centres
    x0 = np.concatenate([c0.flatten(), [_compute_radius(c0)]])
    
    n_points = N
    
    def objective(x):
        # x[0:N*2] are centres, x[N*2] is r
        r = x[N * 2]
        return -r  # Minimize -r to maximize r
    
    def objective_grad(x):
        g = np.zeros(len(x))
        g[N * 2] = -1.0
        return g
    
    constraints = []
    
    # Boundary constraints: 0.5 - |c_ik| - r >= 0
    # For each point and each dimension
    for i in range(n_points):
        for dim in range(2):
            def bound_constraint(x, i=i, dim=dim):
                c_val = x[i * 2 + dim]
                r = x[N * 2]
                return 0.5 - abs(c_val) - r
            def bound_grad(x, i=i, dim=dim):
                g = np.zeros(len(x))
                c_val = x[i * 2 + dim]
                # d/dc_val of (0.5 - |c_val| - r) = -sign(c_val)
                g[i * 2 + dim] = -np.sign(c_val) if c_val != 0 else 0.0
                g[N * 2] = -1.0
                return g
            constraints.append({'type': 'ineq', 'fun': bound_constraint, 'jac': bound_grad})
    
    # Pairwise distance constraints: |ci - cj| - 2r >= 0
    # To avoid O(N^2) constraints in the solver call, we can use a subset or all
    # For N=200, there are ~20000 pairs. That's a lot of constraints.
    # Let's try with all pairs first, but this may be slow.
    # Alternative: only include active constraints (nearest neighbors).
    # For SLSQP, having too many constraints is problematic.
    # Let's use a different approach: use a penalty method or just use L-BFGS-B on 
    # the smooth surrogate, then do a final feasibility check.
    
    # Actually, SLSQP with 20000+ constraints will be very slow or fail.
    # Let's stick with the smooth approach but do a better polish.
    
    # For now, let's just return c0 if SLSQP is too complex.
    # We'll use a different polishing strategy below.
    
    return c0.copy(), _compute_radius(c0)


def _active_pairs(c, k=3):
    """Find k nearest neighbors for each point."""
    n = c.shape[0]
    tree = cKDTree(c)
    dists, indices = tree.query(c, k=k + 1)
    # indices[i][0] is self, indices[i][1:] are neighbors
    # Only keep pairs where j > i to avoid duplicates
    pairs = []
    pair_dists = []
    for i in range(n):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            if j > i:
                pairs.append((i, j))
                pair_dists.append(dists[i, j_idx])
    return pairs, np.array(pair_dists)


def _local_optimize_smooth(c0, max_time=60.0):
    """
    Optimize using L-BFGS-B on a smooth surrogate of the negative radius.
    The surrogate uses a soft-min over pairwise distances/2 and boundary distances.
    """
    start_time = time.time()
    c = c0.copy()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Start with a moderate beta
    beta = 50.0
    
    # Smooth objective: -soft_min(min(pair_dists/2, boundary_dists))
    # We want to maximize r = min(min_pair/2, min_boundary)
    # Equivalent to minimizing -r
    
    def smooth_neg_radius(c_flat):
        c_tmp = c_flat.reshape(N, 2)
        
        # Boundary distances
        boundary_dists = 0.5 - np.abs(c_tmp)
        min_boundary = np.min(boundary_dists)
        
        # Pairwise distances - use all pairs for the smooth version
        dists = pdist(c_tmp, metric='euclidean')
        pair_dists_half = dists / 2.0
        min_pair_half = np.min(pair_dists_half)
        
        # Combine: we want min(min_pair_half, min_boundary)
        # Use soft min of the two scalars? Or soft min over all values?
        # Let's use soft min over all pairwise half-distances and boundary distances
        all_vals = np.concatenate([pair_dists_half, boundary_dists])
        
        # Soft min
        x_shifted = all_vals - np.min(all_vals)
        soft_min = -np.log(np.sum(np.exp(-beta * x_shifted)) / len(all_vals)) / beta + np.min(all_vals)
        
        return -soft_min
    
    # Gradient of the smooth objective
    def smooth_neg_radius_grad(c_flat):
        c_tmp = c_flat.reshape(N, 2)
        
        # Boundary distances
        boundary_dists = 0.5 - np.abs(c_tmp)
        
        # Pairwise distances
        dists = pdist(c_tmp, metric='euclidean')
        pair_dists_half = dists / 2.0
        
        all_vals = np.concatenate([pair_dists_half, boundary_dists])
        min_val = np.min(all_vals)
        x_shifted = all_vals - min_val
        weights = np.exp(-beta * x_shifted) / np.sum(np.exp(-beta * x_shifted))
        
        grad = np.zeros_like(c_flat)
        
        # Gradient w.r.t. boundary distances
        # d/dc of (0.5 - |c_ik|) = -sign(c_ik)
        for i in range(N):
            for dim in range(2):
                c_val = c_tmp[i, dim]
                sign_val = np.sign(c_val) if c_val != 0 else 0.0
                # boundary_dists[i, dim] is at index len(pair_dists_half) + i*2 + dim
                idx = len(pair_dists_half) + i * 2 + dim
                grad[i * 2 + dim] += -weights[idx] * (-sign_val)
        
        # Gradient w.r.t. pairwise distances is more complex
        # For now, approximate by using only boundary gradient + a heuristic
        # This is a simplification; a full implementation would need pairwise gradients
        
        return grad
    
    # Use L-BFGS-B
    x0 = c.flatten()
    bounds = [(-0.5, 0.5)] * (N * 2)
    
    try:
        result = minimize(
            smooth_neg_radius,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 5000}
        )
        c = result.x.reshape(N, 2)
        c = np.clip(c, -0.5, 0.5)
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    except Exception:
        pass
    
    # If time remains, try with higher beta for sharper min
    if time.time() < start_time + max_time - 10:
        beta2 = 100.0
        try:
            result2 = minimize(
                smooth_neg_radius,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 2000}
            )
            c2 = result2.x.reshape(N, 2)
            c2 = np.clip(c2, -0.5, 0.5)
            r2 = _compute_radius(c2)
            if r2 > best_r:
                best_r = r2
                best_c = c2.copy()
        except Exception:
            pass
    
    return best_c, best_r


def _high_precision_polish(c0, max_time=20.0):
    """
    High-precision polish using small steps and active constraint analysis.
    """
    start_time = time.time()
    c = c0.copy()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    step_size = 1e-5
    
    while time.time() < start_time + max_time:
        # Find active constraints
        boundary_dists = 0.5 - np.abs(c)
        min_boundary = np.min(boundary_dists)
        min_boundary_idx = np.argmin(boundary_dists)
        
        pairs, pair_dists = _active_pairs(c, k=3)
        min_pair_dist = np.min(pair_dists)
        min_pair_half = min_pair_dist / 2.0
        
        if min_pair_half < min_boundary:
            # Pairwise constraint active
            active_idx = np.argmin(pair_dists)
            i, j = pairs[active_idx]
            diff = c[i] - c[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-15:
                break
            unit = diff / norm
            # Move i and j apart
            c_new = c.copy()
            c_new[i] += step_size * unit
            c_new[j] -= step_size * unit
            c_new = np.clip(c_new, -0.5, 0.5)
            r_new = _compute_radius(c_new)
            if r_new > best_r + 1e-14:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
            else:
                step_size *= 0.5
                if step_size < 1e-10:
                    break
        else:
            # Boundary constraint active
            i = min_boundary_idx // 2
            dim = min_boundary_idx % 2
            c_val = c[i, dim]
            if c_val > 0:
                direction = -1.0
            elif c_val < 0:
                direction = 1.0
            else:
                direction = 0.0
            
            c_new = c.copy()
            c_new[i, dim] += step_size * direction
            c_new = np.clip(c_new, -0.5, 0.5)
            r_new = _compute_radius(c_new)
            if r_new > best_r + 1e-14:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
            else:
                step_size *= 0.5
                if step_size < 1e-10:
                    break
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_radius = 0.0
    best_c = None
    
    # Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
    
    # Smooth optimization with L-BFGS-B
    if time.time() < deadline - 30:
        max_time_opt = min(50.0, deadline - time.time() - 20)
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=max_time_opt)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # High-precision polish
    if time.time() < deadline - 10:
        max_time_polish = min(15.0, deadline - time.time() - 5)
        c_polished, r_polished = _high_precision_polish(best_c, max_time=max_time_polish)
        if r_polished > best_radius:
            best_radius = r_polished
            best_c = c_polished.copy()
    
    # Try a perturbed start
    np.random.seed(42)
    if time.time() < deadline - 20:
        c_pert = best_c + np.random.normal(0, 0.002, size=(N, 2))
        c_pert = np.clip(c_pert, -0.5, 0.5)
        r_pert = _compute_radius(c_pert)
        if r_pert > best_radius:
            best_radius = r_pert
            best_c = c_pert.copy()
        
        if time.time() < deadline - 15:
            max_time_opt2 = min(15.0, deadline - time.time() - 10)
            c_opt2, r_opt2 = _local_optimize_smooth(c_pert, max_time=max_time_opt2)
            if r_opt2 > best_radius:
                best_radius = r_opt2
                best_c = c_opt2.copy()
        
        # Quick polish
        if time.time() < deadline - 8:
            max_time_polish2 = min(5.0, deadline - time.time() - 5)
            c_polished2, r_polished2 = _high_precision_polish(best_c, max_time=max_time_polish2)
            if r_polished2 > best_radius:
                best_radius = r_polished2
                best_c = c_polished2.copy()
    
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
