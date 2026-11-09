"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _score(c):
    """Compute the radius for a configuration of centers."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside cube [-0.5, 0.5]^3
    if np.any(np.abs(c) > 0.5 + 1e-12):
        return 0.0
    # Min distance to boundary
    r_boundary = np.min(0.5 - np.abs(c))
    # Min pairwise distance / 2
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    r_pair = np.min(d) / 2.0
    return min(r_boundary, r_pair)


def _random_initial():
    """Generate random initial configuration inside the cube."""
    return np.random.uniform(-0.5, 0.5, size=(N, 3))


def _optimize(c0, max_iter=5000, lr_start=0.01, lr_end=0.0001, deadline=None):
    """
    Optimize configuration using gradient ascent on a smooth surrogate.
    Uses soft-min to approximate the min of constraints.
    """
    c = c0.copy()
    n = c.shape[0]
    
    # Precompute index pairs
    iu, ju = np.triu_indices(n, k=1)
    num_pairs = len(iu)
    
    best_c = c.copy()
    best_score = _score(c)
    
    t = 0.0
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute pairwise differences
        diff = c[iu] - c[ju]
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, 1e-15)
        
        # Boundary distances
        bound = 0.5 - np.abs(c)  # shape (n, 3)
        min_bound = np.min(bound, axis=1)
        
        # Use soft-min with temperature beta
        beta = 50.0 + it * 0.05  # increasing sharpness
        
        # Soft-min of pairwise distances
        # We want to maximize min(dist/2, min_bound)
        # Equivalently, minimize -min(...)
        # Use log-sum-exp for soft-min: soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
        
        # Pairwise constraint: dist/2
        x_pair = dist / 2.0
        # Boundary constraint: min_bound
        x_bound = min_bound
        
        # Combine: we want soft min of all constraints
        # Let's work with negative values for log-sum-exp
        # soft_min(x) = -logsumexp(-beta * x) / beta
        
        # For pairwise
        neg_x_pair = -beta * x_pair
        max_neg = np.max(neg_x_pair)
        log_sum_exp_pair = max_neg + np.log(np.sum(np.exp(neg_x_pair - max_neg)))
        soft_min_pair = -log_sum_exp_pair / beta
        
        # For boundary (per point, then min over points)
        # Actually, the boundary constraint is min over i and k of (0.5 - |c_ik|)
        # Let's compute all boundary values
        bound_all = 0.5 - np.abs(c).flatten()
        neg_x_bound = -beta * bound_all
        max_neg_b = np.max(neg_x_bound)
        log_sum_exp_bound = max_neg_b + np.log(np.sum(np.exp(neg_x_bound - max_neg_b)))
        soft_min_bound = -log_sum_exp_bound / beta
        
        # Overall soft min
        # soft_min(soft_min_pair, soft_min_bound)
        # This is getting complex. Let's use a different approach.
        
        # Simpler: use the actual min but with subgradient
        # Or use a penalty approach
        
        # Let's try a direct gradient ascent with the actual min
        # Compute current min
        r_pair = np.min(x_pair)
        r_bound = np.min(x_bound)
        r = min(r_pair, r_bound)
        
        # Gradient of r with respect to c
        grad = np.zeros_like(c)
        
        if r_pair < r_bound - 1e-10:
            # Pairwise constraint is active
            # For each pair with dist/2 close to r_pair, add gradient
            mask = np.abs(x_pair - r_pair) < 0.01
            if np.any(mask):
                for idx in np.where(mask)[0]:
                    i, j = iu[idx], ju[idx]
                    d = diff[idx] / dist[idx]  # unit vector from j to i
                    # To increase distance, move i away from j and j away from i
                    grad[i] += d
                    grad[j] -= d
        elif r_bound < r_pair - 1e-10:
            # Boundary constraint is active
            # For each point with min_bound close to r_bound, push inward
            mask = np.abs(x_bound - r_bound) < 0.01
            if np.any(mask):
                for i in np.where(mask)[0]:
                    # Find which coordinate is closest to boundary
                    b_i = 0.5 - np.abs(c[i])
                    min_k = np.argmin(b_i)
                    # Push away from boundary: if c[i,k] > 0, decrease it; if < 0, increase it
                    if c[i, min_k] > 0:
                        grad[i, min_k] -= 1.0
                    else:
                        grad[i, min_k] += 1.0
        else:
            # Both constraints are active
            mask_p = np.abs(x_pair - r) < 0.01
            mask_b = np.abs(x_bound - r) < 0.01
            if np.any(mask_p):
                for idx in np.where(mask_p)[0]:
                    i, j = iu[idx], ju[idx]
                    d = diff[idx] / dist[idx]
                    grad[i] += d
                    grad[j] -= d
            if np.any(mask_b):
                for i in np.where(mask_b)[0]:
                    b_i = 0.5 - np.abs(c[i])
                    min_k = np.argmin(b_i)
                    if c[i, min_k] > 0:
                        grad[i, min_k] -= 1.0
                    else:
                        grad[i, min_k] += 1.0
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad /= grad_norm
        
        # Learning rate schedule
        lr = lr_start * (lr_end / lr_start) ** (it / max_iter)
        
        # Update
        c_new = c + lr * grad
        # Project back into cube
        c_new = np.clip(c_new, -0.5, 0.5)
        
        # Check if improved
        new_score = _score(c_new)
        if new_score > best_score:
            best_c = c_new.copy()
            best_score = new_score
        
        c = c_new
    
    return best_c, best_score


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    best_c = None
    best_score = 0.0
    
    # Try multiple random restarts
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        c0 = _random_initial()
        c_opt, score_opt = _optimize(c0, max_iter=3000, deadline=deadline - 10.0)
        
        if score_opt > best_score:
            best_score = score_opt
            best_c = c_opt
    
    if best_c is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
    
    return best_c
# EVOLVE-BLOCK-END
