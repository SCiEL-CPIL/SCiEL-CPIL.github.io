"""Pack 50 equal circles in a unit square, maximizing the common radius."""
import numpy as np
import time

N = 50
HALF = 0.5

def _radius(c):
    """Compute the actual (non-smooth) radius for a configuration."""
    n = len(c)
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((c[i] - c[j]) ** 2))
            if d < min_dist:
                min_dist = d
    # Wall distances
    for i in range(n):
        for k in range(2):
            d = HALF - abs(c[i, k])
            if d < min_dist:
                min_dist = d
    return min_dist / 2.0

def _soft_min_radius(c, beta=100.0):
    """
    Smooth surrogate: -soft_min of all constraints.
    We want to maximize r = min(all pairwise distances/2, all wall distances).
    Equivalently minimize -r, i.e., maximize soft_min.
    
    soft_min(x_1,...,x_n) = -(1/beta) * log(sum(exp(-beta*x_i)))
    
    We compute: r_soft = soft_min of {dist_ij/2 for i<j, wall_dist_i for i}
    """
    n = len(c)
    # Collect all constraints: pairwise distances / 2 and wall distances
    # For efficiency, compute in bulk where possible
    
    # Pairwise distances
    # c is (n, 2)
    # Compute all pairwise squared distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = np.zeros((n, 4))
    wall_dists[:, 0] = HALF - c[:, 0]  # right wall
    wall_dists[:, 1] = HALF + c[:, 0]  # left wall
    wall_dists[:, 2] = HALF - c[:, 1]  # top wall
    wall_dists[:, 3] = HALF + c[:, 1]  # bottom wall
    
    # All constraint values (each is a "distance" that r must be <= to)
    # Pairwise: dists[i,j] for i<j, each contributes dist/2 to r
    # Wall: wall_dists[i,k], each contributes wall_dist to r
    # r = min( min_{i<j} dists[i,j]/2, min_i min_k wall_dists[i,k] )
    
    # For soft_min, we need all the "radius candidates"
    # r_candidates = [dists[i,j]/2 for i<j] + [wall_dists[i,k] for i,k]
    
    # To avoid building a huge array, compute log-sum-exp in parts
    # Let's build the array of all candidates
    # Number of pairwise: n*(n-1)/2
    # Number of wall: n*4
    npair = n * (n - 1) // 2
    nwall = n * 4
    total = npair + nwall
    
    # Extract upper triangle of dists
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu] / 2.0  # shape (npair,)
    wall_vals = wall_dists.flatten()  # shape (nwall,)
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # soft_min = -(1/beta) * log(sum(exp(-beta * x)))
    # For numerical stability:
    min_val = np.min(all_vals)
    exp_args = -beta * (all_vals - min_val)
    # Clip to avoid overflow
    exp_args = np.clip(exp_args, -500, 500)
    s = np.sum(np.exp(exp_args))
    soft_min_val = min_val - (1.0 / beta) * np.log(s)
    
    return soft_min_val

def _soft_min_radius_grad(c, beta=100.0):
    """
    Compute soft_min radius and its gradient w.r.t. c.
    """
    n = len(c)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = np.zeros((n, 4))
    wall_dists[:, 0] = HALF - c[:, 0]
    wall_dists[:, 1] = HALF + c[:, 0]
    wall_dists[:, 2] = HALF - c[:, 1]
    wall_dists[:, 3] = HALF + c[:, 1]
    
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu] / 2.0
    wall_vals = wall_dists.flatten()
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    min_val = np.min(all_vals)
    exp_args = -beta * (all_vals - min_val)
    exp_args = np.clip(exp_args, -500, 500)
    exp_vals = np.exp(exp_args)
    s = np.sum(exp_vals)
    weights = exp_vals / s  # softmax weights
    
    soft_min_val = min_val - (1.0 / beta) * np.log(s)
    
    # Gradient: d(soft_min)/dx_i = -sum_j w_j * dx_j/dx_i
    # where x_j are the constraint values
    
    grad = np.zeros_like(c)
    
    # Pairwise contributions
    # For pair (i,j), val = dist(i,j)/2
    # d(val)/dc_i = (c_i - c_j) / dist(i,j)
    # d(val)/dc_j = (c_j - c_i) / dist(i,j)
    
    # We need to map back from pair_vals to indices
    # iu[0][k] and iu[1][k] give the pair index for pair_vals[k]
    w_pair = weights[:len(pair_vals)]
    
    for k in range(len(pair_vals)):
        i, j = iu[0][k], iu[1][k]
        d = dists[i, j]
        if d < 1e-12:
            continue
        w = w_pair[k]
        # d(pair_val)/dc_i = (c_i - c_j) / d
        # d(pair_val)/dc_j = (c_j - c_i) / d
        # soft_min = -1/beta * log(sum exp(-beta * x))
        # d(soft_min)/dc_i = sum_k w_k * d(x_k)/dc_i  ... wait
        # Actually: soft_min = -(1/beta) log(sum exp(-beta x_k))
        # d(soft_min)/dx_k = w_k (where w_k = exp(-beta x_k)/sum)
        # So d(soft_min)/dc_i = sum_k w_k * dx_k/dc_i
        
        vec = (c[i] - c[j]) / d
        grad[i] += w * vec
        grad[j] -= w * vec
    
    # Wall contributions
    w_wall = weights[len(pair_vals):]
    # wall_vals is flattened from wall_dists (n, 4)
    # For circle i, wall 0: val = HALF - c[i,0], d/dc[i,0] = -1
    # wall 1: val = HALF + c[i,0], d/dc[i,0] = 1
    # wall 2: val = HALF - c[i,1], d/dc[i,1] = -1
    # wall 3: val = HALF + c[i,1], d/dc[i,1] = 1
    
    for i in range(n):
        w0 = w_wall[i * 4 + 0]
        w1 = w_wall[i * 4 + 1]
        w2 = w_wall[i * 4 + 2]
        w3 = w_wall[i * 4 + 3]
        grad[i, 0] += (-w0 + w1)
        grad[i, 1] += (-w2 + w3)
    
    return soft_min_val, grad

def _optimize(c0, max_iters=5000, lr_init=0.001, beta_init=50.0, beta_final=200.0, deadline=None):
    """
    Gradient ascent on soft_min_radius to maximize it.
    Uses adaptive learning rate and annealed beta.
    """
    c = c0.copy()
    n = len(c)
    
    best_c = c.copy()
    best_r = _radius(c)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Anneal beta
        frac = it / max_iters
        beta = beta_init + (beta_final - beta_init) * frac
        
        # Adaptive learning rate: start small, can increase if improving
        lr = lr_init * (1.0 + 0.5 * np.sin(frac * np.pi))  # slight modulation
        
        r_val, grad = _soft_min_radius_grad(c, beta)
        
        # Gradient ascent
        c_new = c + lr * grad
        
        # Clamp to square with small margin to keep valid
        margin = 0.01
        c_new[:, 0] = np.clip(c_new[:, 0], -HALF + margin, HALF - margin)
        c_new[:, 1] = np.clip(c_new[:, 1], -HALF + margin, HALF - margin)
        
        # Check if this is an improvement
        r_new = _radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
        
        # Accept if the soft objective improved (even if true radius didn't, to keep moving)
        # Actually, let's accept based on true radius for safety, but use soft for gradient
        if r_new >= r_val - 0.001:  # allow some slack
            c = c_new
        else:
            # Try smaller step
            lr_small = lr * 0.1
            c_try = c + lr_small * grad
            c_try[:, 0] = np.clip(c_try[:, 0], -HALF + margin, HALF - margin)
            c_try[:, 1] = np.clip(c_try[:, 1], -HALF + margin, HALF - margin)
            r_try = _radius(c_try)
            if r_try > best_r:
                best_r = r_try
                best_c = c_try.copy()
            c = c_try
        
    return best_c, best_r

def _local_search(c, max_iters=2000, deadline=None):
    """
    Simple local search: try random perturbations, accept if improves radius.
    """
    n = len(c)
    best_c = c.copy()
    best_r = _radius(c)
    
    step_size = 0.02
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random subset of points to perturb
        n_perturb = max(1, n // 10)
        idx = np.random.choice(n, size=n_perturb, replace=False)
        
        c_new = best_c.copy()
        for i in idx:
            delta = np.random.randn(2) * step_size
            c_new[i] += delta
            c_new[i, 0] = np.clip(c_new[i, 0], -HALF + 0.005, HALF - 0.005)
            c_new[i, 1] = np.clip(c_new[i, 1], -HALF + 0.005, HALF - 0.005)
        
        r_new = _radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            # Occasionally increase step
            if it % 500 == 0:
                step_size *= 1.2
        else:
            # Decrease step slightly
            step_size *= 0.99
        
        step_size = max(step_size, 0.001)
    
    return best_c, best_r

def _refine(c, deadline=None):
    """
    Final refinement: alternate between gradient steps and local search with decreasing step.
    """
    best_c = c.copy()
    best_r = _radius(c)
    
    # Phase 1: Gradient-based with moderate beta
    c1, r1 = _optimize(best_c, max_iters=2000, lr_init=0.0005, beta_init=100.0, beta_final=300.0, deadline=deadline)
    if r1 > best_r:
        best_r = r1
        best_c = c1
    
    # Phase 2: Local search with small steps
    c2, r2 = _local_search(best_c, max_iters=1000, deadline=deadline)
    if r2 > best_r:
        best_r = r2
        best_c = c2
    
    # Phase 3: Another gradient pass with high beta
    c3, r3 = _optimize(best_c, max_iters=1000, lr_init=0.0002, beta_init=200.0, beta_final=500.0, deadline=deadline)
    if r3 > best_r:
        best_r = r3
        best_c = c3
    
    return best_c, best_r

def _initial_grid():
    """Standard grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)

def _initial_random():
    """Random initialization inside the square."""
    c = np.random.uniform(-HALF + 0.02, HALF - 0.02, size=(N, 2))
    return c

def _initial_jittered_grid():
    """Grid with random jitter."""
    c = _initial_grid()
    c += np.random.randn(N, 2) * 0.01
    c[:, 0] = np.clip(c[:, 0], -HALF + 0.01, HALF - 0.01)
    c[:, 1] = np.clip(c[:, 1], -HALF + 0.01, HALF - 0.01)
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_c = None
    best_r = -np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 15
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            c0 = _initial_grid()
        elif restart == 1:
            c0 = _initial_jittered_grid()
        elif restart == 2:
            c0 = _initial_random()
        else:
            # Mix: sometimes grid-based, sometimes random
            if restart % 2 == 0:
                c0 = _initial_jittered_grid()
            else:
                c0 = _initial_random()
        
        # Quick optimization
        c_opt, r_opt = _optimize(c0, max_iters=1500, lr_init=0.001, beta_init=50.0, beta_final=150.0, deadline=deadline)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
        
        # If we have time, refine the best
        if time.time() < deadline - 30.0:
            c_ref, r_ref = _refine(best_c, deadline=deadline)
            if r_ref > best_r:
                best_r = r_ref
                best_c = c_ref.copy()
    
    # Final check: ensure validity
    if best_c is None:
        best_c = _initial_grid()
    
    # Validate
    r_final = _radius(best_c)
    if not np.isfinite(r_final) or r_final <= 0:
        best_c = _initial_grid()
    
    return best_c

# EVOLVE-BLOCK-END
