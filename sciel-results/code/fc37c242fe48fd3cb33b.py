"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """Compute the actual packing radius for centres c."""
    if c.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Boundary constraint
    boundary = 0.5 - np.max(np.abs(c), axis=1)
    min_boundary = np.min(boundary)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_boundary)

def _softmin_objective_and_grad(x, beta):
    """
    Objective: minimize -softmin_radius where softmin uses log-sum-exp.
    x: (3*N,) array of centres
    Returns (f, g) where f is objective value, g is gradient.
    """
    c = x.reshape(N, 3)
    
    # Compute all pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    boundary = 0.5 - np.abs(c)  # (N, 3)
    
    # Soft-min of pairwise distances: r_pair = -1/beta * log(sum(exp(-beta * dists / 2)))
    # We want to maximize the minimum of (dists/2, boundary), so minimize negative soft-min
    # r_soft = softmin(dists/2, boundary)
    
    # For efficiency, compute soft-min of all constraints
    # Constraints: dists[i,j]/2 for i<j, and boundary[i,k]
    # Let's build a vector of all constraint values
    
    # Pairwise: take upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0  # (n_pairs,)
    
    # Boundary: flatten
    bound_vals = boundary.flatten()  # (3*N,)
    
    # All constraints
    all_con = np.concatenate([pair_dists, bound_vals])
    
    # Soft-min: -1/beta * log(sum(exp(-beta * con)))
    # For numerical stability
    m = np.min(all_con)
    exp_terms = np.exp(-beta * (all_con - m))
    log_sum = np.log(np.sum(exp_terms))
    r_soft = m - np.log(np.sum(exp_terms)) / beta
    
    # Objective: minimize -r_soft
    f = -r_soft
    
    # Gradient
    # d(r_soft)/d(con_k) = exp(-beta*(con_k - m)) / sum(exp(-beta*(con_j - m)))
    # = exp(-beta*(con_k - m)) / exp_terms.sum()
    weights = exp_terms / np.sum(exp_terms)  # (n_all,)
    
    # d(con_k)/dx
    # For pairwise k corresponding to (i,j): con = |c_i - c_j| / 2
    # d(con)/d(c_i) = -(c_i - c_j) / (2 * |c_i - c_j|) = -(c_i - c_j) / (2 * dists[i,j])
    # d(con)/d(c_j) = (c_i - c_j) / (2 * dists[i,j])
    
    grad = np.zeros_like(x)
    
    # Pairwise contributions
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = weights[idx]
        d = dists[i, j]
        if d > 1e-12:
            diff_ij = c[i] - c[j]
            # d(con)/d(c_i) = -diff_ij / (2*d)
            # d(con)/d(c_j) = diff_ij / (2*d)
            grad[3*i:3*i+3] += -w * diff_ij / (2.0 * d)
            grad[3*j:3*j+3] += w * diff_ij / (2.0 * d)
    
    # Boundary contributions
    # con = 0.5 - |c_ik|
    # d(con)/d(c_ik) = -sign(c_ik)
    for i in range(N):
        for k in range(3):
            idx = len(pair_dists) + i * 3 + k
            w = weights[idx]
            sign_val = np.sign(c[i, k])
            if sign_val == 0:
                sign_val = 1.0  # subgradient, pick any
            grad[3*i + k] += -w * sign_val
    
    # Objective gradient: d(-r_soft)/dx = -d(r_soft)/dx = -grad
    g = -grad
    
    return f, g

def _generate_grid_start():
    """Generate a jittered grid initial configuration."""
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.seed(42)
    idx = np.random.choice(len(grid), N, replace=False)
    c = grid[idx].copy()
    c += np.random.uniform(-0.05, 0.05, size=(N, 3))
    c = np.clip(c, -0.45, 0.45)
    return c.flatten()

def _generate_random_start(seed):
    """Generate a random initial configuration."""
    np.random.seed(seed)
    c = np.random.uniform(-0.4, 0.4, size=(N, 3))
    return c.flatten()

def _generate_fcc_start():
    """Generate an FCC-like lattice initial configuration."""
    # FCC lattice points in unit cube
    a = 0.5  # lattice constant for ~50 points
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x = i * a + 0.1
                y = j * a + 0.1
                z = k * a + 0.1
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    points.append((x, y, z))
                # FCC basis
                x2 = i * a + 0.1 + a/2
                y2 = j * a + 0.1 + a/2
                z2 = k * a + 0.1
                if -0.5 <= x2 <= 0.5 and -0.5 <= y2 <= 0.5 and -0.5 <= z2 <= 0.5:
                    points.append((x2, y2, z2))
                x3 = i * a + 0.1 + a/2
                y3 = j * a + 0.1
                z3 = k * a + 0.1 + a/2
                if -0.5 <= x3 <= 0.5 and -0.5 <= y3 <= 0.5 and -0.5 <= z3 <= 0.5:
                    points.append((x3, y3, z3))
                x4 = i * a + 0.1
                y4 = j * a + 0.1 + a/2
                z4 = k * a + 0.1 + a/2
                if -0.5 <= x4 <= 0.5 and -0.5 <= y4 <= 0.5 and -0.5 <= z4 <= 0.5:
                    points.append((x4, y4, z4))
    
    points = np.array(points)
    if len(points) >= N:
        np.random.seed(789)
        idx = np.random.choice(len(points), N, replace=False)
        c = points[idx].copy()
        c += np.random.uniform(-0.02, 0.02, size=(N, 3))
        c = np.clip(c, -0.45, 0.45)
        return c.flatten()
    else:
        return _generate_random_start(789)

def _optimize_from_initial(x0, start_time, deadline):
    """Run L-BFGS-B optimization from initial configuration x0."""
    best_r = 0.0
    best_c = None
    
    # Continuation schedule: start with small beta, increase
    betas = [50, 100, 200, 500, 1000, 2000, 5000]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        def obj_func(x):
            f, g = _softmin_objective_and_grad(x, beta)
            return f, g
        
        res = minimize(
            obj_func,
            x0,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (3 * N),
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        c = res.x.reshape(N, 3)
        r = _compute_radius(c)
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Use result as next initial
        x0 = res.x
    
    return best_r, best_c

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_r = 0.0
    best_c = None
    
    # Multiple initial configurations
    starts = [
        _generate_grid_start(),
        _generate_random_start(123),
        _generate_random_start(456),
        _generate_fcc_start(),
        _generate_random_start(789),
    ]
    
    for x0 in starts:
        if time.time() > deadline - 15:
            break
        
        try:
            r, c = _optimize_from_initial(x0, start_time, deadline)
            if r > best_r:
                best_r = r
                best_c = c
        except Exception:
            continue
    
    if best_c is None:
        # Fallback
        return _generate_grid_start().reshape(N, 3)
    
    return best_c

# EVOLVE-BLOCK-END
