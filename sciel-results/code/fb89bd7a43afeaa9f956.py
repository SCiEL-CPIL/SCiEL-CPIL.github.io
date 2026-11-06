"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _radius(c):
    """Compute the maximum common radius for centres c (N,3) in [-0.5,0.5]^3."""
    n = c.shape[0]
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    # Distance to walls
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    return min(0.5 * min_dist, min_wall)


def _soft_min_val_and_weights(constraints, beta):
    """Compute soft min value and weights for a set of constraints."""
    max_c = np.max(constraints)
    shifted = constraints - max_c
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = max_c - log_sum / beta
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights)
    return soft_min_val, weights


def _soft_radius(c, beta=50.0):
    """Soft surrogate for the radius, differentiable. Returns (value, grad)."""
    n = c.shape[0]
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-15)
    np.fill_diagonal(dist, np.inf)
    
    wall = 0.5 - np.abs(c)
    
    # Soft min of all constraints (pair distances / 2 and wall distances)
    # We want to maximize min(constraints)
    # Soft min: -logsumexp(-beta * constraints) / beta
    
    pair_constraints = 0.5 * dist
    wall_constraints = wall
    
    # Flatten pair constraints (upper triangle only to avoid double counting)
    iu = np.triu_indices(n, k=1)
    pc = pair_constraints[iu]
    wc = wall_constraints.flatten()
    
    all_cons = np.concatenate([pc, wc])
    
    soft_min_val, weights = _soft_min_val_and_weights(all_cons, beta)
    
    grad = np.zeros_like(c)
    
    # Pair contributions
    # For each pair (i,j), constraint is 0.5*|c_i-c_j|
    # d(soft_min)/dc_i += w_ij * 0.5 * (c_i-c_j)/|c_i-c_j|
    # d(soft_min)/dc_j += w_ij * 0.5 * (c_j-c_i)/|c_i-c_j|
    
    diff_ij = c[iu[0]] - c[iu[1]]
    dist_ij = dist[iu]
    norm_diff = diff_ij / (dist_ij + 1e-15)[:, None]
    
    w_pair = weights[:len(iu[0])]
    
    # Vectorized gradient accumulation for pairs
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = w_pair[idx]
        grad[i] += 0.5 * w * norm_diff[idx]
        grad[j] -= 0.5 * w * norm_diff[idx]
    
    # Wall contributions
    # constraint = 0.5 - |c_ik|
    # d(soft_min)/dc_ik += w_ik * (-sign(c_ik))
    w_wall = weights[len(iu[0]):]
    signs = np.sign(c)
    grad -= w_wall.reshape(n, 3) * signs
    
    return soft_min_val, grad


def _neg_soft_radius(c, beta=50.0):
    """Negative soft radius for minimization."""
    val, grad = _soft_radius(c, beta)
    return -val, -grad


def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _init_bcc():
    """Body-centered cubic initialization."""
    m = 3
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                x = (ix + 0.5) / m - 0.5
                y = (iy + 0.5) / m - 0.5
                z = (iz + 0.5) / m - 0.5
                points.append([x, y, z])
                xc = (ix + 0.5) / m - 0.5 + 0.5 / m
                yc = (iy + 0.5) / m - 0.5 + 0.5 / m
                zc = (iz + 0.5) / m - 0.5 + 0.5 / m
                points.append([xc, yc, zc])
    arr = np.array(points[:N])
    return arr


def _init_fcc():
    """Face-centered cubic initialization."""
    m = 3
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                base_x = (ix + 0.5) / m - 0.5
                base_y = (iy + 0.5) / m - 0.5
                base_z = (iz + 0.5) / m - 0.5
                h = 0.5 / m
                points.append([base_x, base_y, base_z])
                points.append([base_x + h, base_y + h, base_z])
                points.append([base_x + h, base_y, base_z + h])
                points.append([base_x, base_y + h, base_z + h])
    arr = np.array(points[:N])
    return arr


def _init_random(seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.4, 0.4, size=(N, 3))


def _sa_optimize(centers, time_budget, seed=0):
    """Simulated Annealing optimization."""
    rng = np.random.RandomState(seed)
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    current_r = _radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    T = 0.005
    T_min = 1e-5
    step_size = 0.01
    decay = 0.9995
    
    while time.time() - start < time_budget:
        i = rng.randint(n)
        delta = rng.normal(0, step_size, 3)
        new_center = centers[i].copy() + delta
        new_center = np.clip(new_center, -0.5, 0.5)
        
        new_centers = centers.copy()
        new_centers[i] = new_center
        new_r = _radius(new_centers)
        
        if new_r >= current_r:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            if rng.rand() < np.exp((new_r - current_r) / T):
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
        
        T *= decay
        step_size *= decay
        if T < T_min:
            T = T_min
            step_size = max(step_size, 0.001)
            
    return best_centers, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_c = _init_grid()
    best_r = _radius(best_c)
    
    initializers = [
        ("grid", _init_grid()),
        ("bcc", _init_bcc()),
        ("fcc", _init_fcc()),
        ("rand1", _init_random(42)),
        ("rand2", _init_random(123)),
        ("rand3", _init_random(999)),
        ("rand4", _init_random(7777)),
        ("rand5", _init_random(31415)),
    ]
    
    for name, c0 in initializers:
        if time.time() > deadline:
            break
        
        # Progressive refinement with soft surrogate
        for beta in [20.0, 50.0, 100.0, 200.0, 500.0]:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    _neg_soft_radius,
                    c0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-0.5, 0.5)] * N * 3,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                c0 = res.x
            except Exception:
                break
        
        # Final polish with high beta
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _neg_soft_radius,
                c0,
                args=(500.0,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * N * 3,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            c0 = res.x
        except Exception:
            pass
        
        r = _radius(c0)
        if r > best_r:
            best_r = r
            best_c = c0.copy()
        
        # Simulated annealing polish on the best so far
        if time.time() < deadline - 15:
            sa_budget = min(10.0, deadline - 15 - (time.time() - start_time))
            if sa_budget > 1.0:
                sa_c, sa_r = _sa_optimize(best_c, sa_budget, seed=hash(name) % 10000)
                if sa_r > best_r:
                    best_r = sa_r
                    best_c = sa_c.copy()
    
    # Final validity check
    if not np.all(np.isfinite(best_c)) or np.any(np.abs(best_c) > 0.5):
        best_c = _init_grid()
    
    return best_c

# EVOLVE-BLOCK-END
