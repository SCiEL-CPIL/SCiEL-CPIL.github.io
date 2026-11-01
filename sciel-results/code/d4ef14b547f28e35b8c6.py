"""Improved packing: smooth surrogate objective with soft-min for nonsmooth constraints."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(centers):
    """Compute the maximum common radius for given centers (exact, for evaluation)."""
    centers = np.asarray(centers)
    wall_min = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_min)
    n = len(centers)
    if n < 2:
        return r_wall
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _soft_min(values, beta=50.0):
    """Smooth approximation to min via log-sum-exp (negative log-sum-exp)."""
    values = np.asarray(values)
    m = np.min(values)
    exponent = -beta * (values - m)
    exponent = np.clip(exponent, -700, 700)
    log_sum = np.log(np.sum(np.exp(exponent)))
    return m - log_sum / beta

def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate: minimize -soft_min(all constraints).
    Constraints: wall distances (0.5 - |c_ik|) and half pairwise distances.
    Each pair counted exactly once.
    """
    c = x.reshape(-1, 3)
    
    # Wall constraints: 0.5 - |c_ik| for each coordinate
    wall_constraints = 0.5 - np.abs(c).ravel()
    
    # Pairwise half-distances (each pair once)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    iu = np.triu_indices_from(pair_constraints, k=1)
    pair_constraints = pair_constraints[iu]
    
    # Combine all constraints
    all_constraints = np.concatenate([wall_constraints, pair_constraints])
    
    # Soft-min of all constraints
    s = _soft_min(all_constraints, beta)
    
    # Penalty for going outside bounds (soft)
    out_of_bounds = np.maximum(0, np.abs(c) - 0.4999)
    penalty = 10.0 * np.sum(out_of_bounds ** 2)
    
    return -s + penalty

def _smooth_gradient(x, beta=50.0):
    """Numerical gradient of the smooth objective."""
    eps = 1e-7
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (_smooth_objective(xp, beta) - _smooth_objective(xm, beta)) / (2 * eps)
    return grad

def _initial_grid():
    """FCC-inspired grid initialization."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(grid))
    return grid[perm[:N]].copy()

def _initial_fcc_like():
    """Initialize with FCC-like structure: layers of hexagonal packing."""
    centers = np.zeros((N, 3))
    n_layers = 5
    per_layer = N // n_layers
    remaining = N - per_layer * n_layers
    
    for layer in range(n_layers):
        z = (layer + 0.5) / n_layers - 0.5
        n_in_layer = per_layer + (1 if layer < remaining else 0)
        
        for i in range(n_in_layer):
            row = i // 4
            col = i % 4
            x = (col + 0.5) / 4.0 - 0.5
            y = (row + 0.5) / 3.0 - 0.5
            if row % 2 == 1:
                x += 0.25 / 4.0
            centers[layer * per_layer + i] = [x, y, z]
    
    idx = per_layer * n_layers
    while idx < N:
        centers[idx] = [0, 0, 0]
        idx += 1
    
    return centers

def _initial_bcc():
    """BCC-inspired initialization: body-centered cubic layers."""
    centers = []
    # 3x3x3 grid = 27, plus body centers
    m = 3
    ticks = (np.arange(m) + 0.5) / m - 0.5
    for z in ticks:
        for y in ticks:
            for x in ticks:
                centers.append([x, y, z])
            # Add body centers between
    # Also add face centers
    for z in ticks:
        for y in ticks:
            centers.append([0, y, z])
            centers.append([y, 0, z])
    # Add more to reach 50
    rng = np.random.RandomState(123)
    while len(centers) < N:
        centers.append(rng.uniform(-0.45, 0.45, 3))
    return np.array(centers[:N])

def _local_search_polish(centers, max_iters=500, time_limit=20.0):
    """Coordinate descent local search with adaptive step size."""
    cur = centers.copy()
    cur_score = _score(cur)
    step = 0.01
    t_start = time.time()
    
    for it in range(max_iters):
        if time.time() - t_start > time_limit:
            break
        improved = False
        for i in range(N):
            for coord in range(3):
                for delta in [step, -step, step*0.5, -step*0.5, step*2, -step*2, step*0.1, -step*0.1]:
                    new_c = cur.copy()
                    new_c[i, coord] += delta
                    if np.all(np.abs(new_c) <= 0.5):
                        s = _score(new_c)
                        if s > cur_score + 1e-12:
                            cur = new_c
                            cur_score = s
                            improved = True
        if not improved:
            step *= 0.5
            if step < 1e-8:
                break
    
    return cur, cur_score

def _repulsion_anneal(centers, time_limit=15.0):
    """Repulsion-based annealing: push spheres apart using soft repulsion."""
    cur = centers.copy()
    t_start = time.time()
    
    # Anneal from high to low temperature
    for temp in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        if time.time() - t_start > time_limit:
            break
        
        for it in range(200):
            if time.time() - t_start > time_limit:
                break
            
            forces = np.zeros_like(cur)
            # Pairwise repulsion
            for i in range(N):
                for j in range(i+1, N):
                    diff = cur[i] - cur[j]
                    d = np.linalg.norm(diff)
                    if d < 1e-10:
                        d = 1e-10
                    # Soft repulsion
                    f = 1.0 / (d * d + 0.01)
                    forces[i] += f * diff / d
                    forces[j] -= f * diff / d
            
            # Wall repulsion
            for i in range(N):
                for coord in range(3):
                    if cur[i, coord] > 0.4:
                        forces[i, coord] -= 5.0 * (cur[i, coord] - 0.4)
                    elif cur[i, coord] < -0.4:
                        forces[i, coord] += 5.0 * (-0.4 - cur[i, coord])
            
            # Update with annealing
            cur += temp * forces
            # Project back into cube
            cur = np.clip(cur, -0.499, 0.499)
    
    return cur

def construct():
    t_start = time.time()
    deadline = t_start + 130  # Safety margin before 150s limit
    
    # Try multiple initializations
    inits = [
        ("grid", _initial_grid()),
        ("fcc", _initial_fcc_like()),
        ("bcc", _initial_bcc()),
    ]
    
    best_centers = None
    best_score = -np.inf
    
    for name, init in inits:
        if time.time() > deadline - 30:
            break
        
        cur = init.copy()
        cur_score = _score(cur)
        
        if best_centers is None or cur_score > best_score:
            best_centers = cur.copy()
            best_score = cur_score
        
        # Repulsion anneal
        if time.time() < deadline - 40:
            try:
                rep_cur = _repulsion_anneal(cur, time_limit=min(10.0, deadline - time.time() - 40))
                rep_score = _score(rep_cur)
                if rep_score > best_score:
                    best_centers = rep_cur.copy()
                    best_score = rep_score
                    cur = rep_cur.copy()
            except Exception:
                pass
        
        # Optimize with smooth objective
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline - 20:
                break
            
            try:
                res = minimize(
                    _smooth_objective,
                    cur.ravel(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 3 * N,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                new_centers = res.x.reshape(-1, 3)
                new_score = _score(new_centers)
                
                if new_score > best_score:
                    best_centers = new_centers.copy()
                    best_score = new_score
                    cur = new_centers.copy()
            except Exception:
                pass
        
        # Final polish with coordinate descent
        if time.time() < deadline - 10:
            try:
                pol_cur, pol_score = _local_search_polish(
                    best_centers, 
                    max_iters=300,
                    time_limit=min(15.0, deadline - time.time() - 5)
                )
                if pol_score > best_score:
                    best_centers = pol_cur.copy()
                    best_score = pol_score
            except Exception:
                pass
    
    # Final global polish if time remains
    if time.time() < deadline - 3:
        try:
            pol_cur, pol_score = _local_search_polish(
                best_centers,
                max_iters=200,
                time_limit=min(10.0, deadline - time.time() - 2)
            )
            if pol_score > best_score:
                best_centers = pol_cur.copy()
                best_score = pol_score
        except Exception:
            pass
    
    return best_centers
