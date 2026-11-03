"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

N = 50

def _score(centers):
    """Compute the packing radius for a set of centers in [-0.5, 0.5]^3."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = wall_dists.min()
    
    # Pairwise distances
    dists = pdist(centers)
    min_pair = dists.min() / 2.0
    
    return min(min_wall, min_pair)

def _soft_min_obj_and_grad(centers, beta=100.0):
    """
    Compute objective (negative soft-min of constraints) and its gradient.
    Constraints:
    1. Pairwise distances / 2
    2. Wall distances (0.5 - |x|, 0.5 - |y|, 0.5 - |z|)
    """
    c = centers.reshape(-1, 3)
    
    # Compute pairwise distances
    d = pdist(c)
    n_pairs = len(d)
    
    # Compute wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall_per_point = wall_dists.min(axis=1)
    
    # The set of values for soft-min:
    vals = np.concatenate([d / 2.0, min_wall_per_point])
    
    # Soft min calculation
    v_min = vals.min()
    exp_terms = np.exp(-beta * (vals - v_min))
    sum_exp = np.sum(exp_terms)
    log_sum_exp = np.log(sum_exp)
    
    obj = (1.0 / beta) * log_sum_exp - v_min
    
    # Gradient
    weights = exp_terms / sum_exp
    
    grad_c = np.zeros_like(c)
    
    # Part 1: Pairwise distances
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            w = weights[idx]
            if w > 1e-12:
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-12:
                    factor = w / (2.0 * dist)
                    grad_c[i] += factor * diff
                    grad_c[j] -= factor * diff
            idx += 1
            
    # Part 2: Wall distances
    for i in range(N):
        w = weights[N + i]
        if w > 1e-12:
            x, y, z = c[i]
            dx = 0.5 - abs(x)
            dy = 0.5 - abs(y)
            dz = 0.5 - abs(z)
            
            min_d = min(dx, dy, dz)
            
            eps = 1e-6
            g = np.zeros(3)
            count = 0
            if abs(dx - min_d) < eps:
                g[0] += -np.sign(x)
                count += 1
            if abs(dy - min_d) < eps:
                g[1] += -np.sign(y)
                count += 1
            if abs(dz - min_d) < eps:
                g[2] += -np.sign(z)
                count += 1
            
            if count > 0:
                g /= count
                grad_c[i] += w * g
    
    return obj, -grad_c

def _optimize_pack(centers_init, beta=100.0, max_iter=1000, tol=1e-12):
    """Run L-BFGS-B optimization."""
    x0 = centers_init.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def func_and_grad(x):
        c = x.reshape(-1, 3)
        f, g = _soft_min_obj_and_grad(c, beta)
        return f, g
        
    result = minimize(
        func_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'disp': False}
    )
    
    return result.x.reshape(-1, 3)

def _generate_initial_grid():
    """Generate a reasonable initial grid placement."""
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Initial guess
    best_centers = _generate_initial_grid()
    best_score = _score(best_centers)
    
    # Multi-start optimization
    # 1. Grid start
    if time.time() < deadline - 10:
        res = _optimize_pack(best_centers, beta=50.0, max_iter=200)
        s = _score(res)
        if s > best_score:
            best_score = s
            best_centers = res
    
    # 2. Random starts
    for i in range(5):
        if time.time() > deadline - 20:
            break
            
        centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        
        res = _optimize_pack(centers, beta=20.0, max_iter=100)
        s1 = _score(res)
        
        if time.time() < deadline - 15:
            res = _optimize_pack(res, beta=100.0, max_iter=200)
            s2 = _score(res)
            if s2 > s1:
                s1 = s2
        
        if s1 > best_score:
            best_score = s1
            best_centers = res.copy()
            
    # 3. Perturbation of best
    if time.time() < deadline - 10:
        perturbed = best_centers + rng.normal(0, 0.05, size=best_centers.shape)
        perturbed = np.clip(perturbed, -0.49, 0.49)
        res = _optimize_pack(perturbed, beta=50.0, max_iter=200)
        s = _score(res)
        if s > best_score:
            best_score = s
            best_centers = res

    # Final polish with very high beta and tight tolerance
    if time.time() < deadline - 5:
        res = _optimize_pack(best_centers, beta=200.0, max_iter=500, tol=1e-14)
        s = _score(res)
        if s > best_score:
            best_score = s
            best_centers = res

    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
