import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_score(centers):
    """Compute the radius for a given set of centers."""
    if centers.shape != (N, 3):
        return -np.inf
    if not np.all(np.isfinite(centers)):
        return -np.inf
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return -np.inf
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2
    boundary = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary)
    return min(min_pair, min_boundary)

def fibonacci_sphere(n, rng=None):
    """Generate n points on a unit sphere using Fibonacci lattice."""
    if rng is None:
        rng = np.random.RandomState(42)
    # Fibonacci sphere points
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    points = np.array(points)
    # Add small random perturbation
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    return points

def generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Fibonacci sphere scaled to cube
    rng = np.random.RandomState(42)
    sphere_pts = fibonacci_sphere(N, rng)
    # Scale to fit in [-0.45, 0.45]
    cube_pts = sphere_pts * 0.45
    configs.append(np.clip(cube_pts, -0.49, 0.49))
    
    # 2. Jittered 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    configs.append(grid[:N].copy())
    
    # 3. Random with different seeds
    for seed in range(8):
        rng = np.random.RandomState(seed)
        configs.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # 4. Jittered grids
    for jit in [0.02, 0.05, 0.08]:
        rng = np.random.RandomState(200 + int(jit * 100))
        jittered = grid[:N].copy() + rng.uniform(-jit, jit, size=(N, 3))
        configs.append(np.clip(jittered, -0.49, 0.49))
    
    return configs

def repulsion_preoptimize(centers, n_iter=100, lr=0.01):
    """Pre-optimize using repulsion forces to spread points."""
    c = centers.copy()
    for _ in range(n_iter):
        # Compute repulsion forces
        forces = np.zeros_like(c)
        for i in range(N):
            for j in range(i+1, N):
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    continue
                # Repulsion: stronger when closer
                force = d / (dist**3) * 0.1
                forces[i] += force
                forces[j] -= force
        
        # Boundary repulsion
        for i in range(N):
            for k in range(3):
                if c[i, k] > 0.4:
                    forces[i, k] -= (c[i, k] - 0.4) * 10
                elif c[i, k] < -0.4:
                    forces[i, k] -= (c[i, k] + 0.4) * 10
        
        # Update
        c += lr * forces
        c = np.clip(c, -0.49, 0.49)
    
    return c

def soft_min_obj(x, beta):
    """Smooth soft-min objective using log-sum-exp. Returns negative soft-min."""
    c = x.reshape(-1, 3)
    # Pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    pw = d / 2.0
    mask = ~np.eye(N, dtype=bool)
    pw_flat = pw[mask]
    
    # Boundary distances
    bd = 0.5 - np.abs(c)
    bd_flat = np.min(bd, axis=1)
    
    # Combine all constraints
    all_c = np.concatenate([pw_flat, bd_flat])
    
    # Soft-min via log-sum-exp
    m = np.min(all_c)
    if m < -1e6:
        return 1e6
    exp_vals = np.exp(-beta * (all_c - m))
    soft_min = m - np.log(np.sum(exp_vals)) / beta
    return -soft_min

def soft_min_grad(x, beta):
    """Gradient of the soft-min objective."""
    c = x.reshape(-1, 3)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    pw = d / 2.0
    mask = ~np.eye(N, dtype=bool)
    pw_flat = pw[mask]
    
    bd = 0.5 - np.abs(c)
    bd_flat = np.min(bd, axis=1)
    
    all_c = np.concatenate([pw_flat, bd_flat])
    m = np.min(all_c)
    exp_vals = np.exp(-beta * (all_c - m))
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Gradient with respect to all_c
    grad_all_c = -weights
    
    # Now propagate gradient back to x
    grad_x = np.zeros_like(x)
    
    # Pairwise part
    n_pw = len(pw_flat)
    grad_pw = grad_all_c[:n_pw]
    # Reshape grad_pw back to (N, N)
    grad_pw_full = np.zeros((N, N))
    grad_pw_full[mask] = grad_pw
    
    # For each pair (i,j), contribution to c_i and c_j
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if d[i, j] < 1e-10:
                continue
            w = grad_pw_full[i, j]
            direction = (c[i] - c[j]) / (2.0 * d[i, j])
            grad_x[i] += w * direction
            grad_x[j] -= w * direction
    
    # Boundary part
    n_bd = len(bd_flat)
    grad_bd = grad_all_c[n_pw:]
    for i in range(N):
        for k in range(3):
            sign = 1.0 if c[i, k] > 0 else (-1.0 if c[i, k] < 0 else 0.0)
            grad_x[i, k] += grad_bd[i] * (-sign)
    
    return grad_x.flatten()

def anneal_optimize(centers, start_beta=20.0, end_beta=2000.0, n_anneal=30, max_iter_per_stage=40, time_limit=None):
    """Anneal beta from start to end, optimizing at each stage."""
    x = centers.flatten().copy()
    betas = np.linspace(start_beta, end_beta, n_anneal)
    
    for beta in betas:
        if time_limit is not None and time.time() > time_limit:
            break
        try:
            res = minimize(soft_min_obj, x, method='L-BFGS-B',
                          jac=lambda xx: soft_min_grad(xx, beta),
                          args=(beta,),
                          options={'maxiter': max_iter_per_stage, 'ftol': 1e-12, 'gtol': 1e-8})
            x = res.x
        except Exception:
            pass
    
    return x.reshape(-1, 3)

def nelder_mead_polish(centers, max_iter=200, time_limit=None):
    """Polish using Nelder-Mead on the negative min distance."""
    def nm_obj(x):
        c = x.reshape(-1, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2
        boundary = 0.5 - np.abs(c)
        min_boundary = np.min(boundary)
        return -min(min_pair, min_boundary)
    
    try:
        res = minimize(nm_obj, centers.flatten(), method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
        polished = res.x.reshape(-1, 3)
        polished = np.clip(polished, -0.5 + 1e-10, 0.5 - 1e-10)
        return polished
    except Exception:
        return centers

def local_search_perturb(centers, n_perturb=5, perturb_scale=0.05, time_limit=None):
    """Try small perturbations to escape local minima."""
    best = centers.copy()
    best_r = compute_score(best)
    rng = np.random.RandomState(int(time.time()) % 10000)
    
    for _ in range(n_perturb):
        if time_limit is not None and time.time() > time_limit:
            break
        # Perturb a few points
        n_pts = rng.randint(2, 6)
        idx = rng.choice(N, size=n_pts, replace=False)
        perturbed = best.copy()
        perturbed[idx] += rng.uniform(-perturb_scale, perturb_scale, size=(n_pts, 3))
        perturbed = np.clip(perturbed, -0.49, 0.49)
        
        # Quick optimize
        result = anneal_optimize(perturbed, start_beta=100.0, end_beta=1000.0,
                                 n_anneal=10, max_iter_per_stage=20, time_limit=time_limit)
        result = np.clip(result, -0.5 + 1e-10, 0.5 - 1e-10)
        r = compute_score(result)
        
        if r > best_r:
            best_r = r
            best = result.copy()
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_r = -np.inf
    best_centers = None
    
    # Generate initial configurations
    inits = generate_initial_configs()
    
    # Pre-optimize with repulsion
    optimized_inits = []
    for init in inits:
        if time.time() > deadline - 60:
            break
        preopt = repulsion_preoptimize(init, n_iter=50, lr=0.005)
        optimized_inits.append(preopt)
    
    # Optimize each init using annealing
    for idx, init in enumerate(optimized_inits):
        if time.time() > deadline - 30:
            break
        
        # Run annealing optimization
        result = anneal_optimize(init, start_beta=20.0, end_beta=2000.0,
                                  n_anneal=30, max_iter_per_stage=40, time_limit=deadline - 30)
        result = np.clip(result, -0.5 + 1e-10, 0.5 - 1e-10)
        r = compute_score(result)
        
        if r > best_r:
            best_r = r
            best_centers = result.copy()
        
        # If we have time, do a final high-precision polish
        if time.time() < deadline - 20 and r > 0.13:
            # Nelder-Mead polish
            polished = nelder_mead_polish(result, max_iter=100, time_limit=deadline - 15)
            r_polished = compute_score(polished)
            if r_polished > best_r:
                best_r = r_polished
                best_centers = polished.copy()
            
            # Local search perturbation
            if time.time() < deadline - 10:
                perturbed_best = local_search_perturb(polished, n_perturb=3, perturb_scale=0.03, time_limit=deadline - 5)
                r_perturbed = compute_score(perturbed_best)
                if r_perturbed > best_r:
                    best_r = r_perturbed
                    best_centers = perturbed_best.copy()
    
    # Final high-precision polish if we have time
    if best_centers is not None and time.time() < deadline - 5:
        final_polished = nelder_mead_polish(best_centers, max_iter=200, time_limit=deadline - 2)
        final_r = compute_score(final_polished)
        if final_r > best_r:
            best_r = final_r
            best_centers = final_polished.copy()
    
    if best_centers is None:
        # Fallback
        best_centers = inits[0]
    
    return best_centers
