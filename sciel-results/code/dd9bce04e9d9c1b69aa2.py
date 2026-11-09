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
    grad_all_c = -weights  # d(-soft_min)/d(all_c) = -weights
    
    # Now propagate gradient back to x
    grad_x = np.zeros_like(x)
    
    # Pairwise part
    n_pw = len(pw_flat)
    grad_pw = grad_all_c[:n_pw]
    # Reshape grad_pw back to (N, N)
    grad_pw_full = np.zeros((N, N))
    grad_pw_full[mask] = grad_pw
    
    # For each pair (i,j), contribution to c_i and c_j
    # pw_ij = |c_i - c_j| / 2
    # d(pw_ij)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    # d(pw_ij)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
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
            # bd_ik = 0.5 - |c_ik|
            # d(bd_ik)/d(c_ik) = -sign(c_ik)
            sign = 1.0 if c[i, k] > 0 else (-1.0 if c[i, k] < 0 else 0.0)
            grad_x[i, k] += grad_bd[i] * (-sign)
    
    return grad_x.flatten()

def anneal_optimize(centers, start_beta=20.0, end_beta=2000.0, n_anneal=50, max_iter_per_stage=50):
    """Anneal beta from start to end, optimizing at each stage."""
    x = centers.flatten().copy()
    betas = np.linspace(start_beta, end_beta, n_anneal)
    
    for beta in betas:
        try:
            res = minimize(soft_min_obj, x, method='L-BFGS-B',
                          jac=lambda xx: soft_min_grad(xx, beta),
                          args=(beta,),
                          options={'maxiter': max_iter_per_stage, 'ftol': 1e-12, 'gtol': 1e-8})
            x = res.x
        except Exception:
            pass
    
    return x.reshape(-1, 3)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_r = -np.inf
    best_centers = None
    
    # Generate initial configurations
    inits = []
    
    # 1. Jittered 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N].copy())
    
    # 2. Various random inits
    for seed in range(15):
        rng = np.random.RandomState(seed)
        inits.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # 3. Jittered grids with different jitter
    for jit in [0.02, 0.05, 0.1]:
        rng = np.random.RandomState(100 + int(jit * 100))
        jittered = grid[:N].copy() + rng.uniform(-jit, jit, size=(N, 3))
        inits.append(np.clip(jittered, -0.49, 0.49))
    
    # Optimize each init using annealing
    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Run annealing optimization
        result = anneal_optimize(init, start_beta=20.0, end_beta=2000.0, 
                                  n_anneal=30, max_iter_per_stage=40)
        result = np.clip(result, -0.5 + 1e-10, 0.5 - 1e-10)
        r = compute_score(result)
        
        if r > best_r:
            best_r = r
            best_centers = result.copy()
        
        # If we have time, do a final high-precision polish
        if time.time() < deadline - 30 and r > 0.13:
            try:
                # Final polish with very high beta
                def final_obj(x):
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
                    if m < -1e6:
                        return 1e6
                    exp_vals = np.exp(-5000.0 * (all_c - m))
                    soft_min = m - np.log(np.sum(exp_vals)) / 5000.0
                    return -soft_min
                
                def final_grad(x):
                    return soft_min_grad(x, 5000.0)
                
                res = minimize(final_obj, result.flatten(), method='L-BFGS-B',
                              jac=final_grad,
                              options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10})
                polished = res.x.reshape(-1, 3)
                polished = np.clip(polished, -0.5 + 1e-10, 0.5 - 1e-10)
                r_polished = compute_score(polished)
                if r_polished > best_r:
                    best_r = r_polished
                    best_centers = polished.copy()
            except Exception:
                pass
    
    return best_centers
