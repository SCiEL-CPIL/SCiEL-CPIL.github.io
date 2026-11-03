import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_score(centres):
    """
    Compute the score for a given set of centres.
    Score = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|))
    """
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    if np.any(centres < -0.5 - 1e-9) or np.any(centres > 0.5 + 1e-9):
        return 0.0
        
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    # Boundary distance
    # min_k (0.5 - |c_ik|)
    # This is 0.5 - max_k |c_ik|
    max_abs_coord = np.max(np.abs(centres), axis=1)
    min_boundary = 0.5 - np.min(max_abs_coord)
    
    r = min(min_dist / 2.0, min_boundary)
    return r

def objective_and_grad(x):
    """
    Objective: -score(x)
    x is flattened array of shape (N, 3)
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    min_dist = np.min(dist)
    
    # Boundary distance
    max_abs_coord = np.max(np.abs(c), axis=1)
    min_boundary = 0.5 - np.min(max_abs_coord)
    
    r = min(min_dist / 2.0, min_boundary)
    
    # Gradient calculation
    grad = np.zeros_like(x)
    
    if min_dist / 2.0 < min_boundary:
        # The bottleneck is the closest pair
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        vec = c[i] - c[j]
        norm_vec = np.linalg.norm(vec)
        if norm_vec > 1e-10:
            grad[2*i:2*i+3] = -0.5 * vec / norm_vec
            grad[2*j:2*j+3] = 0.5 * vec / norm_vec
    else:
        # The bottleneck is the boundary
        m_i = np.max(np.abs(c), axis=1)
        i_star = np.argmin(m_i)
        c_i = c[i_star]
        abs_c_i = np.abs(c_i)
        k_star = np.argmax(abs_c_i)
        
        sign_val = np.sign(c_i[k_star])
        if sign_val == 0:
            sign_val = 1.0
        grad[2*i_star + k_star] = sign_val
        
    return -r, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # 1. Initialization: Greedy selection from a grid
    grid_coords = np.arange(-0.5, 0.5 + 0.05, 0.1)
    grid = np.array([(x, y, z) for z in grid_coords for y in grid_coords for x in grid_coords])
    
    selected = []
    remaining = grid.copy()
    
    for _ in range(N):
        if len(remaining) == 0:
            break
        if len(selected) == 0:
            idx = len(remaining) // 2
            p = remaining[idx]
            selected.append(p)
            remaining = np.delete(remaining, idx, axis=0)
        else:
            diff = remaining[:, np.newaxis, :] - selected[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            min_dists = np.min(dists, axis=1)
            idx = np.argmax(min_dists)
            p = remaining[idx]
            selected.append(p)
            remaining = np.delete(remaining, idx, axis=0)
            
    x0 = np.array(selected).flatten()
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_x = x0
    best_score = compute_score(x0.reshape(N, 3))
    
    try:
        res = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        if time.time() < deadline:
            x_opt = res.x
            score_opt = compute_score(x_opt.reshape(N, 3))
            if score_opt > best_score:
                best_x = x_opt
                best_score = score_opt
    except Exception:
        pass
    
    # 2. Random restarts
    if time.time() < deadline - 5:
        for i in range(10):
            if time.time() > deadline - 5:
                break
            x_rand = np.random.uniform(-0.5, 0.5, size=(N * 3))
            try:
                res_rand = minimize(
                    objective_and_grad,
                    x_rand,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                x_opt_rand = res_rand.x
                score_opt_rand = compute_score(x_opt_rand.reshape(N, 3))
                if score_opt_rand > best_score:
                    best_x = x_opt_rand
                    best_score = score_opt_rand
            except Exception:
                pass

    return best_x.reshape(N, 3)
