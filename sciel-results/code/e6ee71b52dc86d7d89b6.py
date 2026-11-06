import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Initial Configuration: BCC Lattice
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k + 0.5])
    points = np.array(points)
    
    # Scale and center to fit in [-0.5, 0.5]^3
    points = (points - 1.75) * (1.0 / 3.5)
    
    # Select N points closest to origin
    dists = np.sum(points**2, axis=1)
    indices = np.argsort(dists)[:N]
    c = points[indices].copy()
    
    # Ensure they are strictly inside
    c = np.clip(c, -0.49, 0.49)
    
    def compute_radius(c):
        dist = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def soft_min_obj_and_grad(c, beta):
        c = c.reshape(N, 3)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        u_pair = dist / 2.0
        u_wall = 0.5 - np.abs(c)
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        u_pairs_flat = u_pair[mask]
        u_walls_flat = u_wall.flatten()
        
        u_all = np.concatenate([u_pairs_flat, u_walls_flat])
        
        min_u = np.min(u_all)
        exp_terms = np.exp(-beta * (u_all - min_u))
        sum_exp = np.sum(exp_terms)
        
        obj = min_u - (1.0 / beta) * np.log(sum_exp)
        
        weights = exp_terms / sum_exp
        
        n_pairs = len(u_pairs_flat)
        w_pairs = weights[:n_pairs].reshape(N, N)
        
        w_pairs_full = np.zeros((N, N))
        w_pairs_full[mask] = w_pairs
        w_pairs_full = w_pairs_full + w_pairs_full.T
        
        dist_safe = np.where(dist > 1e-9, dist, 1.0)
        grad_dist = diff / dist_safe[:, :, np.newaxis]
        
        w_walls = weights[n_pairs:].reshape(N, 3)
        
        grad_c = np.sum(w_pairs_full[:, :, None] * 0.5 * grad_dist, axis=1) - w_walls * np.sign(c)
        
        return obj, grad_c

    best_c = c.copy()
    best_score = compute_radius(best_c)
    
    beta = 10.0
    step_size = 0.01
    
    while time.time() < deadline - 20.0:
        obj, grad = soft_min_obj_and_grad(best_c, beta)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-9:
            grad = grad / grad_norm
        
        new_c = best_c + step_size * grad
        new_c = np.clip(new_c, -0.499, 0.499)
        
        new_score = compute_radius(new_c)
        
        if new_score > best_score:
            best_c = new_c
            best_score = new_score
        else:
            kick = np.random.normal(0, step_size * 0.5, size=(N, 3))
            kicked_c = best_c + kick
            kicked_c = np.clip(kicked_c, -0.499, 0.499)
            kicked_score = compute_radius(kicked_c)
            
            if kicked_score > best_score:
                best_c = kicked_c
                best_score = kicked_score
        
        beta *= 1.05
        step_size *= 0.99
        
        if step_size < 1e-5:
            step_size = 0.01
            kick = np.random.normal(0, 0.05, size=(N, 3))
            kicked_c = best_c + kick
            kicked_c = np.clip(kicked_c, -0.499, 0.499)
            kicked_score = compute_radius(kicked_c)
            if kicked_score > best_score:
                best_c = kicked_c
                best_score = kicked_score
            beta = 10.0

    # Polish
    if time.time() < deadline - 5.0:
        beta_polish = 1000.0
        step_polish = 1e-4
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            obj, grad = soft_min_obj_and_grad(best_c, beta_polish)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-9:
                grad = grad / grad_norm
            new_c = best_c + step_polish * grad
            new_c = np.clip(new_c, -0.499, 0.499)
            new_score = compute_radius(new_c)
            if new_score > best_score:
                best_c = new_c
                best_score = new_score
            else:
                step_polish *= 0.5
                if step_polish < 1e-7:
                    break

    return best_c
