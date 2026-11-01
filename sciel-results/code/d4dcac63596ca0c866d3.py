import numpy as np
from math import fsum, log
from scipy.optimize import minimize

N = 150

def compute_energy_fsum(pts):
    """Energy with fsum for precision."""
    n = len(pts)
    iu = np.triu_indices(n, k=1)
    diff = pts[iu[0]] - pts[iu[1]]
    d = np.sqrt(np.sum(diff**2, axis=1))
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0/d)
    return fsum(terms.tolist())

def construct():
    np.random.seed(42)
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initialize Distance Matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(D, 1.0)
    
    # Compute initial energy
    iu = np.triu_indices(N, k=1)
    current_energy = fsum(np.log(1.0 / D[iu]).tolist())
    best_energy = current_energy
    best_pts = pts.copy()
    
    # Multi-stage Simulated Annealing with multiple move types
    
    # Stage 1: Large moves to explore configuration space
    T0 = 0.5
    Tf = 0.01
    n_iter1 = 300000
    alpha1 = (Tf/T0)**(1.0/n_iter1)
    T = T0
    
    for it in range(n_iter1):
        T = max(T * alpha1, Tf)
        
        move_type = np.random.randint(3)
        
        if move_type == 0:
            # Single point rotation
            k = np.random.randint(N)
            theta = np.random.uniform(0.1, 0.8)
            axis = np.random.randn(3)
            norm_axis = np.linalg.norm(axis)
            if norm_axis < 1e-10:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis /= norm_axis
            
            c = np.cos(theta)
            s = np.sin(theta)
            old_pt = pts[k]
            cross = np.cross(axis, old_pt)
            dot = np.dot(axis, old_pt)
            new_pt = old_pt * c + cross * s + axis * dot * (1.0 - c)
            
            norm_new = np.linalg.norm(new_pt)
            if norm_new < 1e-10:
                continue
            new_pt /= norm_new
            
            others_idx = np.arange(N)
            others_idx = others_idx[others_idx != k]
            
            d_old = D[k, others_idx]
            d_new = np.linalg.norm(new_pt - pts[others_idx], axis=1)
            
            d_old = np.maximum(d_old, 1e-15)
            d_new = np.maximum(d_new, 1e-15)
            
            delta_E = np.sum(np.log(d_old)) - np.sum(np.log(d_new))
            
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E/T):
                pts[k] = new_pt
                current_energy += delta_E
                D[k, others_idx] = d_new
                D[others_idx, k] = d_new
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
        
        elif move_type == 1:
            # Pair swap (exchange positions of two points)
            k1, k2 = np.random.choice(N, 2, replace=False)
            
            old1, old2 = pts[k1].copy(), pts[k2].copy()
            
            # Compute delta energy for swapping
            others_idx = np.arange(N)
            others_idx = others_idx[(others_idx != k1) & (others_idx != k2)]
            
            # Old contributions involving k1 and k2
            d_old_1_others = D[k1, others_idx]
            d_old_2_others = D[k2, others_idx]
            d_old_12 = D[k1, k2]
            
            # New distances after swap
            d_new_1_others = np.linalg.norm(old2 - pts[others_idx], axis=1)
            d_new_2_others = np.linalg.norm(old1 - pts[others_idx], axis=1)
            d_new_12 = d_old_12  # distance between them is same
            
            d_old_1_others = np.maximum(d_old_1_others, 1e-15)
            d_old_2_others = np.maximum(d_old_2_others, 1e-15)
            d_new_1_others = np.maximum(d_new_1_others, 1e-15)
            d_new_2_others = np.maximum(d_new_2_others, 1e-15)
            
            delta_E = (np.sum(np.log(d_old_1_others)) + np.sum(np.log(d_old_2_others)) + log(d_old_12)) - \
                      (np.sum(np.log(d_new_1_others)) + np.sum(np.log(d_new_2_others)) + log(d_new_12))
            
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E/T):
                pts[k1], pts[k2] = old2, old1
                current_energy += delta_E
                
                D[k1, others_idx] = d_new_1_others
                D[others_idx, k1] = d_new_1_others
                D[k2, others_idx] = d_new_2_others
                D[others_idx, k2] = d_new_2_others
                D[k1, k2] = d_new_12
                D[k2, k1] = d_new_12
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
        
        else:
            # Local neighborhood refinement: move point toward nearest neighbor
            k = np.random.randint(N)
            # Find nearest neighbor
            D_k = D[k].copy()
            D_k[k] = np.inf
            j = np.argmin(D_k)
            
            # Move slightly toward j
            direction = pts[j] - pts[k]
            norm_dir = np.linalg.norm(direction)
            if norm_dir < 1e-10:
                continue
            direction /= norm_dir
            
            step = np.random.uniform(0.01, 0.1)
            new_pt = pts[k] + step * direction
            norm_new = np.linalg.norm(new_pt)
            if norm_new < 1e-10:
                continue
            new_pt /= norm_new
            
            others_idx = np.arange(N)
            others_idx = others_idx[others_idx != k]
            
            d_old = D[k, others_idx]
            d_new = np.linalg.norm(new_pt - pts[others_idx], axis=1)
            
            d_old = np.maximum(d_old, 1e-15)
            d_new = np.maximum(d_new, 1e-15)
            
            delta_E = np.sum(np.log(d_old)) - np.sum(np.log(d_new))
            
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E/T):
                pts[k] = new_pt
                current_energy += delta_E
                D[k, others_idx] = d_new
                D[others_idx, k] = d_new
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
        
        # Periodic recompute to avoid drift
        if it % 50000 == 0:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            D = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(D, 1.0)
            current_energy = fsum(np.log(1.0 / D[iu]).tolist())
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
    
    # Stage 2: Fine-tuning with smaller moves
    T0 = 0.01
    Tf = 0.0001
    n_iter2 = 400000
    alpha2 = (Tf/T0)**(1.0/n_iter2)
    T = T0
    
    for it in range(n_iter2):
        T = max(T * alpha2, Tf)
        
        k = np.random.randint(N)
        theta = np.random.uniform(0.005, 0.05)
        axis = np.random.randn(3)
        norm_axis = np.linalg.norm(axis)
        if norm_axis < 1e-10:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis /= norm_axis
        
        c = np.cos(theta)
        s = np.sin(theta)
        old_pt = pts[k]
        cross = np.cross(axis, old_pt)
        dot = np.dot(axis, old_pt)
        new_pt = old_pt * c + cross * s + axis * dot * (1.0 - c)
        
        norm_new = np.linalg.norm(new_pt)
        if norm_new < 1e-10:
            continue
        new_pt /= norm_new
        
        others_idx = np.arange(N)
        others_idx = others_idx[others_idx != k]
        
        d_old = D[k, others_idx]
        d_new = np.linalg.norm(new_pt - pts[others_idx], axis=1)
        
        d_old = np.maximum(d_old, 1e-15)
        d_new = np.maximum(d_new, 1e-15)
        
        delta_E = np.sum(np.log(d_old)) - np.sum(np.log(d_new))
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E/T):
            pts[k] = new_pt
            current_energy += delta_E
            D[k, others_idx] = d_new
            D[others_idx, k] = d_new
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        if it % 100000 == 0:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            D = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(D, 1.0)
            current_energy = fsum(np.log(1.0 / D[iu]).tolist())
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
    
    # Final polish with scipy.optimize.minimize
    x0 = best_pts.flatten()
    
    def penalized_energy(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        penalty = 100.0 * np.sum((norms - 1.0)**2)
        norms_safe = np.maximum(norms, 1e-15)
        pts_proj = pts / norms_safe[:, np.newaxis]
        return compute_energy_fsum(pts_proj) + penalty
    
    def penalized_grad(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        norms_safe = np.maximum(norms, 1e-15)
        pts_proj = pts / norms_safe[:, np.newaxis]
        
        grads_proj = np.zeros_like(pts_proj)
        for k in range(N):
            others = np.delete(pts_proj, k, axis=0)
            diff = pts_proj[k][np.newaxis, :] - others
            dists_sq = np.sum(diff**2, axis=1)
            dists_sq = np.maximum(dists_sq, 1e-30)
            grads_proj[k] = -np.sum(diff / dists_sq[:, np.newaxis], axis=0)
            
        grads = np.zeros_like(pts)
        for i in range(N):
            u = pts_proj[i]
            proj = np.eye(3) - np.outer(u, u)
            grads[i] = (proj @ grads_proj[i]) / norms_safe[i]
            
        penalty_grad = 200.0 * (norms_safe - 1.0)[:, np.newaxis] * pts / norms_safe[:, np.newaxis]
        
        return (grads + penalty_grad).flatten()

    try:
        res = minimize(penalized_energy, x0, jac=penalized_grad, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-12})
        if res.success or res.fun < best_energy:
            final_pts = res.x.reshape(N, 3)
            norms = np.linalg.norm(final_pts, axis=1)
            final_pts = final_pts / norms[:, np.newaxis]
            final_energy = compute_energy_fsum(final_pts)
            if final_energy < best_energy:
                best_pts = final_pts
    except Exception as e:
        pass
    
    return best_pts
