import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # 1. Initial Radial Ring Configuration
    def radial_ring_init(n_points):
        """Create initial configuration with points on concentric rings."""
        points = []
        points.append([0.0, 0.0])
        n_remaining = n_points - 1
        
        ring_indices = 1
        while n_remaining > 0:
            n_on_ring = min(n_remaining, 6 * ring_indices)
            n_remaining -= n_on_ring
            
            ring_radius = 0.95 * np.sqrt(ring_indices / (np.ceil(n_points / 6.0)))
            ring_radius = min(ring_radius, 0.98)
            
            for k in range(n_on_ring):
                angle = 2.0 * np.pi * k / n_on_ring
                x = ring_radius * np.cos(angle)
                y = ring_radius * np.sin(angle)
                points.append([x, y])
            
            ring_indices += 1
        
        return np.array(points[:n_points])
    
    def hex_lattice_init(n_points):
        """Create initial hexagonal lattice configuration."""
        spacing = 2.0 / (np.sqrt(n_points) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            pts = np.array(rows)
            radius = np.hypot(pts[:, 0], pts[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            pts, radius = pts[mask], radius[mask]
            if len(pts) >= n_points:
                return pts[np.argsort(radius, kind="stable")[:n_points]].copy()
            spacing *= 0.98
    
    def force_directed_init(n_points, iterations=200):
        """Create initial configuration using force-directed relaxation."""
        # Start with hex lattice
        x = hex_lattice_init(n_points).copy()
        
        # Force-directed relaxation: push apart close points, pull toward center
        for it in range(iterations):
            if time.time() > deadline - 60:
                break
            
            # Compute pairwise distances
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            
            # Avoid division by zero
            np.fill_diagonal(dist, 1.0)
            
            # Repulsion force (inverse square)
            force = np.zeros_like(x)
            for i in range(n_points):
                for j in range(n_points):
                    if i != j and dist[i, j] < 0.2:
                        f = 0.01 / (dist[i, j] ** 2 + 1e-6)
                        force[i] += f * (x[i] - x[j]) / dist[i, j]
            
            # Attraction to center (weak)
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            force -= 0.001 * x / (norms + 1e-6)
            
            # Update positions
            step = 0.1 / (1.0 + it * 0.01)
            x += step * force
            
            # Keep inside circle
            norms = np.linalg.norm(x, axis=1)
            mask = norms > 0.98
            if np.any(mask):
                x[mask] *= 0.98 / norms[mask]
        
        return x
    
    # Choose better initial configuration
    x0_ring = radial_ring_init(N)
    x0_hex = hex_lattice_init(N)
    
    def get_radius_candidates(x):
        x = x.reshape(-1, 2)
        dist_boundary = 1.0 - np.linalg.norm(x, axis=1)
        d_pair = pdist(x) / 2.0
        return np.concatenate([dist_boundary, d_pair])
    
    score_ring = np.min(get_radius_candidates(x0_ring))
    score_hex = np.min(get_radius_candidates(x0_hex))
    
    # Try force-directed if time allows
    if time.time() < deadline - 30:
        x0_fd = force_directed_init(N, iterations=100)
        score_fd = np.min(get_radius_candidates(x0_fd))
        if score_fd > max(score_ring, score_hex):
            x0_hex = x0_fd
            score_hex = score_fd
    
    best_x = x0_ring.copy() if score_ring >= score_hex else x0_hex.copy()
    best_raw_score = max(score_ring, score_hex)
    
    # 2. Vectorized soft-min objective with analytical gradient
    def soft_min_obj_and_grad(x, beta):
        """
        Returns (objective, gradient) for soft-min objective.
        Vectorized pairwise gradient computation.
        """
        x = x.reshape(-1, 2)
        
        # Compute all radius candidates
        norms = np.linalg.norm(x, axis=1)
        d_boundary = 1.0 - norms
        
        # Pairwise distances (condensed form)
        d_pair = pdist(x) / 2.0
        
        d = np.concatenate([d_boundary, d_pair])
        d = np.maximum(d, 1e-12)
        
        # Compute log-sum-exp for stability
        A = -beta * d
        max_A = np.max(A)
        exp_A = np.exp(A - max_A)
        sum_exp = np.sum(exp_A)
        
        if sum_exp == 0 or not np.isfinite(sum_exp):
            return 1e10, np.zeros(2 * N)
        
        log_sum_exp = max_A + np.log(sum_exp)
        obj = log_sum_exp / beta
        
        # Weights
        w = exp_A / sum_exp
        
        # Gradient computation - vectorized
        grad = np.zeros((N, 2))
        
        # Boundary terms
        valid = norms > 1e-10
        if np.any(valid):
            grad[valid] += (w[:N][valid] * x[valid] / norms[valid])[:, np.newaxis]
        
        # Pairwise terms - vectorized using full distance matrix
        # Build full pairwise distance matrix
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 2)
        dist_full = np.linalg.norm(diff, axis=2)  # (N, N)
        np.fill_diagonal(dist_full, 1.0)
        
        # Reconstruct pairwise weights from condensed form
        # pdist condensed index for (i,j) with i<j: idx = i*(2*N-1-i)//2 + (j-i-1)
        pair_w = w[N:]  # weights for pairwise terms
        
        # Build full weight matrix (symmetric)
        w_full = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                idx = i * (2 * N - 1 - i) // 2 + (j - i - 1)
                w_full[i, j] = pair_w[idx]
                w_full[j, i] = pair_w[idx]
        
        # Gradient contribution from pairwise terms
        # grad[i] -= sum_j w_ij * (x_i - x_j) / (2 * dist_ij)
        # grad[j] += sum_i w_ij * (x_i - x_j) / (2 * dist_ij)
        
        # Vectorized: for each pair (i,j), contribution to grad[i] is -w_ij * diff[i,j] / (2*dist_ij)
        # Use broadcasting
        dist_safe = np.where(dist_full > 1e-10, dist_full, 1.0)
        inv_dist = w_full / (2.0 * dist_safe)
        
        # grad[i] = -sum_j inv_dist[i,j] * diff[i,j]
        grad_pair = -np.sum(inv_dist[:, :, np.newaxis] * diff, axis=1)
        
        grad += grad_pair
        
        grad_flat = grad.flatten()
        
        if not np.all(np.isfinite(grad_flat)):
            grad_flat = np.zeros(2 * N)
        
        return obj, grad_flat
    
    # 3. Annealing loop with increasing beta
    betas = [5, 10, 20, 50, 100, 200, 400, 800]
    current_x = best_x.copy()
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
        
        try:
            res = minimize(soft_min_obj_and_grad, current_x, args=(beta,),
                           method='L-BFGS-B', bounds=bounds, jac=True,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000})
            
            if res.success or res.fun < soft_min_obj_and_grad(current_x, beta)[0]:
                current_x = res.x
        except Exception:
            pass
        
        raw_s = np.min(get_radius_candidates(current_x))
        if raw_s > best_raw_score:
            best_raw_score = raw_s
            best_x = current_x.copy()
        
        if best_raw_score > 0.05315:
            break
    
    # 4. Final high-precision polish
    if time.time() < deadline - 10:
        def neg_raw_score(x):
            x = x.reshape(-1, 2)
            d = get_radius_candidates(x)
            return -np.min(d)
        
        try:
            res = minimize(neg_raw_score, best_x, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14,
                                    'adaptive': True})
            if res.fun < -best_raw_score:
                best_x = res.x
                best_raw_score = -res.fun
        except Exception:
            pass
    
    # 5. Another polish pass
    if time.time() < deadline - 5:
        try:
            res = minimize(neg_raw_score, best_x, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-14, 'fatol': 1e-16,
                                    'adaptive': True})
            if res.fun < -best_raw_score:
                best_x = res.x
                best_raw_score = -res.fun
        except Exception:
            pass
    
    # Final validation
    best_x = best_x.reshape(-1, 2)
    norms = np.linalg.norm(best_x, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_x[mask] *= (1.0 / norms[mask] - 0.001)[:, np.newaxis]
    
    return best_x
