import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 50
DEADLINE = 130.0  # seconds, leaving 20s margin before the 150s kill

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall constraint: min_i min_k (0.5 - |c_ik|)
    wall = np.min(0.5 - np.abs(centers))
    if wall <= 0:
        return 0.0
    # Pairwise constraint: min_{i<j} |c_i - c_j| / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    i_idx, j_idx = np.triu_indices(N, k=1)
    min_pair = np.min(dist[i_idx, j_idx]) / 2.0
    return min(wall, min_pair)

def _soft_min_obj_and_grad(x, beta):
    """
    Objective: negative of the soft-minimum distance (to be minimized).
    Soft-min approximates the true minimum distance (radius).
    x is flat array (N*3,).
    Returns (value, gradient).
    """
    centers = x.reshape(N, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)  # shape (N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dist[i_idx, j_idx] / 2.0  # radius equivalent
    
    # Combine all distances into one array
    all_dists = np.concatenate([wall_dists.ravel(), d_ij])
    
    # Soft min: -beta * log(sum(exp(-d/beta)))
    # We want to MAXIMIZE the min distance, so we MINIMIZE -soft_min
    # Let S = sum(exp(-d/beta)). SoftMin = -beta * log(S).
    # We want to maximize SoftMin, so minimize f = -SoftMin = beta * log(S).
    
    # To avoid overflow/underflow in exp, we can shift by max(all_dists)
    # But since we want min, exp(-d/beta) is largest for smallest d.
    # Let's just compute directly. If beta is small, this is sensitive.
    
    # Numerical stability:
    # log(sum(exp(-d/beta))) = -min(d)/beta + log(sum(exp(-(d-min(d))/beta)))
    # This is more stable.
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_terms = np.exp(-shifted / beta)
    sum_exp = np.sum(exp_terms)
    
    # f = beta * log(S) = beta * ( -min_d/beta + log(sum(exp(-shifted/beta))) )
    # f = -min_d + beta * log(sum_exp)
    value = -min_d + beta * np.log(sum_exp)
    
    # Gradient of value w.r.t x
    # d(value)/d(d_k) = d(-min_d)/d(d_k) + beta * d(log(S))/d(d_k)
    # This is tricky because min_d is not differentiable.
    # However, the soft min gradient is:
    # d(SoftMin)/d(d_k) = exp(-d_k/beta) / S
    # d(value)/d(d_k) = - d(SoftMin)/d(d_k) = - exp(-d_k/beta) / S
    
    weights = exp_terms / sum_exp
    grad_dists = -weights
    
    grad_centers = np.zeros_like(centers)
    
    # Wall gradient
    wall_weights = grad_dists[:N*3].reshape(N, 3)
    # d(wall_dist)/d(c) = -sign(c)
    grad_centers -= wall_weights * np.sign(centers)
    
    # Pair gradient
    pair_weights = grad_dists[N*3:]
    i_masked = i_idx
    j_masked = j_idx
    
    vecs = centers[i_masked] - centers[j_masked]
    d_masked = dist[i_masked, j_masked]
    
    # d(d_k)/d(c_i) = (c_i - c_j) / (2 * d_ij)  (since d_k = d_ij/2)
    coeff = pair_weights / (2.0 * d_masked)
    np.add.at(grad_centers, i_masked, coeff[:, np.newaxis] * vecs)
    np.add.at(grad_centers, j_masked, -coeff[:, np.newaxis] * vecs)
    
    return value, grad_centers.ravel()

def _generate_initial_guess(seed=0):
    """Generate diverse initial guesses."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((N, 3))
    strategy = seed % 5
    
    if strategy == 0:
        # Simple cubic grid + jitter
        # 3x4x4 = 48 points. Add 2 random.
        xs = np.linspace(-0.4, 0.4, 3)
        ys = np.linspace(-0.4, 0.4, 4)
        zs = np.linspace(-0.4, 0.4, 4)
        grid = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)
        centers[:48] = grid
        centers[48:] = rng.uniform(-0.3, 0.3, (2, 3))
        
    elif strategy == 1:
        # BCC-like lattice
        pts = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = (i - 1) * 0.4
                    y = (j - 1) * 0.4
                    z = (k - 1) * 0.4
                    pts.append([x, y, z])
                    if i < 2 and j < 2 and k < 2:
                        pts.append([x + 0.2, y + 0.2, z + 0.2])
        # 27 corners + 8 body centers = 35 points. Add 15 random.
        centers[:35] = np.array(pts)
        centers[35:] = rng.uniform(-0.35, 0.35, (15, 3))
        
    elif strategy == 2:
        # Random uniform
        centers = rng.uniform(-0.4, 0.4, (N, 3))
        
    elif strategy == 3:
        # Two-layer structure
        # Bottom layer: 5x5 grid (25 points)
        # Top layer: 5x5 grid (25 points) offset
        bottom = []
        for i in range(5):
            for j in range(5):
                x = -0.4 + i * 0.2
                y = -0.4 + j * 0.2
                bottom.append([x, y, -0.2])
        top = []
        for i in range(5):
            for j in range(5):
                x = -0.3 + i * 0.2
                y = -0.3 + j * 0.2
                top.append([x, y, 0.2])
        centers[:25] = np.array(bottom)
        centers[25:] = np.array(top)
        centers += rng.normal(0, 0.02, (N, 3))
        
    else:
        # FCC-like arrangement
        # Face-centered cubic lattice points
        pts = []
        # Corner and face centers of a 2x2x2 grid of cells
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = (i - 1) * 0.35
                    y = (j - 1) * 0.35
                    z = (k - 1) * 0.35
                    pts.append([x, y, z])
                    # Face centers
                    if i < 2:
                        pts.append([x + 0.175, y, z])
                    if j < 2:
                        pts.append([x, y + 0.175, z])
                    if k < 2:
                        pts.append([x, y, z + 0.175])
        # Take first 50
        pts = np.array(pts[:50])
        centers = pts + rng.normal(0, 0.03, (N, 3))
    
    centers = np.clip(centers, -0.45, 0.45)
    return centers

def construct():
    """
    Construct optimal sphere packing.
    """
    start_time = time.time()
    
    best_radius = 0.0
    best_centers = None
    
    num_restarts = 20
    
    for restart_idx in range(num_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Generate initial guess
        centers = _generate_initial_guess(seed=restart_idx)
        
        # Optimize using L-BFGS-B to maximize soft-min distance
        # We minimize -soft_min
        x0 = centers.ravel()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            # Start with larger beta for smoothness, decrease it?
            # For simplicity, use a fixed moderate beta.
            # Beta controls the "sharpness" of the min.
            # Small beta -> closer to true min, but noisier gradient.
            # Large beta -> smoother, but less accurate.
            # Let's try beta=0.01 initially.
            res = minimize(
                _soft_min_obj_and_grad,
                x0,
                args=(0.01,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50, 'iprint': -1}
            )
            new_centers = res.x.reshape(N, 3)
            
            # Calculate true radius
            r = _compute_radius(new_centers)
            
            if r > best_radius:
                best_radius = r
                best_centers = new_centers.copy()
                
        except Exception as e:
            pass

    # Final check
    if best_centers is None:
        centers = _generate_initial_guess(seed=99)
        return centers
        
    return best_centers
