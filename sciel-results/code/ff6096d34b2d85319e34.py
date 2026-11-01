import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 200

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min_value_and_grad(c, beta):
    """
    Computes the negative soft-min radius and its gradient.
    Objective: minimize -soft_min_radius
    """
    n = c.shape[0]
    
    # 1. Compute constraint values v_k
    # Boundary terms
    b_left = 0.5 - c[:, 0]
    b_right = 0.5 + c[:, 0]
    b_bottom = 0.5 - c[:, 1]
    b_top = 0.5 + c[:, 1]
    
    # Pairwise terms
    # We need distances and indices
    # pdist returns condensed vector
    d = pdist(c, metric='euclidean')
    p_terms = 0.5 * d
    
    # Combine all values
    # Total count: 4*N + N*(N-1)/2
    all_vals = np.concatenate([b_left, b_right, b_bottom, b_top, p_terms])
    
    # 2. Soft min calculation
    m = np.min(all_vals)
    # Shift to avoid overflow/underflow
    shifted = all_vals - m
    exp_terms = np.exp(-beta * shifted)
    
    S = np.sum(exp_terms)
    soft_min_val = m - (1.0 / beta) * np.log(S)
    
    # Objective is -soft_min_val
    obj_val = -soft_min_val
    
    # 3. Gradient calculation
    # d(obj)/dc = - d(soft_min)/dc
    # d(soft_min)/dc = (1/S) * sum_k w_k * (d v_k / dc)
    # where w_k = exp_terms[k]
    
    grad = np.zeros_like(c)
    
    # Weights
    w = exp_terms / S
    
    # Boundary gradients
    # Indices 0 to N-1: left (0.5 - x). d/dx = -1
    grad[:, 0] -= np.sum(w[:N] * 1.0)  # - (1/S) * sum(w * (-1)) = + sum(w)? 
    # Wait. 
    # v_left = 0.5 - c_i0. dv/dc_i0 = -1.
    # Contribution to d(soft_min)/dc_i0 from left: w_i * (-1).
    # So d(soft_min)/dc = sum w_k dv_k/dc.
    # Let's just accumulate d(soft_min)/dc first, then negate for objective.
    
    dm_dc = np.zeros_like(c)
    
    # Left boundary: dv/dc_x = -1
    dm_dc[:, 0] += w[:N] * (-1.0)
    # Right boundary: dv/dc_x = 1
    dm_dc[:, 0] += w[N:2*N] * (1.0)
    # Bottom boundary: dv/dc_y = -1
    dm_dc[:, 1] += w[2*N:3*N] * (-1.0)
    # Top boundary: dv/dc_y = 1
    dm_dc[:, 1] += w[3*N:4*N] * (1.0)
    
    # Pairwise gradients
    # Need to map condensed index to (i, j)
    # This is the expensive part.
    # For N=200, ~20k pairs.
    # We can vectorize this?
    # Let's use a loop or a precomputed index map.
    # Since this is called many times, precomputation is key.
    
    # Precompute index map once? 
    # It depends on N, which is fixed.
    
    # Let's generate the index map for pairs
    # i < j
    # idx in pdist for (i, j) is i*N - i*(i+1)/2 + j - i - 1
    
    # We can create arrays I and J of length len(d)
    # But doing this inside the gradient function is slow.
    # We should pass it or make it global.
    
    global _pair_indices_I, _pair_indices_J
    if not hasattr(_soft_min_value_and_grad, '_init_done'):
         _pair_indices_I = np.zeros(len(d), dtype=int)
         _pair_indices_J = np.zeros(len(d), dtype=int)
         idx = 0
         for i in range(N):
             for j in range(i+1, N):
                 _pair_indices_I[idx] = i
                 _pair_indices_J[idx] = j
                 idx += 1
         _soft_min_value_and_grad._init_done = True
         
    I = _pair_indices_I
    J = _pair_indices_J
    
    # w_pair = w[4*N:]
    w_pair = w[4*N:]
    
    # For each pair (i, j):
    # v = 0.5 * ||c_i - c_j||
    # dv/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # dv/dc_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
    
    # We have d (distances).
    # vecs = c[I] - c[J]
    vecs = c[I] - c[J]
    # norms are d
    # unit_vecs = vecs / d[:, np.newaxis]
    
    # Avoid division by zero (should not happen if points distinct)
    d_safe = np.where(d > 1e-12, d, 1.0)
    unit_vecs = vecs / d_safe[:, np.newaxis]
    
    # Contribution to dm_dc from pair (i,j) with weight w_pair[k]:
    # dm_dc[i] += w_pair[k] * 0.5 * unit_vecs[k]
    # dm_dc[j] += w_pair[k] * 0.5 * (-unit_vecs[k])
    
    # Use np.add.at for accumulation
    coeff = 0.5 * w_pair
    
    np.add.at(dm_dc, I, coeff[:, np.newaxis] * unit_vecs)
    np.add.at(dm_dc, J, -coeff[:, np.newaxis] * unit_vecs)
    
    # Gradient of objective is - dm_dc
    grad = -dm_dc
    
    return obj_val, grad

def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    # Try to fit 200 circles
    # Hexagonal packing density is pi/sqrt(12) ~ 0.9069
    # Area of square 1. Area of 200 circles 200 * pi * r^2.
    # 200 * pi * r^2 <= 1 * 0.9069 => r^2 <= 0.9069 / (200*pi) => r <= sqrt(0.00144) ~ 0.038
    # So r ~ 0.038 is a good target.
    
    best_c = None
    best_r = 0.0
    
    # Search for a good lattice spacing
    for d in np.linspace(0.050, 0.060, 20):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            # Center the points
            c -= np.mean(c, axis=0)
            # Scale/shift to fit in square if needed? 
            # The lattice generation above already tries to fit.
            # But taking first N might bias to top-left.
            # Better to select N points that are most central?
            # Or just take the first N and let optimizer fix it.
            # For now, take first N.
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        # Fallback grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
        
    return best_c

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    best_c = _initialize_hexagonal()
    best_r = _compute_radius(best_c)
    
    # Optimization loop
    # Beta annealing
    beta = 5.0
    no_improve_count = 0
    max_no_improve = 5
    
    while time.time() < deadline - 5:
        # Run L-BFGS-B
        def obj_and_grad(x):
            c = x.reshape(-1, 2)
            val, grad = _soft_min_value_and_grad(c, beta)
            return val, grad.flatten()
            
        try:
            res = minimize(
                obj_and_grad,
                best_c.flatten(),
                method='L-BFGS-B',
                jac=True, # Use analytical gradient
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            c_opt = res.x.reshape(-1, 2)
            c_opt = np.clip(c_opt, -0.5, 0.5)
            r_opt = _compute_radius(c_opt)
            
            if r_opt > best_r + 1e-8:
                best_r = r_opt
                best_c = c_opt.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                # If stuck, jiggle
                if no_improve_count >= max_no_improve:
                    # Add small noise
                    noise_scale = best_r * 0.05
                    best_c = best_c + np.random.normal(0, noise_scale, size=(N, 2))
                    best_c = np.clip(best_c, -0.5, 0.5)
                    no_improve_count = 0
                    # Increase beta more aggressively if jiggled
                    beta *= 2.0
                    continue
            
        except Exception:
            pass
            
        # Increase beta
        beta *= 1.5
        
        # If beta is very high, the soft min is close to hard min.
        # If we are stuck, jiggle.
        if no_improve_count >= max_no_improve:
             noise_scale = best_r * 0.05
             best_c = best_c + np.random.normal(0, noise_scale, size=(N, 2))
             best_c = np.clip(best_c, -0.5, 0.5)
             no_improve_count = 0
             beta *= 2.0

    # Final polish with high beta
    if time.time() < deadline - 2:
        beta = 1000.0
        def obj_and_grad(x):
            c = x.reshape(-1, 2)
            val, grad = _soft_min_value_and_grad(c, beta)
            return val, grad.flatten()
            
        try:
            res = minimize(
                obj_and_grad,
                best_c.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            c_opt = res.x.reshape(-1, 2)
            c_opt = np.clip(c_opt, -0.5, 0.5)
            r_opt = _compute_radius(c_opt)
            if r_opt > best_r:
                best_c = c_opt
                best_r = r_opt
        except Exception:
            pass
            
    return best_c
