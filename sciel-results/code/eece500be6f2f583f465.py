import numpy as np
import time
import scipy.optimize as opt

N = 300

def construct():
    start_time = time.time()
    time_limit = 140.0  # 10s safety margin
    
    # 1. Initialization: Hexagonal lattice
    # Estimate spacing
    # Area of unit circle is pi. N circles.
    # Hex packing density is pi/sqrt(12) approx 0.9069
    # r * sqrt(12) * N * 0.9069 approx pi
    # r approx sqrt(pi / (N * 12 * 0.9069)) ... 
    # Simpler: use the seed's spacing logic as a starting point
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    
    # Generate hex lattice points
    k = int(1.0 / spacing) + 3
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    points = np.array(rows)
    
    # Keep points inside unit circle
    radius = np.hypot(points[:, 0], points[:, 1])
    # We want to keep points that can fit a circle of radius ~ spacing/2
    # But for initialization, just keep those inside 1 - spacing/2
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    radius = radius[mask]
    
    # If we have more than N, select N closest to center (or random)
    if len(points) > N:
        # Select closest to center to ensure dense core
        idx = np.argsort(radius, kind="stable")[:N]
        points = points[idx]
    elif len(points) < N:
        # If too few, reduce spacing and retry
        while len(points) < N and spacing > 0.01:
            spacing *= 0.98
            k = int(1.0 / spacing) + 3
            rows = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    rows.append((x, y))
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            radius = radius[mask]
            if len(points) > N:
                idx = np.argsort(radius, kind="stable")[:N]
                points = points[idx]
                
    # Add small random noise to break symmetry
    points = points + np.random.normal(0, 1e-4, points.shape)
    
    # Ensure points are strictly inside
    # Clip if necessary, though soft-min handles it
    r = np.hypot(points[:, 0], points[:, 1])
    mask = r < 1.0
    if not np.all(mask):
         points = points[mask]
         # This shouldn't happen with init, but safety check
         if len(points) < N:
             # Re-init if broken
             return points[:N] if len(points)>=N else points
    
    # Objective function: Minimize -softmin(distances)
    # distances: pairwise half-distances and boundary distances
    
    def get_distances(c):
        # c is (N, 2)
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        # Upper triangle indices
        i, j = np.triu_indices(N, k=1)
        pair_dists = dists[i, j] / 2.0
        
        # Boundary
        radii = np.linalg.norm(c, axis=1)
        bound_dists = 1.0 - radii
        
        return pair_dists, bound_dists

    def softmin_value_and_grad(c, beta):
        pair_dists, bound_dists = get_distances(c)
        
        # Combine all distances
        # To avoid overflow in exp, we can use log-sum-exp trick
        # But for gradient, we need the weights w_k = exp(-beta * d_k) / sum(exp(-beta * d_k))
        # Let's compute log_weights = -beta * d_k
        # max_log = max(log_weights)
        # log_sum_exp = max_log + log(sum(exp(log_weights - max_log)))
        # value = -log_sum_exp / beta
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        # For numerical stability
        # We want sum(exp(-beta * d))
        # Let M = max(-beta * d)
        # sum(exp(-beta * d)) = exp(M) * sum(exp(-beta * d - M))
        # log(S) = M + log(sum(exp(-beta * d - M)))
        
        # If beta is large, -beta * d is very negative for small d.
        # Actually, we want to find the min d.
        # The term with min d will have the largest exp(-beta * d).
        
        # Let's just compute exp directly if beta * max_d is not too large.
        # max_d is at most 1 (diameter) or 1 (radius).
        # If beta = 1000, beta * d is up to 1000. exp(-1000) is 0.
        # This is fine.
        
        # However, if all d are large, exp(-beta*d) is tiny.
        # We can scale.
        
        # Let's use the standard stable logsumexp
        log_terms = -beta * all_dists
        max_log = np.max(log_terms)
        
        # If max_log is very negative, sum is 0.
        # But we subtract max_log, so we compute sum(exp(log_terms - max_log))
        # This sum is at least 1 (the max term becomes exp(0)=1).
        
        shifted = np.exp(log_terms - max_log)
        sum_shifted = np.sum(shifted)
        
        log_sum_exp = max_log + np.log(sum_shifted)
        
        value = -log_sum_exp / beta
        
        # Gradient
        # w_k = shifted[k] / sum_shifted
        weights = shifted / sum_shifted
        
        # d(value)/d(c_i) = -1/S * sum_k w_k * S * d(d_k)/d(c_i)  ... wait
        # value = -1/beta * log(S)
        # d(value)/dx = -1/beta * (1/S) * dS/dx
        # dS/dx = sum_k exp(-beta d_k) * (-beta) * d(d_k)/dx
        # dS/dx = -beta * sum_k (exp(-beta d_k) * d(d_k)/dx)
        # d(value)/dx = -1/beta * (1/S) * (-beta * sum_k (exp(-beta d_k) * d(d_k)/dx))
        # d(value)/dx = (1/S) * sum_k (exp(-beta d_k) * d(d_k)/dx)
        # d(value)/dx = sum_k ( (exp(-beta d_k) / S) * d(d_k)/dx )
        # d(value)/dx = sum_k ( weights[k] * d(d_k)/dx )
        
        grad = np.zeros_like(c)
        
        # Pairwise contributions
        # d(d_ij)/d(c_i) = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # d(d_ij)/d(c_j) = 0.5 * (c_j - c_i) / ||c_i - c_j||
        
        # We have pair_dists and weights for pairs (first N_pairs of all_dists)
        n_pairs = len(pair_dists)
        pair_weights = weights[:n_pairs]
        
        # We need to sum contributions for each i
        # For each pair (i, j), contribution to grad[i] is pair_weights[k] * 0.5 * (c_i - c_j)/d_ij_full
        # contribution to grad[j] is pair_weights[k] * 0.5 * (c_j - c_i)/d_ij_full
        
        # Using scatter_add or loops?
        # N=300, pairs=45000. Looping in Python is slow.
        # Use NumPy.
        
        # Reconstruct full distance matrix for gradients?
        # We already have dists (N, N) in get_distances, but we didn't return it.
        # Let's recompute or return it.
        # For efficiency, let's recompute inside gradient or pass it.
        # Recomputing is O(N^2). 300^2 = 90k. Fast.
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_full = np.linalg.norm(diff, axis=-1)
        # Avoid division by zero
        dists_full[dists_full < 1e-12] = 1e-12
        
        # grad_pair[i] = sum_j weights_ij * 0.5 * (c_i - c_j) / dist_ij
        # weights_ij is from pair_weights mapped to (i,j)
        
        # Create a weight matrix W (N, N)
        W = np.zeros((N, N))
        W[i, j] = pair_weights
        W[j, i] = pair_weights
        
        # grad = 0.5 * sum_j W_ij * (c_i - c_j) / dist_ij
        # (c_i - c_j) is diff[i, j]
        # So term is W[i, j] * diff[i, j] / dists_full[i, j]
        
        # Weighted sum over j
        # grad[i] = 0.5 * sum_j ( W[i, j] * diff[i, j] / dists_full[i, j] )
        
        # We can do this with einsum or broadcasting
        # factor = W / dists_full
        factor = W / dists_full
        # grad[i] = 0.5 * sum_j factor[i, j] * diff[i, j]
        # diff is (N, N, 2)
        # factor is (N, N)
        # We need to sum over axis 1
        
        # grad = 0.5 * np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        grad_pair = 0.5 * np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary contributions
        bound_weights = weights[n_pairs:]
        # d(d_i)/d(c_i) = -c_i / ||c_i||
        radii = np.linalg.norm(c, axis=1)
        radii[radii < 1e-12] = 1e-12
        grad_bound = -bound_weights[:, np.newaxis] * c / radii[:, np.newaxis]
        
        grad = grad_pair + grad_bound
        
        return value, grad

    # Optimization loop
    best_score = 0
    best_points = points.copy()
    
    # Initial score
    p_d, b_d = get_distances(points)
    all_d = np.concatenate([p_d, b_d])
    current_score = np.min(all_d)
    if current_score > best_score:
        best_score = current_score
        best_points = points.copy()

    beta = 10.0
    # Increase beta gradually
    # We want to go from smooth to sharp
    # Start with small beta to find global structure, then refine
    # But L-BFGS-B is a local optimizer.
    # We can run multiple restarts with different betas or just increase beta.
    
    # Let's try a sequence of betas
    betas = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    for b in betas:
        if time.time() - start_time > time_limit:
            break
            
        # Define objective for scipy
        def obj_and_grad(x):
            c = x.reshape(N, 2)
            val, grad = softmin_value_and_grad(c, b)
            return val, grad.reshape(-1)
            
        # L-BFGS-B
        # maxiter: limit iterations to prevent hanging
        # maxfun: limit function evals
        try:
            res = opt.minimize(
                lambda x: obj_and_grad(x)[0], 
                points.reshape(-1), 
                jac=lambda x: obj_and_grad(x)[1], 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_points = res.x.reshape(N, 2)
            
            # Calculate actual score
            p_d, b_d = get_distances(new_points)
            all_d = np.concatenate([p_d, b_d])
            new_score = np.min(all_d)
            
            # Check validity
            if np.all(np.isfinite(new_points)) and new_score > 0:
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    
        except Exception as e:
            # If optimization fails, keep previous best
            pass

    return best_points
