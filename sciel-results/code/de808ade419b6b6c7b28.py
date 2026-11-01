import numpy as np
import time
import scipy.optimize as opt

N = 50

def _hex_lattice_init(n):
    """Generate initial points on a hexagonal lattice inside the unit circle."""
    # Adjust spacing to fit n points roughly
    # Density approx pi / (2 * sqrt(3) * r^2) ? 
    # Area of hexagon side s is 2*sqrt(3)*s^2. 
    # Let's just use a fixed spacing and filter.
    spacing = 0.25 # Initial guess
    points = []
    # Generate grid
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * spacing + (0.5 * spacing if j % 2 else 0)
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
    
    points = np.array(points)
    # If we have too many, take the closest to center? Or random?
    # Taking closest to center might cluster them. 
    # Taking a random subset or those with best local density is better.
    # For now, let's take the first N or sort by radius.
    if len(points) > n:
        # Sort by distance to origin to keep them central? 
        # Or just take a random sample?
        # The seed took argsort(radius)[:N]. Let's do that.
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii)
        points = points[idx[:n]]
    elif len(points) < n:
        # Need to generate more or adjust spacing
        # This simple init might fail if spacing is wrong.
        # Let's try to fill in.
        # For robustness, if not enough, add random points inside.
        while len(points) < n:
            x, y = np.random.uniform(-1, 1, 2)
            if np.hypot(x, y) < 0.9:
                points = np.vstack([points, [x, y]])
    return points

def _compute_score(points):
    """Calculate the valid packing radius."""
    if len(points) != N:
        return 0.0
    if np.any(~np.isfinite(points)):
        return 0.0
    if np.any(np.hypot(points[:, 0], points[:, 1]) >= 1.0):
        return 0.0
    
    min_dist = np.inf
    # Pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    
    # Boundary distances
    for i in range(N):
        d = 1.0 - np.linalg.norm(points[i])
        if d < min_dist:
            min_dist = d
            
    return min_dist / 2.0

def _objective_and_grad(x_flat, tau=10.0):
    """
    Objective: Minimize -softmin(constraints)
    Which is equivalent to minimizing the log-sum-exp of negative constraints.
    Constraints c_k >= 0 are required for validity, but we optimize the value.
    c_k = 1 - |x_i| or |x_i - x_j|/2.
    We want to maximize min(c_k).
    softmin(c) = -1/tau * log(sum(exp(-tau * c_k)))
    We want to maximize softmin, so minimize -softmin = 1/tau * log(sum(exp(-tau * c_k)))
    """
    points = x_flat.reshape(-1, 2)
    
    # Boundary constraints
    radii = np.linalg.norm(points, axis=1)
    c_boundary = 1.0 - radii
    
    # Pairwise constraints
    # Compute all pairwise distances
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf to ignore self
    np.fill_diagonal(dists, np.inf)
    c_pair = dists / 2.0
    
    # Combine all constraints into a single array
    # c_boundary has N elements. c_pair has N*(N-1) elements (symmetric, so we can just use upper triangle or all).
    # Using all is fine for log-sum-exp, just doubles the weight of pairs.
    # To be precise, let's use unique pairs.
    i_idx, j_idx = np.triu_indices(N, k=1)
    c_pair_unique = dists[i_idx, j_idx] / 2.0
    
    all_c = np.concatenate([c_boundary, c_pair_unique])
    
    # Log-sum-exp for stability
    # We want to minimize: (1/tau) * log(sum(exp(-tau * c)))
    # Let z = -tau * c
    # log(sum(exp(z))) = max(z) + log(sum(exp(z - max(z))))
    z = -tau * all_c
    z_max = np.max(z)
    if np.isinf(z_max):
        # Handle case where some c are inf or -inf
        # If c is -inf, z is inf. exp(inf) is inf. 
        # This happens if points coincide (dist 0 -> c 0 -> z 0) or outside (c negative -> z positive).
        # If c is very negative, z is very positive.
        pass
        
    # For numerical stability, subtract max
    z_shifted = z - z_max
    sum_exp = np.sum(np.exp(z_shifted))
    log_sum_exp = z_max + np.log(sum_exp)
    
    val = (1.0 / tau) * log_sum_exp
    
    # Gradient
    # d/dx [ (1/tau) * log(sum(exp(-tau * c_k))) ]
    # = (1/tau) * (1/sum(exp(-tau c_k))) * sum( exp(-tau c_k) * (-tau * dc_k/dx) )
    # = - sum( exp(-tau c_k) / sum(exp(-tau c_k)) * dc_k/dx )
    # Let p_k = exp(-tau c_k) / sum(exp(-tau c_k))
    # grad = - sum( p_k * grad(c_k) )
    
    p = np.exp(z_shifted) / sum_exp
    
    grad_c_boundary = -points / np.maximum(radii, 1e-8)[:, np.newaxis] # dc/dx = -x/|x|
    # Note: if radius is 0, grad is undefined, but we assume points are distinct.
    
    # grad for pairs
    # c_ij = |x_i - x_j| / 2
    # dc_ij/dx_i = (x_i - x_j) / (2 * |x_i - x_j|)
    # dc_ij/dx_j = (x_j - x_i) / (2 * |x_i - x_j|)
    
    # We need to accumulate gradients for each point.
    grad = np.zeros_like(points)
    
    # Boundary contribution
    # p_boundary are the first N elements of p
    p_b = p[:N]
    grad += np.sum(p_b[:, np.newaxis] * grad_c_boundary, axis=0)
    # Wait, the sign.
    # Objective to minimize: f = (1/tau) log sum exp(-tau c)
    # df/dx = - sum p_k (dc_k/dx)
    # So we subtract the weighted gradients of c.
    # My formula above: grad = - sum( p_k * grad(c_k) )
    # So:
    grad -= np.sum(p_b[:, np.newaxis] * grad_c_boundary, axis=0)
    
    # Pairwise contribution
    p_p = p[N:]
    # We need to sum over all pairs.
    # For each pair (i,j) with weight p_k:
    # grad_i += - p_k * (x_i - x_j) / (2 * d_ij)
    # grad_j += - p_k * (x_j - x_i) / (2 * d_ij)
    
    # Vectorized update is tricky with unique pairs.
    # Loop over pairs for gradient? N=50, ~1200 pairs. 1200 iterations per step is fine.
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        w = p_p[k]
        if w < 1e-10:
            continue
        d = dists[i, j]
        if d < 1e-8:
            continue
        vec = points[i] - points[j]
        unit_vec = vec / d
        # dc/dx_i = unit_vec / 2
        # dc/dx_j = -unit_vec / 2
        grad[i] -= w * unit_vec * 0.5
        grad[j] += w * unit_vec * 0.5
        
    return val, grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial guess
    x0 = _hex_lattice_init(N).flatten()
    
    # Optimization
    best_x = x0
    best_score = _compute_score(x0.reshape(-1, 2))
    
    # Try BFGS
    try:
        res = opt.minimize(
            _objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 2000, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.x is not None:
            candidate = res.x
            score = _compute_score(candidate.reshape(-1, 2))
            if score > best_score:
                best_score = score
                best_x = candidate
    except Exception:
        pass
    
    # If time remains, try a second round with different init or polish
    if time.time() < deadline - 20.0:
        # Try random restarts?
        # Or just polish the current best with a smaller tau (sharper min)?
        try:
            res2 = opt.minimize(
                lambda x: _objective_and_grad(x, tau=20.0)[0], 
                best_x, 
                method='L-BFGS-B', 
                jac=lambda x: _objective_and_grad(x, tau=20.0)[1],
                options={'maxiter': 1000, 'maxfun': 2000}
            )
            if res2.x is not None:
                candidate = res2.x
                score = _compute_score(candidate.reshape(-1, 2))
                if score > best_score:
                    best_score = score
                    best_x = candidate
        except Exception:
            pass

    # Final validity check and return
    final_points = best_x.reshape(-1, 2)
    if _compute_score(final_points) <= 0:
        # Fallback to seed-like behavior if optimization failed
        return _hex_lattice_init(N)
        
    return final_points
