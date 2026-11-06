"""Physics-based relaxation for packing 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 115.0  # seconds, safety margin before 120s limit


def _pairwise_distances(points):
    """Compute all pairwise distances between points.
    
    Returns array of shape (N, N) with distances.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=-1)


def _score(points):
    """Compute the packing score: min(min_pairwise_dist/2, min(1-|c_i|))."""
    if points.shape != (N, 2) or not np.all(np.isfinite(points)):
        return -1.0
    # Check all inside unit circle
    radii = np.linalg.norm(points, axis=1)
    if np.any(radii >= 1.0):
        return -1.0
    boundary_min = np.min(1.0 - radii)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    # Exclude self-pairs
    np.fill_diagonal(dists, np.inf)
    pair_min = np.min(dists) / 2.0
    return min(boundary_min, pair_min)


def _energy_and_grad(points, softness=100.0, wall_softness=50.0):
    """Compute smooth energy and gradient for the packing problem.
    
    Energy = -soft_min(pairwise_dists/2, boundary_dists)
    where soft_min uses log-sum-exp trick for the minimum.
    """
    n = points.shape[0]
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    radii = np.linalg.norm(points, axis=1)
    boundary = 1.0 - radii
    
    # Use soft-min via log-sum-exp for the minimum of all constraints
    # We want to maximize r = min(min_pair, min_boundary)
    # Equivalent to minimizing -r
    # Soft approximation: -log(sum(exp(-x_i / t))) * t  as t->0 gives -min(x_i)
    
    # Combine all constraints: pairwise dists/2 and boundary dists
    pair_half = dists / 2.0
    # Flatten pairwise (upper triangle only to avoid double counting)
    idx = np.triu_indices(n, k=1)
    pair_vals = pair_half[idx]
    all_vals = np.concatenate([pair_vals, boundary])
    
    # Soft-min: min(all_vals) approximated by -t * log(sum(exp(-all_vals/t)))
    t = softness
    # Clip to avoid overflow
    shifted = all_vals - np.min(all_vals)
    log_sum = np.log(np.sum(np.exp(-shifted / t))) + np.min(all_vals) / t
    soft_min_val = -t * log_sum  # This approximates -min(all_vals)... 
    
    # Actually let's use a cleaner formulation:
    # f(x) = -soft_min(all_vals) where soft_min approximates min
    # soft_min = t * log(sum(exp(x_i / t)))  (this is soft-max, not soft-min)
    # soft-min = -t * log(sum(exp(-x_i / t)))
    
    # Let me redo: we want to maximize r = min(constraints)
    # So we minimize E = -r ≈ t * log(sum(exp(-x_i/t)))
    # where x_i are the constraint values (pair_half and boundary)
    
    x = all_vals
    # For numerical stability
    x_min = np.min(x)
    exponents = -x / t
    exponents -= np.max(exponents)  # shift for stability
    log_sum_exp = np.log(np.sum(np.exp(exponents))) + np.max(exponents)
    energy = t * log_sum_exp
    
    # Gradient with respect to points
    # dE/dx_i = exp(-x_i/t) / sum(exp(-x_j/t))
    weights = np.exp(exponents)
    weights /= np.sum(weights)
    
    grad_points = np.zeros_like(points)
    
    # Pairwise part
    # For pair (i,j): x_k = |p_i - p_j|/2
    # d(x_k)/dp_i = (p_i - p_j) / (2 * |p_i - p_j|)
    # d(x_k)/dp_j = (p_j - p_i) / (2 * |p_i - p_j|)
    
    for k in range(len(pair_vals)):
        i, j = idx[0][k], idx[1][k]
        d = diff[i, j]
        dist = np.linalg.norm(d)
        if dist < 1e-12:
            continue
        w = weights[k]
        # dE/dp_i += w * d(dist/2)/dp_i = w * d/(2*dist)
        grad_points[i] += w * d / (2.0 * dist)
        grad_points[j] -= w * d / (2.0 * dist)
    
    # Boundary part
    for i in range(n):
        k = len(pair_vals) + i
        w = weights[k]
        r = radii[i]
        if r < 1e-12:
            # At origin, gradient is ill-defined; use small perturbation
            grad_points[i] -= w * 0.01 * np.array([1.0, 0.0])
        else:
            # d(boundary_i)/dp_i = d(1-|p_i|)/dp_i = -p_i/|p_i|
            # dE/dp_i = w * dE/dboundary_i * dboundary_i/dp_i
            # dE/dboundary_i = -w (since E = t*log(sum(exp(-x/t))), dE/dx_i = -w_i)
            # Wait, let me recheck.
            # E = t * log(sum(exp(-x_j/t)))
            # dE/dx_i = t * (1/sum) * exp(-x_i/t) * (-1/t) = -exp(-x_i/t)/sum = -w_i
            # So dE/dp_i = sum_k (-w_k) * dx_k/dp_i
            # For boundary: dx_i/dp_i = -p_i/r_i
            # dE/dp_i from boundary = (-w_i) * (-p_i/r_i) = w_i * p_i/r_i
            grad_points[i] += w * points[i] / r
    
    return energy, grad_points


def _fast_energy_and_grad(points, t=50.0):
    """Vectorized energy and gradient computation."""
    n = points.shape[0]
    
    # Pairwise
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # Boundary
    radii = np.linalg.norm(points, axis=1)
    boundary = 1.0 - radii
    
    # Gather all constraint values
    idx_i, idx_j = np.triu_indices(n, k=1)
    pair_vals = dists[idx_i, idx_j] / 2.0
    all_vals = np.concatenate([pair_vals, boundary])
    m = len(all_vals)
    
    # Soft energy: E = t * log(sum(exp(-x/t)))
    exponents = -all_vals / t
    exponents -= np.max(exponents)
    log_sum_exp = np.log(np.sum(np.exp(exponents))) + np.max(exponents)
    energy = t * log_sum_exp
    
    # Weights
    weights = np.exp(exponents)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(points)
    
    # Pairwise gradients
    pair_weights = weights[:len(pair_vals)]
    # For each pair (i,j), contribution to grad[i] and grad[j]
    # dE/dp_i gets -w_k * d(pair_val_k)/dp_i for each k where pair involves i
    # pair_val_k = dists[i,j]/2, d/dp_i = (p_i - p_j)/(2*dist)
    # So dE/dp_i = -w_k * (p_i - p_j)/(2*dist) ... wait sign
    # E = t*log(sum(exp(-x/t))), dE/dx = -w
    # x_k = dist/2, dx_k/dp_i = (p_i-p_j)/(2*dist)
    # dE/dp_i = sum_k (-w_k) * dx_k/dp_i
    
    for k in range(len(pair_vals)):
        i, j = idx_i[k], idx_j[k]
        d = diff[i, j]
        dist = dists[i, j]
        if dist < 1e-12:
            continue
        w = pair_weights[k]
        # dE/dp_i += -w * (p_i - p_j)/(2*dist)
        grad[i] -= w * d / (2.0 * dist)
        # dE/dp_j += -w * (p_j - p_i)/(2*dist) = w * (p_i - p_j)/(2*dist)
        grad[j] += w * d / (2.0 * dist)
    
    # Boundary gradients
    boundary_weights = weights[len(pair_vals):]
    for i in range(n):
        r = radii[i]
        w = boundary_weights[i]
        if r < 1e-12:
            continue
        # x_i = 1 - r, dx_i/dp_i = -p_i/r
        # dE/dp_i += -w * (-p_i/r) = w * p_i/r
        grad[i] += w * points[i] / r
    
    return energy, grad


def _anneal_energy_and_grad(points, t):
    """Energy and gradient with annealing parameter t."""
    return _fast_energy_and_grad(points, t)


def _optimize_single_start(x0, t_start=100.0, t_end=5.0, n_iters=2000, deadline=None):
    """Run gradient descent with annealing from a single start."""
    x = x0.copy()
    best_x = x0.copy()
    best_score = _score(x0)
    
    t = t_start
    lr = 0.01
    decay = (t_end / t_start) ** (1.0 / n_iters)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute energy and gradient
        e, g = _fast_energy_and_grad(x, t)
        
        # Gradient descent step
        x_new = x - lr * g
        
        # Validate
        if not np.all(np.isfinite(x_new)):
            break
        
        # Check if inside unit circle (with small tolerance)
        radii = np.linalg.norm(x_new, axis=1)
        if np.any(radii > 1.05):
            # Project back
            mask = radii > 1.0
            x_new[mask] = x_new[mask] / radii[mask][:, None] * 0.99
            if np.any(np.linalg.norm(x_new, axis=1) > 1.0):
                x_new = x  # revert
                lr *= 0.5
                continue
        
        x = x_new
        t *= decay
        
        # Occasionally check score
        if it % 50 == 0:
            s = _score(x)
            if s > best_score:
                best_score = s
                best_x = x.copy()
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            # Reduce learning rate if energy is increasing
            pass
    
    # Final score check
    s = _score(x)
    if s > best_score:
        best_score = s
        best_x = x.copy()
    
    return best_x, best_score


def _polish(points):
    """Final polish using L-BFGS-B on the smooth objective."""
    def neg_score_smooth(x_flat, t=10.0):
        x = x_flat.reshape(N, 2)
        e, g = _fast_energy_and_grad(x, t)
        return e, g.reshape(-1)
    
    x0 = points.reshape(-1)
    try:
        res = minimize(neg_score_smooth, x0, method='L-BFGS-B',
                       options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10})
        x_opt = res.x.reshape(N, 2)
        # Check validity
        if not np.all(np.isfinite(x_opt)):
            return points
        radii = np.linalg.norm(x_opt, axis=1)
        if np.any(radii >= 1.0):
            # Project
            mask = radii >= 1.0
            x_opt[mask] = x_opt[mask] / radii[mask][:, None] * 0.999
            if np.any(np.linalg.norm(x_opt, axis=1) >= 1.0):
                return points
        return x_opt
    except Exception:
        return points


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Generate diverse initial configurations
    np.random.seed(42)
    
    best_points = None
    best_score = -1.0
    
    # Strategy 1: Multiple random starts with gradient descent
    n_starts = 8
    for s in range(n_starts):
        if time.time() > deadline - 5.0:
            break
        
        if s == 0:
            # Start with hexagonal lattice
            spacing = 2.0 / (np.sqrt(N) + 1.0)
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points = points[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                points = points[np.argsort(radius, kind="stable")[:N]]
            else:
                # Pad with random points
                while len(points) < N:
                    angle = np.random.uniform(0, 2 * np.pi)
                    r = np.random.uniform(0, 0.8)
                    points = np.vstack([points, [[r * np.cos(angle), r * np.sin(angle)]]])
        elif s == 1:
            # Random uniform in disk
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 0.81, N))
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        elif s == 2:
            # Ring-like distribution
            n_rings = 5
            points = np.zeros((N, 2))
            for ring in range(n_rings):
                r = (ring + 1) / (n_rings + 0.5) * 0.9
                n_in_ring = N // n_rings + (1 if ring < N % n_rings else 0)
                start_idx = ring * (N // n_rings)
                end_idx = start_idx + n_in_ring
                if start_idx >= N:
                    break
                end_idx = min(end_idx, N)
                angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False)
                points[start_idx:end_idx, 0] = r * np.cos(angles)
                points[start_idx:end_idx, 1] = r * np.sin(angles)
            # Fill remaining
            remaining = N - len(points[points[:, 0] != 0 | points[:, 1] != 0])
            if remaining > 0:
                angles = np.random.uniform(0, 2 * np.pi, remaining)
                radii = np.random.uniform(0.1, 0.5, remaining)
                extra = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
                points = np.vstack([points[:N - remaining], extra])
        else:
            # Perturbed random
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 0.85, N))
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            # Add noise
            points += np.random.normal(0, 0.05, points.shape)
        
        # Ensure valid starting point
        radii_check = np.linalg.norm(points, axis=1)
        points[radii_check > 0.95] = points[radii_check > 0.95] / radii_check[radii_check > 0.95][:, None] * 0.9
        
        # Run annealed gradient descent
        x_opt, score = _optimize_single_start(points, t_start=80.0, t_end=3.0, 
                                               n_iters=1500, deadline=deadline - 3.0)
        
        if score > best_score:
            best_score = score
            best_points = x_opt.copy()
    
    # Polish the best result
    if best_points is not None and time.time() < deadline - 5.0:
        polished = _polish(best_points)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished
    
    # Final validation
    if best_points is None:
        # Fallback: hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            best_points = points[np.argsort(radius, kind="stable")[:N]]
        else:
            best_points = np.zeros((N, 2))
            for i in range(N):
                angle = 2 * np.pi * i / N
                r = 0.5
                best_points[i] = [r * np.cos(angle), r * np.sin(angle)]
    
    # Final check
    final_score = _score(best_points)
    if final_score < 0:
        # Invalid, use fallback
        best_points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.3
            best_points[i] = [r * np.cos(angle), r * np.sin(angle)]
    
    return best_points
