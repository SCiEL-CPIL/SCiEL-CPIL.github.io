"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hexagonal_initial():
    """Generate initial hexagonal lattice packing."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]].copy()
        spacing *= 0.98


def _compute_score(points):
    """Compute the maximum common radius for given points."""
    if len(points) < 2:
        return 0.0
    # Vectorized pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(len(points), k=1)
    min_pair = np.min(dists[iu]) / 2.0
    norms = np.hypot(points[:, 0], points[:, 1])
    min_boundary = 1.0 - np.max(norms)
    return min(min_pair, min_boundary)


def _soft_min_obj(points_flat, beta):
    """Smooth objective: negative soft-min of (pair_dists, boundary_dists)."""
    points = points_flat.reshape(N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary_dists = 1.0 - norms
    all_dists = np.concatenate([pair_dists, boundary_dists])
    # soft_min via log-sum-exp
    scaled = -beta * all_dists
    max_val = np.max(scaled)
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    soft_min = log_sum_exp / beta  # this is actually -soft_min since scaled = -beta*x
    # Wait: soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    # So soft_min = -(1/beta)*(max_val + log(sum(exp(scaled - max_val))))
    # where scaled = -beta*x
    # Let me redo:
    # soft_min = -(1/beta) * log(sum(exp(-beta * x_i)))
    # Let y_i = -beta * x_i
    # log(sum(exp(y_i))) = max(y) + log(sum(exp(y - max(y))))
    # soft_min = -(1/beta) * [max(y) + log(sum(exp(y - max(y))))]
    soft_min_val = -(log_sum_exp) / beta
    return -soft_min_val  # minimize this = maximize soft_min


def _soft_min_obj_grad(points_flat, beta):
    """Objective and gradient for L-BFGS-B."""
    points = points_flat.reshape(N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary_dists = 1.0 - norms
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    scaled = -beta * all_dists
    max_val = np.max(scaled)
    exp_scaled = np.exp(scaled - max_val)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp  # softmax weights
    
    # soft_min = -(1/beta) * (max_val + log(sum_exp))
    # d(soft_min)/d(x_i) = -(1/beta) * d/dx_i [max_val + log(sum_exp)]
    # = -(1/beta) * sum_j weights_j * d(y_j)/dx_i where y_j = -beta*x_j
    # = -(1/beta) * sum_j weights_j * (-beta * dx_j/dx_i)
    # = sum_j weights_j * dx_j/dx_i
    
    # Compute gradients
    grad = np.zeros_like(points_flat)
    
    # Pair distance gradients
    # For pair (i,j): dist_ij = |p_i - p_j|
    # d(dist_ij)/dp_i = (p_i - p_j) / dist_ij
    # d(dist_ij)/dp_j = (p_j - p_i) / dist_ij
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = weights[idx]
        d = pair_dists[idx]
        if d > 1e-15:
            direction = (points[i] - points[j]) / d
            grad[2*i] += w * direction[0]
            grad[2*i+1] += w * direction[1]
            grad[2*j] -= w * direction[0]
            grad[2*j+1] -= w * direction[1]
    
    # Boundary distance gradients
    # boundary_dist_i = 1 - |p_i|
    # d(boundary_dist_i)/dp_i = -p_i / |p_i|
    for idx in range(N):
        w = weights[N + idx]  # boundary distances start at index N
        norm = norms[idx]
        if norm > 1e-15:
            grad[2*idx] -= w * points[idx, 0] / norm
            grad[2*idx+1] -= w * points[idx, 1] / norm
    
    soft_min_val = -(max_val + np.log(sum_exp)) / beta
    obj = -soft_min_val
    return obj, grad


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    
    # Start with hexagonal lattice
    best_points = _hexagonal_initial()
    best_score = _compute_score(best_points)
    
    # Multi-restart L-BFGS-B optimization with increasing beta
    rng = np.random.default_rng(42)
    
    for restart in range(8):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            x0 = best_points.flatten().copy()
        else:
            # Perturb best points
            x0 = best_points.flatten().copy()
            noise_scale = 0.02 * np.exp(-0.3 * restart)
            x0 += rng.normal(0, noise_scale, size=2*N)
            # Ensure points stay in unit circle
            pts = x0.reshape(N, 2)
            norms = np.hypot(pts[:, 0], pts[:, 1])
            mask = norms > 1.0
            if np.any(mask):
                pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.999
            x0 = pts.flatten()
        
        # Use L-BFGS-B with analytic gradient
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > deadline - 10.0:
                break
            
            result = minimize(
                _soft_min_obj_grad,
                x0,
                method='L-BFGS-B',
                args=(beta,),
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 20}
            )
            
            if result.success or result.fun < 1e10:
                opt_points = result.x.reshape(N, 2)
                
                # Verify validity
                if np.all(np.isfinite(opt_points)):
                    norms = np.hypot(opt_points[:, 0], opt_points[:, 1])
                    if np.all(norms <= 1.0):
                        score = _compute_score(opt_points)
                        if score > best_score:
                            best_score = score
                            best_points = opt_points.copy()
                            x0 = opt_points.flatten().copy()
    
    # Final polish with high beta
    if time.time() < deadline - 5.0:
        for beta in [200.0, 500.0]:
            if time.time() > deadline - 3.0:
                break
            result = minimize(
                _soft_min_obj_grad,
                best_points.flatten(),
                method='L-BFGS-B',
                args=(beta,),
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 10}
            )
            if result.success or result.fun < 1e10:
                opt_points = result.x.reshape(N, 2)
                if np.all(np.isfinite(opt_points)):
                    norms = np.hypot(opt_points[:, 0], opt_points[:, 1])
                    if np.all(norms <= 1.0):
                        score = _compute_score(opt_points)
                        if score > best_score:
                            best_score = score
                            best_points = opt_points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
