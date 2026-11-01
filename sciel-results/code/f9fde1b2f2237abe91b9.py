import numpy as np
import time
from scipy.optimize import minimize

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 110.0

def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    r_pair = 0.5 * np.min(d)
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0

def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97

def _soft_min_obj_grad(points, beta):
    """
    Compute the soft-min objective and its gradient.
    Objective: -softmin( concat(0.5 * dist_ij, 1 - norm_i) )
    We want to MINIMIZE this objective, which maximizes the soft-min radius.
    """
    N = points.shape[0]
    
    # 1. Pairwise distances
    diff = points[:, None, :] - points[None, :, :]
    dists_full = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists_full, np.inf)
    
    # Boundary terms
    norms = np.linalg.norm(points, axis=1)
    S_bounds = 1.0 - norms
    
    # Indices for i<j
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dists_full[i_idx, j_idx]
    S_pairs = 0.5 * d_ij
    
    # All S values
    S_all = np.concatenate([S_pairs, S_bounds])
    
    # Calculate weights using log-sum-exp for stability
    # We want w_k = exp(-beta * S_k) / Sum(exp(-beta * S_j))
    # Let u = -beta * S.
    u = -beta * S_all
    max_u = np.max(u)
    exp_u = np.exp(u - max_u)
    Z = np.sum(exp_u)
    weights = exp_u / Z
    
    w_pairs = weights[:len(S_pairs)]
    w_bounds = weights[len(S_pairs):]
    
    # Objective value: -softmin = -( -1/beta * log(Sum(exp(-beta*S))) ) = 1/beta * log(Sum(exp(-beta*S)))
    # log(Sum(exp(-beta*S))) = max_u + log(Z)
    obj = 1.0 / beta * (max_u + np.log(Z))
    
    # Gradient
    grad = np.zeros_like(points)
    
    # Pair contributions
    # For each pair (i, j) with weight w:
    # dS_pair/dp_i = 0.5 * (p_i - p_j) / d_ij
    # grad[i] += w * dS_pair/dp_i
    # grad[j] += w * dS_pair/dp_j = w * 0.5 * (p_j - p_i) / d_ij = -w * dS_pair/dp_i
    
    diff_ij = points[i_idx] - points[j_idx]
    d_safe = np.where(d_ij < 1e-9, 1e-9, d_ij)
    unit_vec = diff_ij / d_safe[:, None]
    
    # dS/dp_i vector
    contrib_i = w_pairs[:, None] * 0.5 * unit_vec
    # dS/dp_j vector is -contrib_i
    
    np.add.at(grad, i_idx, contrib_i)
    np.add.at(grad, j_idx, -contrib_i)
    
    # Boundary contributions
    # S_bound_i = 1 - norm_i
    # dS_bound_i/dp_i = -p_i / norm_i
    # grad[i] += w_bound_i * (-p_i / norm_i)
    
    r_safe = np.where(norms < 1e-9, 1e-9, norms)
    bound_grad = -w_bounds[:, None] * points / r_safe[:, None]
    grad += bound_grad
    
    return obj, grad

def _optimize(points, beta, maxiter):
    """Run L-BFGS-B optimization for a given beta."""
    
    def obj_grad(x):
        p = x.reshape(N, 2)
        val, g = _soft_min_obj_grad(p, beta)
        return val, g.flatten()
    
    bounds = [(-1.5, 1.5)] * (N * 2)
    
    res = minimize(
        obj_grad, 
        points.flatten(), 
        method='L-BFGS-B', 
        jac=True, 
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    return res.x.reshape(N, 2)

def _polish(points):
    """
    Final polish using a very high beta to sharpen the min.
    Also tries to fix any small violations.
    """
    if time.time() - T_START > DEADLINE - 5.0:
        return points
    
    # Use a very high beta for the final polish
    beta = 5000.0
    
    def obj_grad(x):
        p = x.reshape(N, 2)
        val, g = _soft_min_obj_grad(p, beta)
        return val, g.flatten()
    
    bounds = [(-1.0, 1.0)] * (N * 2) # Tighter bounds for polish
    
    res = minimize(
        obj_grad, 
        points.flatten(), 
        method='L-BFGS-B', 
        jac=True, 
        bounds=bounds,
        options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
    )
    
    return res.x.reshape(N, 2)

def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Initial seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    best = pts.copy()
    best_r = _score(best)
    
    # Schedule of betas
    # Start low (smooth) to end high (sharp)
    # More intermediate steps for smoother transition
    betas = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    
    for beta in betas:
        if time.time() - t0 > DEADLINE - 15.0:
            break
            
        # Jitter to escape symmetry / local minima
        jitter = 0.005 * rng.standard_normal((N, 2))
        start_pts = best + jitter
        # Clip to disk roughly
        rad = np.linalg.norm(start_pts, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            start_pts[mask] = start_pts[mask] / rad[mask, None] * 0.99
            
        # Optimize
        try:
            # Reduce maxiter as beta increases? Or keep constant?
            # L-BFGS is fast.
            maxiter = 200
            opt_pts = _optimize(start_pts, beta, maxiter)
            r = _score(opt_pts)
            if r > best_r:
                best_r = r
                best = opt_pts
        except Exception:
            pass
            
    # Final polish
    if time.time() - t0 < DEADLINE - 5.0:
        try:
            polished = _polish(best)
            r = _score(polished)
            if r > best_r:
                best = polished
                best_r = r
        except Exception:
            pass
            
    return best
