"""Packing of 100 equal spheres in a unit cube. Uses a grid seed + L-BFGS-B on a smooth
penalized objective, plus a final radius polish pass."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(c):
    """Compute the largest common radius for centres c (N,3) inside [-0.5,0.5]^3."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Wall distances
    wall = 0.5 - np.abs(c)
    min_wall = wall.min()
    if min_wall <= 0:
        return 0.0
    # Pairwise distances
    # Use chunked to limit memory
    r = min_wall
    for i in range(N):
        d = np.linalg.norm(c - c[i], axis=1)
        d[i] = np.inf
        r = min(r, d.min() / 2.0)
    return r

def _grid_seed():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _objective(x, c0):
    """Smooth penalized objective: we want to maximize r.
    Use a soft-min over all constraints (pairwise and wall) with a temperature parameter.
    Minimize -soft_min."""
    c = x.reshape(N, 3)
    # Wall terms: 0.5 - |c_ik|
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()
    # Pairwise: |c_i - c_j| / 2
    # For efficiency, compute all pairwise
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    pair = dist / 2.0
    pair_flat = pair.ravel()
    
    # Soft-min via log-sum-exp (negative)
    # soft_min(a) = -logsumexp(-a / t) * t  (for small t)
    # We use a fixed temperature that gives a smooth approximation
    t = 0.01
    all_constraints = np.concatenate([wall_flat, pair_flat])
    # Clip to avoid underflow
    all_constraints = np.clip(all_constraints, 1e-10, None)
    neg_scaled = -all_constraints / t
    # Subtract max for numerical stability
    m_neg = np.max(neg_scaled)
    logsum = m_neg + np.log(np.sum(np.exp(neg_scaled - m_neg)))
    soft_min_val = -logsum * t
    return -soft_min_val

def _objective_grad(x, c0):
    """Gradient of the soft-min objective (numerical fallback not needed if we use finite differences,
    but let's try analytic for speed. Actually, let's use finite differences for correctness first,
    then switch if needed. Given 300 variables, finite differences might be slow but acceptable for a few iterations.
    Let's try analytic gradient for the soft-min."""
    c = x.reshape(N, 3)
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    pair = dist / 2.0
    pair_flat = pair.ravel()
    
    t = 0.01
    all_constraints = np.concatenate([wall_flat, pair_flat])
    all_constraints = np.clip(all_constraints, 1e-10, None)
    
    # Weights for soft-min: w_i = exp(-a_i / t) / sum_j exp(-a_j / t)
    neg_scaled = -all_constraints / t
    m_neg = np.max(neg_scaled)
    exp_vals = np.exp(neg_scaled - m_neg)
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of soft_min w.r.t. a_i is -w_i
    # Gradient of objective = -soft_min, so grad_obj = -grad(soft_min) = w_i * grad(a_i)
    
    grad = np.zeros_like(x)
    
    # Wall gradients: a = 0.5 - |c_k|
    # d a / d c_k = -sign(c_k)
    wall_weights = weights[:N*3].reshape(N, 3)
    # d(0.5 - |c|)/dc = -sign(c)
    wall_grad = -np.sign(c)
    # If c is 0, sign is 0, which is fine (subgradient)
    grad += (wall_weights * wall_grad).ravel()
    
    # Pairwise gradients: a_ij = |c_i - c_j| / 2
    # d a_ij / d c_i = (c_i - c_j) / (2 * |c_i - c_j|)
    # d a_ij / d c_j = (c_j - c_i) / (2 * |c_i - c_j|)
    pair_weights = weights[N*3:].reshape(N, N)
    # Zero out diagonal
    np.fill_diagonal(pair_weights, 0)
    
    # For each i, sum over j: w_ij * (c_i - c_j) / (2 * |c_i - c_j|)
    # And for j, sum over i: w_ij * (c_j - c_i) / (2 * |c_i - c_j|)
    # Note: w_ij = w_ji due to symmetry of distances, but our matrix is full.
    
    # Compute unit vectors
    dist_safe = np.where(dist > 1e-12, dist, 1.0)
    unit = diff / dist_safe[:, :, np.newaxis]
    # For diagonal, unit is 0/1 = 0, which is fine
    
    # grad_i = sum_j w_ij * unit_ij / 2
    # grad_j = sum_i w_ij * (-unit_ij) / 2 = -sum_i w_ij * unit_ij / 2
    
    grad_i = np.sum(pair_weights[:, :, np.newaxis] * unit / 2.0, axis=1)
    grad_j = np.sum(pair_weights[:, :, np.newaxis] * unit / 2.0, axis=0)
    
    # grad_j corresponds to the second index, which is c_j
    # So grad for c_i gets grad_i, and grad for c_j gets -grad_j? 
    # Let's re-derive:
    # a_ij = |c_i - c_j|/2
    # d a_ij / d c_i = (c_i - c_j) / (2|c_i - c_j|) = unit_ij / 2
    # d a_ij / d c_j = (c_j - c_i) / (2|c_i - c_j|) = -unit_ij / 2
    
    # So total gradient for c_i = sum_j w_ij * (unit_ij / 2)
    # Total gradient for c_j = sum_i w_ij * (-unit_ij / 2)
    
    # My grad_i calculation: sum_j w_ij * unit_ij / 2 -> correct for c_i
    # My grad_j calculation: sum_i w_ij * unit_ij / 2 -> this is sum_i w_ij * unit_ij / 2
    # But we need sum_i w_ij * (-unit_ij / 2) = -grad_j_calc
    
    # Wait, pair_weights[i,j] is w_ij.
    # grad_i[i] = sum_j w_ij * unit[i,j] / 2
    # grad_j[j] = sum_i w_ij * unit[i,j] / 2  <-- this is sum_i w_ij * unit[i,j] / 2
    # The gradient contribution to c_j from term (i,j) is w_ij * (-unit[i,j] / 2)
    # So total grad for c_j = - sum_i w_ij * unit[i,j] / 2 = -grad_j[j]
    
    grad += (grad_i - grad_j).ravel()
    
    return grad

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # Start with grid seed
    c0 = _grid_seed()
    
    # Try a few random perturbations of the grid
    best_c = c0.copy()
    best_r = _compute_radius(best_c)
    
    rng = np.random.RandomState(42)
    
    # Optimize from grid
    x0 = c0.ravel()
    try:
        res = minimize(
            _objective, x0, args=(c0,),
            jac=_objective_grad,
            method='L-BFGS-B',
            bounds=[(-0.49, 0.49)] * (N * 3),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        elapsed = time.time() - start_time
        if elapsed < deadline:
            c_new = res.x.reshape(N, 3)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
    except Exception:
        pass
    
    # Try random restarts if time permits
    for restart in range(5):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        # Random jitter
        c_rand = c0 + rng.uniform(-0.05, 0.05, size=(N, 3))
        c_rand = np.clip(c_rand, -0.49, 0.49)
        x0 = c_rand.ravel()
        
        try:
            res = minimize(
                _objective, x0, args=(c0,),
                jac=_objective_grad,
                method='L-BFGS-B',
                bounds=[(-0.49, 0.49)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            c_new = res.x.reshape(N, 3)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    
    # Final polish: try to increase radius by moving spheres
    # Use a simple local search: for each sphere, try small moves
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        c = best_c.copy()
        r = best_r
        improved = True
        while improved and (time.time() - start_time) < deadline - 10:
            improved = False
            for i in range(N):
                # Try small perturbations
                for delta in [0.001, 0.0005, -0.001, -0.0005]:
                    c_test = c.copy()
                    c_test[i, 0] += delta
                    c_test = np.clip(c_test, -0.499, 0.499)
                    r_test = _compute_radius(c_test)
                    if r_test > r + 1e-10:
                        c = c_test
                        r = r_test
                        improved = True
                        break
        if r > best_r:
            best_c = c
            best_r = r
    
    return best_c
