import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _exact_radius(C):
    """Compute the exact packing radius."""
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    if n < 2:
        return np.inf
    
    # Pairwise distances
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distance
    min_bnd = np.min(0.5 - np.abs(C))
    
    return min(min_pair, min_bnd)

def _soft_min_radius_and_grad(C, beta):
    """
    Compute the negative soft-min radius and its gradient.
    Objective: minimize -softmin(radius_terms)
    softmin(x) = -1/beta * log(sum(exp(-beta * x)))
    We want to maximize softmin, so we minimize -softmin.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    # Pairwise terms: d_ij / 2
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    triu_idx = np.triu_indices(n, k=1)
    pair_dists = dists[triu_idx]
    pair_terms = pair_dists / 2.0
    
    # Boundary terms: 0.5 - |c_ik|
    boundary_terms = 0.5 - np.abs(C)
    boundary_terms_flat = boundary_terms.flatten()
    
    all_terms = np.concatenate([pair_terms, boundary_terms_flat])
    
    # Soft min calculation using log-sum-exp for numerical stability
    min_val = np.min(all_terms)
    shifted = -beta * (all_terms - min_val)
    
    # Clip to prevent underflow/overflow
    shifted = np.clip(shifted, -700, 700)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0 or not np.isfinite(sum_exp):
        weights = np.zeros_like(all_terms)
        min_indices = np.where(all_terms == min_val)[0]
        weights[min_indices] = 1.0 / len(min_indices)
    else:
        weights = exp_vals / sum_exp
    
    soft_min_val = min_val - (1.0 / beta) * np.log(sum_exp)
    
    grad_C = np.zeros_like(C)
    
    pair_weights = weights[:len(pair_terms)]
    i_idx = triu_idx[0]
    j_idx = triu_idx[1]
    
    d_vecs = C[i_idx] - C[j_idx]
    d_norms = np.linalg.norm(d_vecs, axis=1)
    d_norms_safe = np.where(d_norms < 1e-15, 1e-15, d_norms)
    
    # Gradient of soft_min_val w.r.t C
    # soft_min_val = min_val - (1/beta) log(sum_exp)
    # d(soft_min)/d(term_k) = -weights_k
    # d(term_k)/d(C) ...
    
    # We are minimizing -soft_min_val.
    # So objective_grad = - d(soft_min)/d(C)
    # d(soft_min)/d(C) = sum_k (-weights_k) * d(term_k)/d(C)
    # Thus, objective_grad = sum_k (weights_k) * d(term_k)/d(C)
    
    # d(term_k)/d(c_i) for pair term (i,j): (c_i - c_j) / (2 * d_ij)
    grad_pair_i = pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
    grad_pair_j = -pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
    
    np.add.at(grad_C, i_idx, grad_pair_i)
    np.add.at(grad_C, j_idx, grad_pair_j)
    
    # Boundary gradient
    # term = 0.5 - |c_ik|
    # d(term)/d(c_ik) = -sign(c_ik)
    # objective_grad += weight * (-sign(c_ik))
    bnd_weights = weights[len(pair_terms):].reshape(n, 3)
    signs = np.sign(C)
    signs[np.abs(C) < 1e-15] = 1.0
    grad_C += -bnd_weights * signs
    
    # Return objective value (-soft_min) and gradient of objective
    return -soft_min_val, -grad_C.flatten()

def _bcc_lattice_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    # BCC: 2 points per unit cell
    # 3x3x3 grid of cells -> 27 cells * 2 = 54 points.
    m = 3
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz + 0.5])
                
    points = np.array(points)
    idx = rng.permutation(len(points))
    points = points[idx[:n_spheres]]
    
    # Center and scale
    points = points - 1.5
    points *= 0.3 # Scale to fit well within [-0.5, 0.5]
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    return points

def _fcc_lattice_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    # FCC: 4 points per unit cell
    # 3x3x2 = 18 cells * 4 = 72 points.
    m_x, m_y, m_z = 3, 3, 2
    points = []
    for ix in range(m_x):
        for iy in range(m_y):
            for iz in range(m_z):
                points.append([ix, iy, iz])
                points.append([ix + 0.5, iy + 0.5, iz])
                points.append([ix + 0.5, iy, iz + 0.5])
                points.append([ix, iy + 0.5, iz + 0.5])
                
    points = np.array(points)
    idx = rng.permutation(len(points))
    points = points[idx[:n_spheres]]
    
    cx, cy, cz = (m_x-1)/2.0, (m_y-1)/2.0, (m_z-1)/2.0
    points = points - np.array([cx, cy, cz])
    points *= 0.3
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    return points

def _random_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.4, 0.4, size=(n_spheres, 3))

def _grid_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    # 5x5x4 grid = 100 points
    m_x, m_y, m_z = 5, 5, 4
    points = []
    for ix in range(m_x):
        for iy in range(m_y):
            for iz in range(m_z):
                points.append([ix, iy, iz])
    points = np.array(points)
    idx = rng.permutation(len(points))
    points = points[idx[:n_spheres]]
    
    cx, cy, cz = (m_x-1)/2.0, (m_y-1)/2.0, (m_z-1)/2.0
    points = points - np.array([cx, cy, cz])
    points *= 0.22
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    return points

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # 5s margin
    
    initializations = [
        _bcc_lattice_init(N, seed=42),
        _fcc_lattice_init(N, seed=42),
        _grid_init(N, seed=42),
        _random_init(N, seed=7),
        _bcc_lattice_init(N, seed=123),
        _fcc_lattice_init(N, seed=123),
        _grid_init(N, seed=123),
        _random_init(N, seed=456),
    ]
    
    best_C = None
    best_r = -np.inf
    
    # Evaluate initializations
    for init in initializations:
        r = _exact_radius(init)
        if r > best_r:
            best_r = r
            best_C = init.copy()
    
    # Optimization Schedule
    # Start with smooth landscape (low beta) and sharpen (high beta)
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for init in initializations:
        if time.time() > deadline - 15.0:
            break
            
        C_curr = init.copy()
        
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
                
            def soft_obj(x, b=beta):
                C = x.reshape(-1, 3)
                val, _ = _soft_min_radius_and_grad(C, b)
                return val
                
            def soft_grad(x, b=beta):
                C = x.reshape(-1, 3)
                _, grad = _soft_min_radius_and_grad(C, b)
                return grad

            try:
                res = minimize(
                    soft_obj,
                    C_curr.flatten(),
                    jac=soft_grad,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                C_new = res.x.reshape(-1, 3)
                
                r_new = _exact_radius(C_new)
                
                if r_new > best_r:
                    best_r = r_new
                    best_C = C_new.copy()
                
                C_curr = C_new
            except Exception:
                pass
    
    # Final high-precision polish on global best
    if time.time() < deadline - 5.0 and best_C is not None:
        # Use very high beta for final polish
        for beta_final in [10000.0, 20000.0]:
            if time.time() > deadline - 2.0:
                break
                
            def final_obj(x, b=beta_final):
                C = x.reshape(-1, 3)
                val, _ = _soft_min_radius_and_grad(C, b)
                return val
                
            def final_grad(x, b=beta_final):
                C = x.reshape(-1, 3)
                _, grad = _soft_min_radius_and_grad(C, b)
                return grad

            try:
                res = minimize(
                    final_obj,
                    best_C.flatten(),
                    jac=final_grad,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
                )
                C_new = res.x.reshape(-1, 3)
                r_new = _exact_radius(C_new)
                if r_new > best_r:
                    best_r = r_new
                    best_C = C_new.copy()
            except Exception:
                pass

    # Ensure validity: clip to cube
    if best_C is not None:
        best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
    else:
        best_C = _random_init(N)
        best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
        
    return best_C
