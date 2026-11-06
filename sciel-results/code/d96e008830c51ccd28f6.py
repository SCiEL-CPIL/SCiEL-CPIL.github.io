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

def _hard_min_radius_and_grad(C):
    """
    Compute the exact negative radius (min of pairwise and boundary distances)
    and a subgradient.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, np.inf)
    
    triu_idx = np.triu_indices(n, k=1)
    pair_dists = dists[triu_idx]
    pair_terms = pair_dists / 2.0
    
    boundary_terms = 0.5 - np.abs(C)
    boundary_terms_flat = boundary_terms.flatten()
    
    all_terms = np.concatenate([pair_terms, boundary_terms_flat])
    min_val = np.min(all_terms)
    
    grad_C = np.zeros_like(C)
    
    # Find all terms achieving the minimum (within tolerance)
    tol = 1e-10
    min_mask = all_terms <= min_val + tol
    min_count = np.sum(min_mask)
    
    if min_count == 0:
        return -min_val, grad_C.flatten()
    
    # Pair contributions
    pair_min_mask = min_mask[:len(pair_terms)]
    i_idx = triu_idx[0][pair_min_mask]
    j_idx = triu_idx[1][pair_min_mask]
    
    if len(i_idx) > 0:
        d_vecs = C[i_idx] - C[j_idx]
        d_norms = np.linalg.norm(d_vecs, axis=1)
        d_norms_safe = np.where(d_norms < 1e-15, 1e-15, d_norms)
        
        w = 1.0 / min_count
        
        grad_pair_i = w * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
        grad_pair_j = -w * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
        
        np.add.at(grad_C, i_idx, grad_pair_i)
        np.add.at(grad_C, j_idx, grad_pair_j)
    
    # Boundary contributions
    bnd_min_mask = min_mask[len(pair_terms):]
    bnd_min_idx = np.where(bnd_min_mask)[0]
    
    if len(bnd_min_idx) > 0:
        w = 1.0 / min_count
        n_idx = bnd_min_idx // 3
        k_idx = bnd_min_idx % 3
        
        signs = np.sign(C[n_idx, k_idx])
        signs[np.abs(C[n_idx, k_idx]) < 1e-15] = 1.0
        
        grad_C[n_idx, k_idx] -= w * signs
    
    return -min_val, -grad_C.flatten()

def _repulsive_energy_and_grad(C, alpha=10.0):
    """
    Compute a smooth repulsive energy to spread points.
    E = sum_{i<j} exp(-alpha * d_ij)
    This is smooth and encourages large minimum distances.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1) + 1e-12
    np.fill_diagonal(dists_sq, np.inf)
    
    triu_idx = np.triu_indices(n, k=1)
    pair_dists_sq = dists_sq[triu_idx]
    
    d = np.sqrt(pair_dists_sq)
    exp_term = np.exp(-alpha * d)
    energy = np.sum(exp_term)
    
    # Gradient: dE/dx_i = sum_j (-0.5 * alpha / d_ij) * exp(-alpha * d_ij) * (x_i - x_j)
    i_idx = triu_idx[0]
    j_idx = triu_idx[1]
    d_vecs = C[i_idx] - C[j_idx]
    d_safe = np.where(d < 1e-15, 1e-15, d)
    
    coeff = -0.5 * alpha * exp_term / d_safe
    
    grad_C = np.zeros_like(C)
    np.add.at(grad_C, i_idx, coeff[:, np.newaxis] * d_vecs)
    np.add.at(grad_C, j_idx, -coeff[:, np.newaxis] * d_vecs)
    
    return energy, grad_C.flatten()

def _bcc_lattice_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
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
    
    points = points - 1.5
    points *= 0.35
    points += rng.uniform(-0.05, 0.05, size=points.shape)
    return points

def _fcc_lattice_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
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
    points *= 0.35
    points += rng.uniform(-0.05, 0.05, size=points.shape)
    return points

def _random_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n_spheres, 3))

def _grid_init(n_spheres=50, seed=42):
    rng = np.random.RandomState(seed)
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
    points *= 0.30
    points += rng.uniform(-0.05, 0.05, size=points.shape)
    return points

def _simplex_init(n_spheres=50, seed=42):
    """Create a simple 3D lattice with jitter."""
    rng = np.random.RandomState(seed)
    m = 4
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                points.append([ix, iy, iz])
    points = np.array(points)
    idx = rng.permutation(len(points))
    points = points[idx[:n_spheres]]
    
    cx, cy, cz = (m-1)/2.0, (m-1)/2.0, (m-1)/2.0
    points = points - np.array([cx, cy, cz])
    points *= 0.35
    points += rng.uniform(-0.05, 0.05, size=points.shape)
    return points

def construct():
    t_start = time.time()
    deadline = t_start + 145.0
    
    initializations = [
        _bcc_lattice_init(N, seed=42),
        _fcc_lattice_init(N, seed=42),
        _grid_init(N, seed=42),
        _random_init(N, seed=7),
        _bcc_lattice_init(N, seed=123),
        _fcc_lattice_init(N, seed=123),
        _grid_init(N, seed=123),
        _random_init(N, seed=456),
        _simplex_init(N, seed=99),
        _simplex_init(N, seed=200),
    ]
    
    best_C = None
    best_r = -np.inf
    
    for init in initializations:
        r = _exact_radius(init)
        if r > best_r:
            best_r = r
            best_C = init.copy()
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Phase 1: Repulsive energy to spread points (quick, limited)
    for init in initializations[:6]:
        if time.time() > deadline - 40.0:
            break
        
        C_curr = init.copy()
        
        def rep_obj(x, a=8.0):
            C = x.reshape(-1, 3)
            val, _ = _repulsive_energy_and_grad(C, a)
            return val
            
        def rep_grad(x, a=8.0):
            C = x.reshape(-1, 3)
            _, grad = _repulsive_energy_and_grad(C, a)
            return grad

        try:
            res = minimize(
                rep_obj,
                C_curr.flatten(),
                jac=rep_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-9, 'maxls': 30}
            )
            C_new = res.x.reshape(-1, 3)
            r_new = _exact_radius(C_new)
            
            if r_new > best_r:
                best_r = r_new
                best_C = C_new.copy()
        except Exception:
            pass
    
    # Phase 2: Soft-min radius optimization with progressive beta
    betas = [5.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    
    for init in initializations:
        if time.time() > deadline - 25.0:
            break
            
        C_curr = init.copy()
        
        for beta in betas:
            if time.time() > deadline - 15.0:
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
                max_iter = 500 if beta < 100 else 800
                res = minimize(
                    soft_obj,
                    C_curr.flatten(),
                    jac=soft_grad,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 40}
                )
                C_new = res.x.reshape(-1, 3)
                
                r_new = _exact_radius(C_new)
                
                if r_new > best_r:
                    best_r = r_new
                    best_C = C_new.copy()
                
                C_curr = C_new
            except Exception:
                pass
    
    # Phase 3: Hard min subgradient polishing with better step control
    if time.time() < deadline - 8.0 and best_C is not None:
        C_curr = best_C.copy()
        r_curr = _exact_radius(C_curr)
        step = 0.02
        stall_count = 0
        
        for _ in range(3000):
            if time.time() > deadline - 3.0:
                break
                
            val, grad = _hard_min_radius_and_grad(C_curr)
            
            g_norm = np.linalg.norm(grad)
            if g_norm < 1e-14:
                break
                
            # Adaptive step: start moderate, reduce on failure
            C_trial = C_curr - (step / g_norm) * grad.reshape(-1, 3)
            C_trial = np.clip(C_trial, -0.5 + 1e-14, 0.5 - 1e-14)
            
            r_trial = _exact_radius(C_trial)
            
            if r_trial > r_curr + 1e-13:
                C_curr = C_trial
                r_curr = r_trial
                step = min(step * 1.1, 0.05)  # grow step slightly on success
                stall_count = 0
                if r_trial > best_r:
                    best_r = r_trial
                    best_C = C_curr.copy()
            else:
                step *= 0.5
                stall_count += 1
                if step < 1e-15 or stall_count > 100:
                    break
    
    # Phase 4: Final high-precision polish with very high beta
    if time.time() < deadline - 4.0 and best_C is not None:
        for beta_final in [5000.0, 10000.0, 20000.0, 50000.0]:
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
                    options={'maxiter': 2500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
                )
                C_new = res.x.reshape(-1, 3)
                r_new = _exact_radius(C_new)
                if r_new > best_r:
                    best_r = r_new
                    best_C = C_new.copy()
            except Exception:
                pass

    if best_C is not None:
        best_C = np.clip(best_C, -0.5 + 1e-14, 0.5 - 1e-14)
    else:
        best_C = _random_init(N)
        best_C = np.clip(best_C, -0.5 + 1e-14, 0.5 - 1e-14)
        
    return best_C
