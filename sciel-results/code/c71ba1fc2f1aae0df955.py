"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + Adaptive Basin Hopping."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    # Gradient factor: dE/dr2 = 4 * (-6*r^-14 + 3*r^-8) / (2*r) * 2? 
    # E = 4(r^-12 - r^-6) = 4((1/r2)^6 - (1/r2)^3)
    # dE/dr2 = 4(-6*(1/r2)^7 + 3*(1/r2)^4) = (1/r2^4) * 4 * (3 - 6/r2^3) = 4(3r2^3 - 6)/r2^7
    # Wait, standard LJ force formula: F = 24(2r^-14 - r^-8) * r_vec
    # dE/dr_vec_i = 24 * (2/r^14 - 1/r^8) * (pos_i - pos_j)
    # Let's stick to the stable vectorized form derived previously.
    # factor[i,j] corresponds to dE/d(r2[i,j])? No, we need dE/dpos.
    # dE/dpos_i = sum_j dE/dr2_ij * d(r2_ij)/dpos_i
    # d(r2_ij)/dpos_i = 2 * (pos_i - pos_j)
    # dE/dr2 = 4 * (-6*r^-14 + 3*r^-8) = 4 * r^-8 * (3 - 6*r^-6)
    # Let's use the form: 24 * (2/r12 - 1/r6) * (pos_i - pos_j)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # coeff[i,j] = 24 * (2/r12 - 1/r6)
        coeff = 24.0 * (2.0 / r12 - 1.0 / r6)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = sum_j coeff[i,j] * (pos_i - pos_j)
    # grad_i = sum_j coeff[i,j]*pos_i - sum_j coeff[i,j]*pos_j
    sum_coeff = np.sum(coeff, axis=1, keepdims=True)
    sum_coeff_pos = np.sum(coeff[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_coeff - sum_coeff_pos
    
    return energy, grad

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    second_inner = edge_mids * r2
    second_outer = verts_unit * r3
    
    all_pos = np.vstack([center, first_shell, second_inner, second_outer])
    return all_pos

def _relax_lbfgs(pos, deadline, maxiter=1000, gtol=1e-10):
    """Relax using L-BFGS-B."""
    start_time = time.time()
    best_energy, _ = _lj_energy_grad(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def _adaptive_basin_hopping(pos0, deadline, rng):
    """Adaptive Basin Hopping with varied perturbation strategies."""
    best_pos = pos0.copy()
    best_energy, _ = _lj_energy_grad(pos0)
    current_pos = pos0.copy()
    current_energy = best_energy
    
    sigma = 0.15
    no_improve_count = 0
    max_sigma = 0.5
    min_sigma = 0.05
    
    step = 0
    while time.time() < deadline - 20.0:
        step += 1
        
        # Adaptive sigma logic
        if no_improve_count > 5:
            sigma = min(sigma * 1.2, max_sigma)
            no_improve_count = 0
        elif no_improve_count > 0:
            pass # keep sigma
        else:
            sigma = max(sigma * 0.95, min_sigma)
            
        # Choose perturbation strategy
        r = rng.random()
        perturbed = current_pos.copy()
        
        if r < 0.4:
            # Global small noise
            perturbed += rng.normal(0, sigma, size=(N, 3))
        elif r < 0.7:
            # Kick 10 random atoms
            idx = rng.choice(N, size=10, replace=False)
            perturbed[idx] += rng.normal(0, sigma * 2.0, size=(10, 3))
        else:
            # Kick 1-2 atoms strongly (structural rearrangement attempt)
            k = rng.randint(1, 3)
            idx = rng.choice(N, size=k, replace=False)
            perturbed[idx] += rng.normal(0, 0.3, size=(k, 3))
            
        # Relax
        relaxed, energy = _relax_lbfgs(perturbed, deadline=deadline, maxiter=500, gtol=1e-8)
        
        # Acceptance
        if energy < current_energy:
            current_pos = relaxed.copy()
            current_energy = energy
            no_improve_count = 0
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        else:
            no_improve_count += 1
            # Metropolis acceptance for non-improving steps to escape local minima
            T = 0.1 + 0.05 * min(step / 10.0, 1.0)
            delta_e = energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                current_pos = relaxed.copy()
                current_energy = energy
                
    return best_pos, best_energy

def _final_polish(pos, deadline):
    """High precision polish."""
    best_energy, _ = _lj_energy_grad(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
        
    x0 = pos.flatten()
    
    # Stage 1
    if time.time() < deadline - 5.0:
        try:
            result = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                              options={'maxiter': 5000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16})
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass

    # Stage 2
    if time.time() < deadline - 1.0:
        try:
            result = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                              options={'maxiter': 10000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16})
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
        except Exception:
            pass
            
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # 1. Generate optimized Mackay Icosahedron seed
    # Quick radial optimization
    def radial_energy(params):
        r1, r2, r3 = params
        pos = _make_mackay_icosahedron(r1, r2, r3)
        e, _ = _lj_energy_grad(pos)
        return e
        
    x0_rad = np.array([1.05, 1.75, 1.95])
    try:
        res_rad = minimize(radial_energy, x0_rad, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if res_rad.fun < radial_energy(x0_rad):
            x0_rad = res_rad.x
    except Exception:
        pass
        
    seed_pos = _make_mackay_icosahedron(*x0_rad)
    
    # 2. Initial Relaxation
    best_pos, best_energy = _relax_lbfgs(seed_pos, deadline=deadline, maxiter=2000, gtol=1e-10)
    
    # 3. Adaptive Basin Hopping
    if time.time() < deadline - 30.0:
        bh_pos, bh_energy = _adaptive_basin_hopping(best_pos, deadline, rng)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    
    # 4. Final Polish
    if time.time() < deadline - 2.0:
        pol_pos, pol_energy = _final_polish(best_pos, deadline)
        if pol_energy < best_energy:
            best_energy = pol_energy
            best_pos = pol_pos
            
    # 5. Constraint Check
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
        
    return best_pos

# EVOLVE-BLOCK-END
