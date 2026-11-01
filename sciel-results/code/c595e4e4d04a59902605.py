"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + L-BFGS relaxation."""
import numpy as np
import time

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 24.0 * (r6 - 2.0) / (r2 ** 7)
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms.
    1 center + 12 first shell + 30 edge-midpoints + 12 vertex-like = 55
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges of the icosahedron (12 vertices, 30 edges)
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
    
    # Edge midpoints (30 atoms)
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

def _optimize_radial_params(max_iter=500, deadline=None):
    """Optimize the 3 radial parameters of the Mackay icosahedron using Nelder-Mead."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    
    def radial_energy(params):
        r1, r2, r3 = params
        pos = _make_mackay_icosahedron(r1, r2, r3)
        return _lj_energy(pos)
    
    # Start from reasonable defaults
    x0 = np.array([1.05, 1.75, 1.95])
    
    try:
        result = minimize(
            radial_energy, x0, method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True,
                     'initial_simplex': np.array([[1.05, 1.75, 1.95],
                                                  [1.06, 1.76, 1.96],
                                                  [1.04, 1.74, 1.94],
                                                  [1.05, 1.75, 1.96]])}
        )
        if deadline is not None and time.time() > deadline:
            return x0
        return result.x
    except Exception:
        return x0

def _relax_lbfgs(pos, deadline=None, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100,
                     'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def _polish(pos, deadline=None):
    """High-precision polish pass with tighter tolerance and more iterations."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    # Multi-stage polish: progressively tighter
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
        {'maxiter': 20000, 'gtol': 1e-18, 'maxls': 2000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos, best_energy

def _basin_hopping(pos0, deadline, n_steps=15, sigma=0.15):
    """Basin-hopping with rigid-body displacements and L-BFGS relaxation.
    Each step: add small random displacement to all atoms, relax, accept if lower energy."""
    from scipy.optimize import minimize
    
    best_pos = pos0.copy()
    best_energy = _lj_energy(pos0)
    current_pos = pos0.copy()
    current_energy = best_energy
    
    np.random.seed(12345)
    
    for step in range(n_steps):
        if time.time() > deadline - 30.0:
            break
        
        # Rigid-body displacement: small random vector added to all atoms
        disp = np.random.randn(3) * sigma
        perturbed = current_pos + disp
        
        # Also try a few individual atom perturbations for more diversity
        if step % 3 == 0:
            # Perturb a random subset of atoms
            idx = np.random.choice(55, size=10, replace=False)
            perturbed2 = current_pos.copy()
            perturbed2[idx] += np.random.randn(10, 3) * sigma * 0.5
            perturbed = perturbed2
        
        # Relax the perturbed structure
        relaxed, energy = _relax_lbfgs(perturbed, deadline=deadline, maxiter=300, gtol=1e-10)
        
        # Metropolis acceptance: always accept if lower, accept higher with probability exp(-dE/T)
        if energy < current_energy:
            current_pos = relaxed.copy()
            current_energy = energy
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        else:
            # Simple acceptance: accept if not too much worse (helps escape local minima)
            T = 0.5  # temperature parameter
            if np.random.rand() < np.exp(-(energy - current_energy) / T):
                current_pos = relaxed.copy()
                current_energy = energy
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    # Step 1: Optimize radial parameters for the Mackay icosahedron
    optimal_radii = _optimize_radial_params(max_iter=500, deadline=start_time + 12.0)
    r1_opt, r2_opt, r3_opt = optimal_radii
    
    # Step 2: Generate diverse candidates
    candidates = []
    
    # Primary candidate: optimized Mackay
    candidates.append(_make_mackay_icosahedron(r1_opt, r2_opt, r3_opt))
    
    # Additional radius combinations around the optimum
    radius_combos = [
        (r1_opt, r2_opt, r3_opt),
        (r1_opt * 0.98, r2_opt * 0.98, r3_opt * 0.98),
        (r1_opt * 1.02, r2_opt * 1.02, r3_opt * 1.02),
        (r1_opt * 0.95, r2_opt * 1.05, r3_opt * 1.00),
        (r1_opt * 1.05, r2_opt * 0.95, r3_opt * 1.00),
        (r1_opt * 1.00, r2_opt * 1.05, r3_opt * 0.95),
        (r1_opt * 1.00, r2_opt * 0.95, r3_opt * 1.05),
    ]
    
    for r1, r2, r3 in radius_combos:
        cand = _make_mackay_icosahedron(r1, r2, r3)
        candidates.append(cand)
    
    # Random perturbations of the optimized Mackay
    np.random.seed(42)
    base = _make_mackay_icosahedron(r1_opt, r2_opt, r3_opt)
    for scale in [0.02, 0.05, 0.10, 0.15, 0.20]:
        perturbed = base + np.random.randn(55, 3) * scale
        candidates.append(perturbed)
    
    # Sort candidates by initial energy
    cand_energies = [(c, _lj_energy(c)) for c in candidates]
    cand_energies.sort(key=lambda x: x[1])
    candidates = [c for c, _ in cand_energies]
    
    # Step 3: Relax the top candidates
    best_energy = float('inf')
    best_pos = candidates[0].copy()
    
    # First pass: relax top 5 candidates (fewer to save time for basin hopping)
    top_n = min(5, len(candidates))
    for i in range(top_n):
        if time.time() > deadline - 80.0:
            break
        remaining = deadline - time.time()
        max_iter = max(300, int(remaining * 3))
        relaxed, energy = _relax_lbfgs(candidates[i], deadline=deadline, maxiter=max_iter, gtol=1e-10)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    # Step 4: Basin-hopping starting from the best relaxed structure
    if time.time() < deadline - 40.0:
        bh_pos, bh_energy = _basin_hopping(
            best_pos, deadline=deadline, n_steps=20, sigma=0.12
        )
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos.copy()
    
    # Step 5: Aggressive polish on the best result
    if time.time() < deadline - 20.0:
        polished, polished_energy = _polish(best_pos, deadline=deadline)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pos = polished
    
    # Step 6: Micro-perturbations around the best to escape shallow local min
    if time.time() < deadline - 10.0:
        np.random.seed(999)
        for scale in [0.003, 0.005, 0.01, 0.02]:
            if time.time() > deadline - 8.0:
                break
            perturbed = best_pos + np.random.randn(55, 3) * scale
            relaxed, energy = _relax_lbfgs(perturbed, deadline=deadline, maxiter=800, gtol=1e-12)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        # Final ultra-tight polish
        if time.time() < deadline - 5.0:
            polished, polished_energy = _polish(best_pos, deadline=deadline)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_pos = polished
    
    # Final check: ensure minimum distance >= 0.5
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
    
    return best_pos
# EVOLVE-BLOCK-END
