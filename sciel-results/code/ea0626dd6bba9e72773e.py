"""Lennard-Jones 55-atom global minimum: Mackay icosahedron with optimized radial parameters + multi-stage polish."""
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
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
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

def _relax_lbfgs(pos, deadline=None, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    from scipy.optimize import minimize
    
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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def _polish(pos, deadline=None):
    """Multi-stage high-precision polish: progressively tighten L-BFGS tolerance."""
    from scipy.optimize import minimize
    
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    x0 = pos.flatten()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-15},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-16},
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
            else:
                break  # No improvement, stop polishing
        except Exception:
            break
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Curated radius combinations for Mackay icosahedron
    radius_combos = [
        (1.05, 1.75, 1.95),
        (1.00, 1.70, 1.90),
        (1.10, 1.80, 2.00),
        (1.05, 1.80, 2.00),
        (1.00, 1.75, 1.95),
        (1.02, 1.72, 1.92),
        (1.08, 1.78, 1.98),
    ]
    
    # Generate candidates
    candidates = []
    for r1, r2, r3 in radius_combos:
        cand = _make_mackay_icosahedron(r1, r2, r3)
        candidates.append(cand)
    
    # Add a couple of small perturbations of the default Mackay
    np.random.seed(42)
    base = _make_mackay_icosahedron()
    for scale in [0.02, 0.05]:
        perturbed = base + np.random.randn(55, 3) * scale
        candidates.append(perturbed)
    
    # Sort candidates by initial energy
    cand_energies = [(c, _lj_energy(c)) for c in candidates]
    cand_energies.sort(key=lambda x: x[1])
    candidates = [c for c, _ in cand_energies]
    
    # Phase 1: Quick relaxation of top candidates
    best_energy = float('inf')
    best_pos = candidates[0].copy()
    
    top_n = min(6, len(candidates))
    for i in range(top_n):
        if time.time() > deadline - 60.0:
            break
        relaxed, energy = _relax_lbfgs(candidates[i], deadline=deadline, maxiter=800, gtol=1e-10)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    # Phase 2: Aggressive multi-stage polish on the best result
    if time.time() < deadline - 15.0:
        polished, polished_energy = _polish(best_pos, deadline=deadline)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pos = polished
    
    # Phase 3: Micro-perturbations around the best to escape shallow local minima
    if time.time() < deadline - 10.0:
        np.random.seed(999)
        for scale in [0.003, 0.008, 0.015]:
            if time.time() > deadline - 8.0:
                break
            perturbed = best_pos + np.random.randn(55, 3) * scale
            relaxed, energy = _relax_lbfgs(perturbed, deadline=deadline, maxiter=500, gtol=1e-12)
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
