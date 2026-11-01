"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + L-BFGS relaxation."""
import numpy as np
import time
import sys

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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

def _get_icosahedron_data():
    """
    Returns icosahedron vertex unit vectors and edge list.
    12 vertices, 30 edges.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges: pairs of vertices at minimum distance
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
    
    return verts_unit, edges

def _edge_fraction_to_positions(r1, r2, r3, t_vec, verts_unit, edges):
    """
    Convert parameters to 55 atom positions.
    r1, r2, r3: shell radii
    t_vec: 30 edge fractions (one per edge), position = (1-t)*v_i + t*v_j scaled by r2
    Returns (55, 3) array.
    """
    n_edges = len(edges)
    
    # Center atom
    pos = np.zeros((55, 3))
    
    # Shell 1: 12 vertices at radius r1
    pos[1:13] = verts_unit * r1
    
    # Shell 2: 30 edge atoms at radius r2, parameterized by t along each edge
    for k, (i, j) in enumerate(edges):
        t = np.clip(t_vec[k], 0.0, 1.0)
        # Point on edge between verts_unit[i] and verts_unit[j]
        point = (1.0 - t) * verts_unit[i] + t * verts_unit[j]
        norm = np.linalg.norm(point)
        if norm > 1e-10:
            point = point / norm
        pos[13 + k] = point * r2
    
    # Shell 3: 12 vertices at radius r3 (same directions as shell 1)
    pos[43:55] = verts_unit * r3
    
    return pos

def _edge_fraction_energy_grad(params, verts_unit, edges):
    """
    Compute energy and gradient for the edge-fraction parameterization.
    params = [r1, r2, r3, t_1, ..., t_30]
    """
    r1 = max(params[0], 0.1)
    r2 = max(params[1], 0.1)
    r3 = max(params[2], 0.1)
    t_vec = np.clip(params[3:33], 0.0, 1.0)
    
    pos = _edge_fraction_to_positions(r1, r2, r3, t_vec, verts_unit, edges)
    e, grad_pos = _lj_energy_grad(pos)
    
    # Gradient w.r.t r1: sum of grad_pos[1:13] dot verts_unit
    grad_r1 = np.sum(grad_pos[1:13] * verts_unit)
    
    # Gradient w.r.t r2: for each edge atom, grad_pos[13+k] dot (direction of that atom)
    # direction_k = normalize((1-t_k)*v_i + t_k*v_j)
    grad_r2 = 0.0
    for k, (i, j) in enumerate(edges):
        t = t_vec[k]
        point = (1.0 - t) * verts_unit[i] + t * verts_unit[j]
        norm = np.linalg.norm(point)
        if norm > 1e-10:
            direction = point / norm
            grad_r2 += np.sum(grad_pos[13 + k] * direction)
    
    # Gradient w.r.t r3
    grad_r3 = np.sum(grad_pos[43:55] * verts_unit)
    
    # Gradient w.r.t t_k: d(pos[13+k])/dt_k = r2 * d(direction_k)/dt_k
    # direction_k = normalize((1-t)*v_i + t*v_j)
    # d(direction)/dt = [(v_j - v_i) - direction * dot(direction, v_j - v_i)] / norm
    grad_t = np.zeros(30)
    for k, (i, j) in enumerate(edges):
        t = t_vec[k]
        point = (1.0 - t) * verts_unit[i] + t * verts_unit[j]
        norm = np.linalg.norm(point)
        if norm > 1e-10:
            direction = point / norm
            edge_vec = verts_unit[j] - verts_unit[i]
            # d(direction)/dt = (edge_vec - direction * dot(direction, edge_vec)) / norm
            d_dir = (edge_vec - direction * np.dot(direction, edge_vec)) / norm
            grad_t[k] = r2 * np.sum(grad_pos[13 + k] * d_dir)
    
    grad = np.concatenate([[grad_r1, grad_r2, grad_r3], grad_t])
    
    return e, grad

def _relax_lbfgs(pos, maxiter=5000, gtol=1e-12):
    """Relax using L-BFGS-B with tight tolerance."""
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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def _relax_manifold(params0, verts_unit, edges, maxiter=2000, gtol=1e-12):
    """Relax in the 33-dimensional manifold (3 radii + 30 edge fractions)."""
    def obj_and_grad(params):
        e, g = _edge_fraction_energy_grad(params, verts_unit, edges)
        return e, g
    
    try:
        result = minimize(
            obj_and_grad, params0, jac=True, method='L-BFGS-B',
            bounds=[(0.1, 5.0), (0.1, 5.0), (0.1, 5.0)] + [(0.0, 1.0)] * 30,
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15}
        )
        return result.x, result.fun
    except Exception:
        return params0, _edge_fraction_energy_grad(params0, verts_unit, edges)[0]

def construct():
    """Construct 55-atom LJ cluster using edge-fraction manifold + full relaxation."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit)
    
    # 1. Build icosahedron data
    verts_unit, edges = _get_icosahedron_data()
    n_edges = len(edges)  # Should be 30
    
    # 2. Optimize in the 33D manifold (3 radii + 30 edge fractions)
    # Initial guess: r1=1.0, r2=1.5, r3=2.0, all t=0.5 (midpoints)
    params0 = np.array([1.0, 1.5, 2.0] + [0.5] * n_edges)
    
    if time.time() < deadline - 60.0:
        params_opt, e_manifold = _relax_manifold(params0, verts_unit, edges, maxiter=3000, gtol=1e-12)
    else:
        params_opt = params0
        e_manifold = _edge_fraction_energy_grad(params0, verts_unit, edges)[0]
    
    # Convert to Cartesian positions
    r1 = max(params_opt[0], 0.1)
    r2 = max(params_opt[1], 0.1)
    r3 = max(params_opt[2], 0.1)
    t_vec = np.clip(params_opt[3:33], 0.0, 1.0)
    pos_seed = _edge_fraction_to_positions(r1, r2, r3, t_vec, verts_unit, edges)
    
    best_pos = pos_seed.copy()
    best_energy = _lj_energy(pos_seed)
    
    # 3. Full 3D L-BFGS relaxation from manifold result
    if time.time() < deadline - 30.0:
        pos_relaxed, energy_relaxed = _relax_lbfgs(pos_seed, maxiter=5000, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # 4. Multi-start: perturb in manifold space and re-relax
    rng = np.random.default_rng(42)
    search_deadline = deadline - 15.0
    
    for trial in range(5):
        if time.time() >= search_deadline:
            break
        
        # Small perturbation in manifold space
        noise_r = rng.normal(0, 0.05, size=3)
        noise_t = rng.normal(0, 0.05, size=n_edges)
        params_pert = params_opt.copy()
        params_pert[0] = np.clip(params_pert[0] + noise_r[0], 0.5, 3.0)
        params_pert[1] = np.clip(params_pert[1] + noise_r[1], 0.5, 3.0)
        params_pert[2] = np.clip(params_pert[2] + noise_r[2], 0.5, 3.0)
        params_pert[3:33] = np.clip(params_pert[3:33] + noise_t, 0.0, 1.0)
        
        params_opt2, e2 = _relax_manifold(params_pert, verts_unit, edges, maxiter=1000, gtol=1e-10)
        
        r1 = max(params_opt2[0], 0.1)
        r2 = max(params_opt2[1], 0.1)
        r3 = max(params_opt2[2], 0.1)
        t_vec2 = np.clip(params_opt2[3:33], 0.0, 1.0)
        pos2 = _edge_fraction_to_positions(r1, r2, r3, t_vec2, verts_unit, edges)
        
        # Full 3D relaxation
        if time.time() < search_deadline:
            pos_relaxed, energy_relaxed = _relax_lbfgs(pos2, maxiter=2000, gtol=1e-12)
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed
    
    # 5. Final high-precision polish
    if time.time() < deadline - 5.0:
        pos_polished, energy_polished = _relax_lbfgs(best_pos, maxiter=10000, gtol=1e-14)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
    
    # Final min-distance check
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    if min_dist_found < 0.5:
        best_pos = best_pos * (0.5 / min_dist_found * 1.01)
    
    return best_pos
# EVOLVE-BLOCK-END
