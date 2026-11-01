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

def _get_mackay_icosahedron_dirs():
    """
    Returns unit vectors for the 55 positions in a Mackay icosahedron.
    1 center (0,0,0)
    12 first shell (icosahedron vertices)
    30 second shell (icosahedron edge midpoints)
    12 third shell (icosahedron vertices, same direction as first shell)
    Returns dirs for the 54 non-center atoms.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges of the icosahedron
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
    
    # Order: Shell1 (12), Shell2 (30), Shell3 (12)
    # Shell3 has same directions as Shell1
    dirs = np.vstack([verts_unit, edge_mids, verts_unit])
    return dirs

def _build_ideal_mackay(radii):
    """
    Build positions for ideal Mackay icosahedron given shell radii [r1, r2, r3].
    Returns (55, 3) array.
    """
    dirs = _get_mackay_icosahedron_dirs()
    center = np.array([[0.0, 0.0, 0.0]])
    
    # dirs is (54, 3)
    # First 12 are shell 1, next 30 are shell 2, last 12 are shell 3
    pos_s1 = dirs[:12] * radii[0]
    pos_s2 = dirs[12:42] * radii[1]
    pos_s3 = dirs[42:54] * radii[2]
    
    pos = np.vstack([center, pos_s1, pos_s2, pos_s3])
    return pos

def _optimize_shell_radii():
    """Optimize the 3 shell radii using Nelder-Mead with better convergence."""
    def radii_energy(radii):
        pos = _build_ideal_mackay(radii)
        return _lj_energy(pos)
    
    # Multiple initial guesses to find the best local minimum
    initial_guesses = [
        [1.05, 1.75, 1.95],
        [1.00, 1.80, 2.00],
        [1.10, 1.70, 1.90],
        [0.95, 1.85, 2.05],
    ]
    
    best_radii = None
    best_energy = np.inf
    
    for r0 in initial_guesses:
        try:
            result = minimize(radii_energy, r0, method='Nelder-Mead',
                              options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 5000,
                                       'adaptive': True})
            if result.fun < best_energy:
                best_energy = result.fun
                best_radii = result.x
        except Exception:
            continue
    
    if best_radii is None:
        return np.array([1.05, 1.75, 1.95])
    return best_radii

def _relax_lbfgs(pos, deadline, maxiter=2000, gtol=1e-12):
    """Relax using L-BFGS-B with deadline guard."""
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
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-14, 'maxls': 200}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster using ideal Mackay seed + L-BFGS relaxation."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Optimize shell radii for ideal Mackay icosahedron
    if time.time() < deadline - 30.0:
        radii_opt = _optimize_shell_radii()
    else:
        radii_opt = np.array([1.05, 1.75, 1.95])
    
    # 2. Build ideal structure and relax with tight tolerance
    ideal_pos = _build_ideal_mackay(radii_opt)
    
    best_pos = ideal_pos.copy()
    best_energy = _lj_energy(ideal_pos)
    
    # Single tight-tolerance L-BFGS relaxation
    if time.time() < deadline - 10.0:
        pos_relaxed, energy_relaxed = _relax_lbfgs(ideal_pos, deadline, maxiter=5000, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # 3. Final ultra-tight polish pass
    if time.time() < deadline - 2.0:
        pos_polished, energy_polished = _relax_lbfgs(best_pos, deadline, maxiter=10000, gtol=1e-14)
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
