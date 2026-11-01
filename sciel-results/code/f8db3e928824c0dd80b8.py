"""Lennard-Jones 55-atom global minimum: Parameterized Mackay icosahedron optimization."""
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

def _get_icosahedron_geometry():
    """
    Returns:
        v_inner: 12 unit vectors for the first shell (vertices).
        v_edge: 30 unit vectors for the second shell (edge midpoints).
        v_outer: 12 unit vectors for the second shell (vertices, same directions as inner).
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Exact normalization: |v| = sqrt(1 + phi^2) = sqrt(1 + (3+sqrt(5))/2) = sqrt((5+sqrt(5))/2)
    norm_val = np.sqrt(1.0 + phi * phi)
    v_inner = verts / norm_val
    
    # Find edges: closest pair distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(v_inner[i] - v_inner[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    v_edge = []
    for i, j in edges:
        mid = (v_inner[i] + v_inner[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            v_edge.append(mid / norm)
    v_edge = np.array(v_edge)
    
    v_outer = v_inner.copy()
    
    return v_inner, v_edge, v_outer

V_INNER, V_EDGE, V_OUTER = _get_icosahedron_geometry()

def _construct_from_params(params):
    """
    params: 54 radial values [r1_1..r1_12, r2_1..r2_30, r3_1..r3_12]
    r1: 12 radii for inner vertices
    r2: 30 radii for edge midpoints  
    r3: 12 radii for outer vertices
    """
    r1 = params[0:12]
    r2 = params[12:42]
    r3 = params[42:54]
    
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = V_INNER * r1[:, np.newaxis]
    shell2a = V_EDGE * r2[:, np.newaxis]
    shell2b = V_OUTER * r3[:, np.newaxis]
    
    pos = np.vstack([center, shell1, shell2a, shell2b])
    return pos

def _objective(params):
    pos = _construct_from_params(params)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 0.5:
        return 1e10
    return _lj_energy(pos)

def _relax_lbfgs(pos, deadline=None, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    if deadline is not None and time.time() > deadline - 5.0:
        return best_pos, best_energy
    
    return best_pos, best_energy

def _polish(pos, deadline=None):
    """High-precision multi-stage polish pass."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-15, 'maxls': 1000, 'ftol': 1e-17},
        {'maxiter': 20000, 'gtol': 1e-16, 'maxls': 2000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 3.0:
            break
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy - 1e-15:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos, best_energy

def _optimize_radial_params(start_params, deadline):
    """Optimize 54 radial parameters using Nelder-Mead with time guard."""
    best_params = start_params.copy()
    best_energy = _objective(best_params)
    
    try:
        res = minimize(
            _objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x
    except Exception:
        pass
    
    if deadline is not None and time.time() > deadline - 5.0:
        return best_params, best_energy
    
    return best_params, best_energy

def construct():
    """
    Construct 55-atom LJ cluster by optimizing the radial parameters
    of the Mackay icosahedron model, then performing full 3D relaxation.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    # Initial guess for 54 radial parameters
    # Typical optimal radii for LJ55 Mackay icosahedron:
    # r1 ~ 1.04 (12 inner), r2 ~ 1.76 (30 edge), r3 ~ 1.98 (12 outer)
    
    r1_init = np.full(12, 1.04)
    r2_init = np.full(30, 1.76)
    r3_init = np.full(12, 1.98)
    
    init_params = np.concatenate([r1_init, r2_init, r3_init])
    
    # Step 1: Optimize radial parameters
    best_params, best_param_energy = _optimize_radial_params(init_params, deadline)
    
    # Try a few alternative starts if time permits
    if time.time() < deadline - 30.0:
        alt_starts = [
            np.concatenate([np.full(12, 1.05), np.full(30, 1.75), np.full(12, 1.95)]),
            np.concatenate([np.full(12, 1.00), np.full(30, 1.70), np.full(12, 1.90)]),
            np.concatenate([np.full(12, 1.10), np.full(30, 1.80), np.full(12, 2.00)]),
        ]
        for alt in alt_starts:
            if time.time() > deadline - 10.0:
                break
            alt_params, alt_energy = _optimize_radial_params(alt, deadline)
            if alt_energy < best_param_energy:
                best_param_energy = alt_energy
                best_params = alt_params
    
    # Construct the initial structure from optimized parameters
    pos_0 = _construct_from_params(best_params)
    
    # Step 2: Full 3D L-BFGS relaxation
    pos_1, e_1 = _relax_lbfgs(pos_0, deadline=deadline, maxiter=1000, gtol=1e-10)
    
    # Step 3: High precision multi-stage polish
    pos_final, e_final = _polish(pos_1, deadline=deadline)
    
    # Verify constraints
    diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        pos_final = pos_final * scale
        if time.time() < deadline - 5.0:
            pos_final, e_final = _relax_lbfgs(pos_final, deadline=deadline, maxiter=500, gtol=1e-12)
    
    return pos_final
# EVOLVE-BLOCK-END
