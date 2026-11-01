"""Lennard-Jones 55-atom global minimum: Per-atom radial manifold + full 3D relaxation."""
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
    Returns 54 unit vectors: 12 inner vertices, 30 edge midpoints, 12 outer vertices.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    v_inner = verts / norms
    
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
BASE_DIRS = np.vstack([V_INNER, V_EDGE, V_OUTER])  # 54 unit vectors

def _rotation_matrix_from_angles(theta, phi, psi):
    """Compute rotation matrix from Euler angles (XYZ convention)."""
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    cs, ss = np.cos(psi), np.sin(psi)
    
    R = np.array([
        [cs*cp, cs*sp*st - ss*ct, cs*sp*ct + ss*st],
        [ss*cp, ss*sp*st + cs*ct, ss*sp*ct - cs*st],
        [-sp,   cp*st,            cp*ct]
    ])
    return R

def _construct_from_manifold(params):
    """
    params: [r1..r54, theta, phi, psi, tx, ty, tz]
    54 radial params + 3 rotation + 3 translation = 60 params
    Center atom is fixed at origin.
    """
    n_radial = 54
    r_vec = params[:n_radial]
    theta, phi, psi = params[n_radial:n_radial+3]
    t_vec = params[n_radial+3:n_radial+6]
    
    scaled = BASE_DIRS * r_vec[:, np.newaxis]
    R = _rotation_matrix_from_angles(theta, phi, psi)
    rotated = scaled @ R.T
    translated = rotated + t_vec[np.newaxis, :]
    
    center = np.array([[0.0, 0.0, 0.0]])
    pos = np.vstack([center, translated])
    return pos

def _manifold_objective(params):
    """Objective with soft constraint on minimum distance."""
    pos = _construct_from_manifold(params)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    
    if min_r2 < 0.25:  # (0.5)^2
        penalty = 1e6 * (0.25 - min_r2)
        return _lj_energy(pos) + penalty
    
    return _lj_energy(pos)

def _manifold_objective_grad(params):
    """Compute manifold objective and gradient via finite differences (60 params is manageable)."""
    e = _manifold_objective(params)
    n = len(params)
    g = np.zeros(n)
    eps = 1e-8
    for i in range(n):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        g[i] = (_manifold_objective(p_plus) - _manifold_objective(p_minus)) / (2 * eps)
    return e, g

def _optimize_manifold(initial_params, deadline, maxiter=2000):
    """Multi-stage L-BFGS-B optimization on the 60-dim manifold."""
    best_energy = _manifold_objective(initial_params)
    best_params = initial_params.copy()
    x0 = initial_params.copy()
    
    stages = [
        {'maxiter': maxiter, 'gtol': 1e-8, 'maxls': 100, 'ftol': 1e-10},
        {'maxiter': maxiter, 'gtol': 1e-10, 'maxls': 200, 'ftol': 1e-12},
        {'maxiter': maxiter, 'gtol': 1e-12, 'maxls': 300, 'ftol': 1e-14},
    ]
    
    for stage_opts in stages:
        if time.time() > deadline - 20.0:
            break
        try:
            result = minimize(
                _manifold_objective, x0, jac=_manifold_objective_grad,
                method='L-BFGS-B', options=stage_opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.copy()
                x0 = result.x
        except Exception:
            pass
    
    return best_params, best_energy

def _relax_lbfgs(pos, deadline, maxiter=3000, gtol=1e-12):
    """Full 3D L-BFGS-B relaxation with deadline guard."""
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

def _polish(pos, deadline):
    """Multi-stage high-precision polish with progressively tighter tolerances."""
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    x0 = pos.flatten()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if time.time() > deadline - 5.0:
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

def construct():
    """Construct 55-atom LJ cluster: per-atom radial manifold + full 3D relaxation + polish."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Initial manifold parameters: 54 radii + 3 rotation + 3 translation
    # Standard Mackay icosahedron radii: shell1 ~1.05, shell2 ~1.75, shell3 ~1.95
    r1 = 1.05
    r2 = 1.75
    r3 = 1.95
    
    initial_radii = np.concatenate([
        np.full(12, r1),   # inner vertices
        np.full(30, r2),   # edge midpoints
        np.full(12, r3),   # outer vertices
    ])
    
    initial_params = np.concatenate([
        initial_radii,
        np.array([0.0, 0.0, 0.0]),  # rotation
        np.array([0.0, 0.0, 0.0]),  # translation
    ])
    
    # Step 1: Optimize on the 60-dim manifold
    if time.time() < deadline - 60.0:
        params_opt, energy_opt = _optimize_manifold(initial_params, deadline - 50.0, maxiter=3000)
        pos_manifold = _construct_from_manifold(params_opt)
        
        # Check minimum distance
        diff = pos_manifold[:, np.newaxis, :] - pos_manifold[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist >= 0.5:
            best_pos = pos_manifold.copy()
            best_energy = _lj_energy(best_pos)
        else:
            # Fallback: scale to satisfy constraint
            scale = 0.5 / min_dist * 1.01
            best_pos = pos_manifold * scale
            best_energy = _lj_energy(best_pos)
    else:
        # Fallback: standard Mackay icosahedron
        pos_std = np.vstack([
            np.array([[0.0, 0.0, 0.0]]),
            V_INNER * r1,
            V_EDGE * r2,
            V_OUTER * r3,
        ])
        best_pos = pos_std.copy()
        best_energy = _lj_energy(best_pos)
    
    # Step 2: Full 3D relaxation
    if time.time() < deadline - 30.0:
        pos_relaxed, energy_relaxed = _relax_lbfgs(best_pos, deadline - 20.0, maxiter=5000, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # Step 3: Small perturbation multi-start (limited, time-budgeted)
    rng = np.random.default_rng(42)
    polish_deadline = deadline - 10.0
    
    for scale in [0.03, 0.08]:
        if time.time() >= polish_deadline - 15.0:
            break
        n_pert = 2
        for _ in range(n_pert):
            if time.time() >= polish_deadline - 8.0:
                break
            noise = rng.normal(0, scale, size=best_pos.shape)
            perturbed = best_pos + noise
            
            # Enforce min distance
            diff = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            if min_dist < 0.5:
                s = 0.5 / min_dist * 1.01
                perturbed = perturbed * s
            
            pos_r, energy_r = _relax_lbfgs(perturbed, polish_deadline - 5.0, maxiter=2000, gtol=1e-10)
            if energy_r < best_energy:
                best_energy = energy_r
                best_pos = pos_r
    
    # Step 4: Final high-precision polish
    if time.time() < deadline - 5.0:
        pos_polished, energy_polished = _polish(best_pos, deadline - 2.0)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
    
    # Final constraint check
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
    
    return best_pos
# EVOLVE-BLOCK-END
