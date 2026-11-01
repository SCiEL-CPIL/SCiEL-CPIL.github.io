"""Lennard-Jones 55-atom global minimum: Parameterized Mackay icosahedron optimization."""
import numpy as np
import time
import sys
from scipy.optimize import minimize, minimize_scalar

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
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    v_inner = verts / norms
    
    # Find edges
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
    
    # Outer vertices have same directions as inner vertices in the standard Mackay model
    v_outer = v_inner.copy()
    
    return v_inner, v_edge, v_outer

V_INNER, V_EDGE, V_OUTER = _get_icosahedron_geometry()

def _construct_from_params(params):
    """
    params: [r1, r2, r3]
    r1: radius for 12 inner vertices
    r2: radius for 30 edge midpoints
    r3: radius for 12 outer vertices
    """
    r1, r2, r3 = params
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = V_INNER * r1
    shell2a = V_EDGE * r2
    shell2b = V_OUTER * r3
    
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

def _objective_with_relax(params):
    """Optimize radii, then relax in 3D for a more accurate energy estimate."""
    pos = _construct_from_params(params)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 0.5:
        return 1e10
    
    # Quick relaxation to get a better energy estimate
    pos_relaxed, e_relaxed = _relax_lbfgs(pos, maxiter=100, gtol=1e-6)
    return e_relaxed

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
    """High-precision polish pass with progressively tighter tolerances."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    # More aggressive polishing with more iterations at tight tolerances
    stages = [
        {'maxiter': 3000, 'gtol': 1e-10, 'maxls': 300, 'ftol': 1e-14},
        {'maxiter': 5000, 'gtol': 1e-12, 'maxls': 500, 'ftol': 1e-15},
        {'maxiter': 10000, 'gtol': 1e-14, 'maxls': 1000, 'ftol': 1e-16},
        {'maxiter': 20000, 'gtol': 1e-15, 'maxls': 2000, 'ftol': 1e-17},
        {'maxiter': 30000, 'gtol': 1e-16, 'maxls': 3000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 10.0:
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
    """
    Construct 55-atom LJ cluster using literature-optimal Mackay icosahedron
    radii, multi-start parameter optimization, noise perturbation, and 
    high-precision polishing.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    # Literature-optimal shell radii for LJ55 Mackay icosahedron
    literature_radii = np.array([1.135, 1.162, 1.283])
    
    # Multi-start optimization on the 3D parameter space
    starts = [
        [1.135, 1.162, 1.283],
        [1.130, 1.155, 1.275],
        [1.140, 1.170, 1.290],
        [1.125, 1.150, 1.270],
        [1.145, 1.175, 1.295],
        [1.135, 1.150, 1.283],
        [1.135, 1.175, 1.283],
        [1.135, 1.162, 1.270],
        [1.135, 1.162, 1.295],
        [1.120, 1.145, 1.260],
        [1.150, 1.180, 1.300],
    ]
    
    best_params = None
    best_param_energy = float('inf')
    
    # First pass: quick Nelder-Mead on raw energy
    for start in starts:
        if time.time() > deadline - 40.0:
            break
        try:
            res = minimize(
                _objective, 
                start, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_param_energy:
                best_param_energy = res.fun
                best_params = res.x
        except Exception:
            continue
    
    # Second pass: refine top candidates with relaxed energy objective
    if best_params is not None and time.time() < deadline - 25.0:
        # Generate nearby starts around the best found
        rng_refine = np.random.default_rng(123)
        refine_starts = [best_params.copy()]
        for _ in range(5):
            perturb = rng_refine.normal(0, 0.01, 3)
            refine_starts.append(best_params + perturb)
        
        for start in refine_starts:
            if time.time() > deadline - 15.0:
                break
            try:
                res = minimize(
                    _objective_with_relax, 
                    start, 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < best_param_energy:
                    best_param_energy = res.fun
                    best_params = res.x
            except Exception:
                continue
            
    if best_params is None:
        best_params = literature_radii.copy()
        best_param_energy = _objective(best_params)
        
    pos_0 = _construct_from_params(best_params)
    
    # Multi-start 3D relaxation with systematic noise perturbations
    best_pos_3d = pos_0.copy()
    best_e_3d = _lj_energy(pos_0)
    
    # More systematic perturbation: combine small and medium noise with multiple seeds
    noise_configs = [
        (0.0, 0),      # no noise
        (0.005, 1),    # very small
        (0.01, 2),     # small
        (0.02, 3),     # small-medium
        (0.05, 4),     # medium
        (0.10, 5),     # large
        (0.02, 100),   # small with different seed
        (0.05, 200),   # medium with different seed
    ]
    
    for noise_scale, seed in noise_configs:
        if time.time() > deadline - 25.0:
            break
        
        rng = np.random.default_rng(seed)
        
        if noise_scale == 0.0:
            pos_try = pos_0.copy()
        else:
            pos_try = pos_0 + rng.normal(0, noise_scale, pos_0.shape)
        
        pos_relaxed, e_relaxed = _relax_lbfgs(pos_try, deadline=deadline, maxiter=400, gtol=1e-10)
        
        if e_relaxed < best_e_3d:
            best_e_3d = e_relaxed
            best_pos_3d = pos_relaxed
    
    # High-precision polish on the best structure
    pos_final, e_final = _polish(best_pos_3d, deadline=deadline)
    
    # Verify constraints
    diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        pos_final = pos_final * scale
        pos_final, e_final = _relax_lbfgs(pos_final, deadline=deadline, maxiter=500, gtol=1e-12)

    return pos_final
# EVOLVE-BLOCK-END
