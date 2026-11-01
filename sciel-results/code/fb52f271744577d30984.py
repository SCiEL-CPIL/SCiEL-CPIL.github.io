import numpy as np
from scipy.optimize import minimize
import time

N = 55

def _lj_energy_grad(pos):
    """
    Compute total LJ energy and gradient for a set of positions.
    Vectorized for speed.
    """
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = -48.0 / (r2 ** 7) + 24.0 / (r2 ** 4)
    
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = factor @ pos
    
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_mackay_icosahedron_unit_vectors():
    """
    Returns unit vectors for the 55 positions in a standard Mackay icosahedron.
    """
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
    
    mask = dists > 1e-6
    min_dist = np.min(dists[mask])
    
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
    shell1_dir = verts_unit
    shell2_dir = edge_mids
    shell3_dir = verts_unit
    
    return center, shell1_dir, shell2_dir, shell3_dir

def _make_mackay_icosahedron(r1, r2, r3):
    """
    Construct a two-shell Mackay icosahedron for 55 atoms with specific radii.
    """
    center, s1, s2, s3 = _get_mackay_icosahedron_unit_vectors()
    
    first_shell = s1 * r1
    second_shell = s2 * r2
    third_shell = s3 * r3
    
    all_pos = np.vstack([center, first_shell, second_shell, third_shell])
    return all_pos

def _radial_energy_and_grad(radii, dirs):
    """
    Compute energy and gradient with respect to radial coordinates.
    """
    n_noncenter = len(radii)
    pos = np.zeros((n_noncenter + 1, 3))
    pos[1:] = dirs * radii[:, np.newaxis]
    
    e, g_pos = _lj_energy_grad(pos)
    g_r = np.sum(g_pos[1:] * dirs, axis=1)
    
    return e, g_r

def _optimize_radial_manifold(dirs, r0, deadline, maxiter=500):
    """
    Optimize on the radial manifold: each atom moves along its fixed icosahedral direction.
    """
    def obj_and_grad(r):
        e, g = _radial_energy_and_grad(r, dirs)
        return e, g
    
    x0 = r0.copy()
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(0.5, 5.0)] * len(r0),
            options={'maxiter': maxiter, 'gtol': 1e-10, 'maxls': 100, 'ftol': 1e-12}
        )
        return result.x, result.fun
    except Exception:
        return x0, _radial_energy_and_grad(x0, dirs)[0]

def _relax_lbfgs(pos, deadline, maxiter=500, gtol=1e-8):
    """
    Relax using L-BFGS-B with analytic gradient and deadline guard.
    """
    best_energy = _lj_energy_grad(pos)[0]
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def construct():
    """
    Constructs the 55-atom LJ cluster by:
    1. Starting from the Mackay icosahedron
    2. Optimizing on the radial manifold (54D) to find optimal shell radii
    3. Full 165D relaxation
    4. Iterative radial+full refinement
    5. Multi-start with diverse perturbations
    6. Final high-precision polish
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    center, s1, s2, s3 = _get_mackay_icosahedron_unit_vectors()
    dirs = np.vstack([s1, s2, s3])
    
    # Initial radii guess
    r1_init = 1.12
    r2_init = 1.75
    r3_init = 1.95
    r0 = np.concatenate([np.full(12, r1_init), np.full(30, r2_init), np.full(12, r3_init)])
    
    best_pos = None
    best_energy = np.inf
    
    # Step 1: Radial manifold optimization
    if time.time() < deadline - 60.0:
        r_opt, e_radial = _optimize_radial_manifold(dirs, r0, deadline, maxiter=500)
        pos_radial = np.zeros((55, 3))
        pos_radial[1:] = dirs * r_opt[:, np.newaxis]
        best_energy = e_radial
        best_pos = pos_radial.copy()
        
        # Step 2: Full 3D relaxation
        if time.time() < deadline - 50.0:
            pos_relaxed, energy_relaxed = _relax_lbfgs(pos_radial, deadline, maxiter=500, gtol=1e-8)
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed.copy()
    
    # Step 3: Iterative refinement - alternate radial and full
    # Extract radii from current best position
    for iter_count in range(3):
        if time.time() > deadline - 40.0:
            break
        
        if best_pos is None:
            break
        
        # Compute current radii from best_pos
        radii_current = np.zeros(54)
        for i in range(54):
            d = dirs[i]
            radii_current[i] = np.dot(best_pos[i+1], d)
        
        # Radial optimization from current radii
        r_new, e_new = _optimize_radial_manifold(dirs, radii_current, deadline, maxiter=300)
        pos_new = np.zeros((55, 3))
        pos_new[1:] = dirs * r_new[:, np.newaxis]
        
        if e_new < best_energy:
            best_energy = e_new
            best_pos = pos_new.copy()
        
        # Full relaxation
        if time.time() < deadline - 30.0:
            pos_relaxed, energy_relaxed = _relax_lbfgs(pos_new, deadline, maxiter=300, gtol=1e-10)
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed.copy()
    
    # Step 4: Multi-start with diverse perturbations
    rng = np.random.default_rng(42)
    
    # Try different perturbation scales
    noise_scales = [0.01, 0.02, 0.05, 0.1]
    
    for scale in noise_scales:
        if time.time() > deadline - 15.0:
            break
        
        if best_pos is not None:
            pos_seed = best_pos.copy() + rng.normal(0, scale, size=best_pos.shape)
        else:
            pos_seed = _make_mackay_icosahedron(1.12, 1.75, 1.95)
        
        pos_relaxed, energy_relaxed = _relax_lbfgs(pos_seed, deadline, maxiter=400, gtol=1e-10)
        
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed.copy()
    
    # Step 5: Final high-precision polish
    if time.time() < deadline - 5.0 and best_pos is not None:
        pos_final, energy_final = _relax_lbfgs(
            best_pos, deadline, maxiter=2000, gtol=1e-14
        )
        if energy_final < best_energy:
            best_energy = energy_final
            best_pos = pos_final
        
        # Ultra-fine polish
        if time.time() < deadline - 2.0:
            pos_ultra, energy_ultra = _relax_lbfgs(
                best_pos, deadline, maxiter=500, gtol=1e-16
            )
            if energy_ultra < best_energy:
                best_energy = energy_ultra
                best_pos = pos_ultra
    
    # Final Safety Check: Minimum Distance
    if best_pos is not None:
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 0.5 - 1e-9:
            scale_factor = 0.5 / min_dist * 1.001
            best_pos *= scale_factor
    
    if best_pos is None:
        best_pos = _make_mackay_icosahedron(1.12, 1.75, 1.95)
    
    return best_pos
