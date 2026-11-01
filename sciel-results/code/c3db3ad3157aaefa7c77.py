"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + L-BFGS relaxation."""
import numpy as np
import time
import sys

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

def _get_mackay_icosahedron_coords():
    """
    Returns unit vectors for the 55 positions in a Mackay icosahedron.
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
    shell1_dir = verts_unit
    shell2_dir = edge_mids
    shell3_dir = verts_unit
    
    return center, shell1_dir, shell2_dir, shell3_dir

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms."""
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    
    first_shell = s1 * r1
    second_inner = s2 * r2
    second_outer = s3 * r3
    
    all_pos = np.vstack([center, first_shell, second_inner, second_outer])
    return all_pos

def _radial_energy_grad(radii, directions):
    """
    Compute energy and gradient with respect to radial coordinates.
    """
    pos = directions * radii[:, np.newaxis]
    e, grad_pos = _lj_energy_grad(pos)
    grad_r = np.sum(grad_pos * directions, axis=1)
    return e, grad_r

def _optimize_radial_manifold(initial_radii, directions, deadline, maxiter=2000, gtol=1e-10):
    """Optimize over the 55 radial coordinates (low-dimensional manifold)."""
    from scipy.optimize import minimize
    
    best_energy = _radial_energy_grad(initial_radii, directions)[0]
    best_radii = initial_radii.copy()
    
    def obj_and_grad(r):
        e, g = _radial_energy_grad(r, directions)
        return e, g
    
    try:
        result = minimize(
            obj_and_grad, initial_radii, jac=True, method='L-BFGS-B',
            bounds=[(0.01, 10.0)] * len(initial_radii),
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_radii = result.x
    except Exception:
        pass
    
    return best_radii, best_energy

def _optimize_full_structure(pos0, deadline, maxiter=5000, gtol=1e-12):
    """
    Optimize the full 3D structure using L-BFGS-B with tight tolerance.
    """
    from scipy.optimize import minimize
    
    best_energy = _lj_energy(pos0)
    best_pos = pos0.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos0.flatten()
    
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

def _relax_lbfgs(pos, deadline, maxiter=1000, gtol=1e-10):
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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x
    except Exception:
        pass
    
    return best_pos, best_energy

def _build_rigid_fragment_cluster(params):
    """
    Build 55-atom cluster from 13-atom core + 4 x 11-atom fragments.
    
    params (14-dim):
      params[0:3]   - core position (x, y, z)
      params[3:6]   - fragment 1 position relative to core
      params[6:9]   - fragment 2 position relative to core
      params[9:12]  - fragment 3 position relative to core
      params[12]    - rotation angle for all fragments (radians)
      params[13]    - inter-fragment distance scale
    
    Returns: (55, 3) array of positions
    """
    # Core: 13-atom Mackay (center + 12 icosahedron vertices)
    _, s1, _, _ = _get_mackay_icosahedron_coords()
    core_radius = 1.05
    core_local = np.vstack([np.array([[0.0, 0.0, 0.0]]), s1 * core_radius])
    
    # Fragment: 11-atom bicapped trigonal prism (approximated as icosahedron subset)
    # Use 11 of the 12 icosahedron vertices + 1 edge midpoint
    frag_local = np.vstack([s1[:10] * 1.0, s1[10:11] * 1.0])
    # Add one more atom at an edge midpoint for 11 total
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    edge_mid_dir = np.array([0.0, 1.0, phi]) / np.linalg.norm([0.0, 1.0, phi])
    frag_local = np.vstack([frag_local, edge_mid_dir * 1.75])
    
    # Apply rotation
    theta = params[12]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_t, -sin_t],
        [0.0, sin_t, cos_t]
    ])
    
    # Scale factor for inter-fragment distances
    scale = params[13]
    
    # Core position
    core_pos = params[0:3]
    core_atoms = core_local + core_pos
    
    # Four fragments at tetrahedral-like positions
    # Base directions for 4 fragments
    base_dirs = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    base_dirs = base_dirs / np.linalg.norm(base_dirs, axis=1, keepdims=True)
    
    # Fragment positions
    frag_positions = []
    for i in range(4):
        # Base direction scaled
        base_pos = base_dirs[i] * scale
        # Add relative offset from params
        if i == 0:
            rel_offset = params[3:6]
        elif i == 1:
            rel_offset = params[6:9]
        elif i == 2:
            rel_offset = params[9:12]
        else:
            rel_offset = np.array([0.0, 0.0, 0.0])  # Fragment 4 fixed
        
        frag_center = core_pos + base_pos + rel_offset
        frag_atoms = (frag_local @ R.T) + frag_center
        frag_positions.append(frag_atoms)
    
    all_atoms = np.vstack([core_atoms] + frag_positions)
    return all_atoms

def _rigid_fragment_energy_grad(params):
    """Compute energy and gradient for rigid fragment parameterization."""
    pos = _build_rigid_fragment_cluster(params)
    e, grad_pos = _lj_energy_grad(pos)
    
    # Numerical gradient for the 14 parameters
    grad_params = np.zeros(14)
    eps = 1e-7
    for i in range(14):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        
        pos_plus = _build_rigid_fragment_cluster(params_plus)
        pos_minus = _build_rigid_fragment_cluster(params_minus)
        
        e_plus = _lj_energy(pos_plus)
        e_minus = _lj_energy(pos_minus)
        
        grad_params[i] = (e_plus - e_minus) / (2 * eps)
    
    return e, grad_params

def _optimize_rigid_fragments(params0, deadline, maxiter=2000, gtol=1e-10):
    """Optimize the 14-dimensional rigid fragment parameters."""
    from scipy.optimize import minimize
    
    best_energy = _rigid_fragment_energy_grad(params0)[0]
    best_params = params0.copy()
    
    def obj_and_grad(p):
        e, g = _rigid_fragment_energy_grad(p)
        return e, g
    
    try:
        result = minimize(
            obj_and_grad, params0, jac=True, method='L-BFGS-B',
            bounds=[(-3.0, 3.0)] * 13 + [(0.5, 5.0)],
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-12}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x
    except Exception:
        pass
    
    return best_params, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum using radial manifold + Cartesian relaxation."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Get icosahedral directions
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    directions = np.vstack([center, s1, s2, s3])
    
    # 2. Initial radial coordinates
    initial_radii = np.concatenate([
        [0.0],
        [1.05] * 12,
        [1.75] * 30,
        [1.95] * 12
    ])
    
    # 3. Optimize on radial manifold first
    if time.time() < deadline - 80.0:
        radii_opt, energy_radial = _optimize_radial_manifold(
            initial_radii, directions, deadline, maxiter=3000, gtol=1e-10
        )
        pos_from_radial = directions * radii_opt[:, np.newaxis]
    else:
        pos_from_radial = _make_mackay_icosahedron(1.05, 1.75, 1.95)
    
    best_pos = pos_from_radial.copy()
    best_energy = _lj_energy(best_pos)
    
    # 4. Full Cartesian relaxation from the radial-optimized structure
    if time.time() < deadline - 60.0:
        pos_optimized, energy_optimized = _optimize_full_structure(
            best_pos, deadline, maxiter=5000, gtol=1e-12
        )
        if energy_optimized < best_energy:
            best_energy = energy_optimized
            best_pos = pos_optimized
    
    # 5. Rigid fragment assembly search
    if time.time() < deadline - 50.0:
        # Initial parameters: core at origin, fragments at tetrahedral positions
        scale0 = 2.5
        params0 = np.concatenate([
            [0.0, 0.0, 0.0],           # core position
            [0.0, 0.0, 0.0],           # frag 1 offset
            [0.0, 0.0, 0.0],           # frag 2 offset
            [0.0, 0.0, 0.0],           # frag 3 offset
            [0.0],                     # rotation
            [scale0]                   # scale
        ])
        
        params_opt, energy_frag = _optimize_rigid_fragments(
            params0, deadline, maxiter=2000, gtol=1e-10
        )
        pos_frag = _build_rigid_fragment_cluster(params_opt)
        
        # Relax the fragment-based structure
        if time.time() < deadline - 40.0:
            pos_frag_relaxed, energy_frag_relaxed = _relax_lbfgs(
                pos_frag, deadline, maxiter=3000, gtol=1e-12
            )
            if energy_frag_relaxed < best_energy:
                best_energy = energy_frag_relaxed
                best_pos = pos_frag_relaxed
    
    # 6. Multi-start stochastic search with perturbed candidates
    rng = np.random.default_rng(42)
    
    perturbation_scales = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2]
    
    polish_time = 25.0
    search_deadline = deadline - polish_time
    
    for scale in perturbation_scales:
        if time.time() >= search_deadline - 10.0:
            break
        
        n_perturbations = 3 if scale < 0.1 else 2 if scale < 0.5 else 1
        
        for _ in range(n_perturbations):
            if time.time() >= search_deadline - 5.0:
                break
            
            noise = rng.normal(0, scale, size=best_pos.shape)
            perturbed_pos = best_pos + noise
            
            diff = perturbed_pos[:, np.newaxis, :] - perturbed_pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            if min_dist < 0.5:
                scale_factor = 0.5 / min_dist * 1.01
                perturbed_pos = perturbed_pos * scale_factor
            
            if scale < 0.1:
                maxiter = 3000
                gtol = 1e-10
            elif scale < 0.5:
                maxiter = 2000
                gtol = 1e-9
            else:
                maxiter = 1500
                gtol = 1e-8
            
            pos_relaxed, energy_relaxed = _relax_lbfgs(
                perturbed_pos, search_deadline, maxiter=maxiter, gtol=gtol
            )
            
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed
    
    # 7. Final high-precision polish
    if time.time() < deadline - 5.0:
        pos_polished, energy_polished = _optimize_full_structure(
            best_pos, deadline, maxiter=10000, gtol=1e-14
        )
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
    
    # Final Check: Minimum distance constraint
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
    
    return best_pos
# EVOLVE-BLOCK-END
