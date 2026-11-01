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

def _get_mackay_icosahedron_coords():
    """
    Returns unit vectors for the 55 positions in a Mackay icosahedron.
    1 center (0,0,0)
    12 first shell (icosahedron vertices)
    30 second shell (icosahedron edge midpoints)
    12 third shell (icosahedron vertices, same direction as first shell)
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
    
    # Construct full unit vectors
    # Order: Center, Shell1 (12), Shell2 (30), Shell3 (12)
    center = np.array([[0.0, 0.0, 0.0]])
    shell1_dir = verts_unit
    shell2_dir = edge_mids
    shell3_dir = verts_unit
    
    return center, shell1_dir, shell2_dir, shell3_dir

def _build_direction_manifold():
    """Build the 54 unit direction vectors for non-center atoms."""
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    dirs = np.vstack([s1, s2, s3])
    return dirs

def _radial_to_positions(radii, dirs):
    """Convert 54 radial distances to 54 positions (plus center at origin)."""
    pos_noncenter = dirs * radii[:, np.newaxis]
    center = np.array([[0.0, 0.0, 0.0]])
    return np.vstack([center, pos_noncenter])

def _radial_energy_and_grad(radii, dirs):
    """Energy and gradient w.r.t. radial parameters on the direction manifold."""
    pos = _radial_to_positions(radii, dirs)
    e, grad_pos = _lj_energy_grad(pos)
    grad_rad = np.sum(grad_pos[1:] * dirs, axis=1)
    return e, grad_rad

def _optimize_radial_manifold(dirs, r1_init, r2_init, r3_init, deadline):
    """Optimize all 54 individual radial distances on the icosahedral manifold."""
    x0 = np.zeros(54)
    x0[:12] = r1_init
    x0[12:42] = r2_init
    x0[42:54] = r3_init
    
    def obj_and_grad(x):
        return _radial_energy_and_grad(x, dirs)
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(0.3, 5.0)] * 54,
            options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-14, 'maxls': 200}
        )
        return result.x, result.fun
    except Exception:
        return x0, obj_and_grad(x0)[0]

def _relax_lbfgs(pos, deadline, maxiter=1000, gtol=1e-10):
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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

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

def construct():
    """Construct 55-atom LJ cluster near global minimum using multi-start stochastic search."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Create base Mackay icosahedron with reasonable initial radii
    # Standard Mackay radii are roughly 1.05, 1.75, 1.95
    base_pos = _make_mackay_icosahedron([1.05, 1.75, 1.95])
    
    best_pos = base_pos.copy()
    best_energy = _lj_energy(base_pos)
    
    # 2. Optimize the radial distances specifically (fast, low-dimensional)
    if time.time() < deadline - 10.0:
        pos_radial_opt = _optimize_radii(base_pos, deadline)
        energy_radial = _lj_energy(pos_radial_opt)
        if energy_radial < best_energy:
            best_energy = energy_radial
            best_pos = pos_radial_opt
    
    # 3. First, do a high-precision full 3D relaxation of the symmetric structure
    if time.time() < deadline - 60.0:
        pos_optimized, energy_optimized = _optimize_full_structure(best_pos, deadline, maxiter=5000, gtol=1e-12)
        if energy_optimized < best_energy:
            best_energy = energy_optimized
            best_pos = pos_optimized
    
    # 4. Multi-start stochastic search with perturbed candidates
    rng = np.random.default_rng(42)
    
    # Define perturbation scales to try (from small to large)
    perturbation_scales = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    
    # Reserve time for final polish
    polish_time = 20.0
    search_deadline = deadline - polish_time
    
    # Try perturbed starts
    for scale in perturbation_scales:
        if time.time() >= search_deadline - 10.0:
            break
        
        # Generate multiple perturbations at this scale for better coverage
        n_perturbations = 5 if scale < 0.05 else 2
        
        for _ in range(n_perturbations):
            if time.time() >= search_deadline - 5.0:
                break
            
            # Generate a perturbed version
            noise = rng.normal(0, scale, size=best_pos.shape)
            perturbed_pos = best_pos + noise
            
            # Check minimum distance constraint
            diff = perturbed_pos[:, np.newaxis, :] - perturbed_pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            if min_dist < 0.5:
                # Scale up to satisfy constraint
                scale_factor = 0.5 / min_dist * 1.01
                perturbed_pos = perturbed_pos * scale_factor
            
            # Relax this perturbed candidate
            if scale < 0.1:
                maxiter = 3000
                gtol = 1e-10
            else:
                maxiter = 1500
                gtol = 1e-8
            
            pos_relaxed, energy_relaxed = _relax_lbfgs(perturbed_pos, search_deadline, maxiter=maxiter, gtol=gtol)
            
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed
    
    # 5. Final high-precision polish with very tight tolerance
    if time.time() < deadline - 5.0:
        pos_polished, energy_polished = _optimize_full_structure(best_pos, deadline, maxiter=10000, gtol=1e-14)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
        
        # Second polish pass to ensure convergence
        if time.time() < deadline - 1.0:
            pos_polished2, energy_polished2 = _optimize_full_structure(best_pos, deadline, maxiter=5000, gtol=1e-16)
            if energy_polished2 < best_energy:
                best_energy = energy_polished2
                best_pos = pos_polished2

    # Final Check: Minimum distance constraint
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        # If constraint is violated, scale up
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
    
    return best_pos

def _make_mackay_icosahedron(radii):
    """
    Construct a two-shell Mackay icosahedron for 55 atoms given specific radii.
    radii: [r1, r2, r3]
    """
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    
    first_shell = s1 * radii[0]
    second_inner = s2 * radii[1]
    second_outer = s3 * radii[2]
    
    all_pos = np.vstack([center, first_shell, second_inner, second_outer])
    return all_pos

def _optimize_radii(pos0, deadline):
    """
    Optimize the radial distances of the shells while keeping directions fixed.
    This is a 3D optimization problem (r1, r2, r3) which is much faster than 165D.
    """
    from scipy.optimize import minimize
    
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    
    # Extract initial radii from pos0
    r1_init = np.linalg.norm(pos0[1])
    r2_init = np.linalg.norm(pos0[13])
    r3_init = np.linalg.norm(pos0[43])
    
    def radii_energy(radii):
        r1, r2, r3 = radii
        pos = _make_mackay_icosahedron([r1, r2, r3])
        return _lj_energy(pos)
    
    x0 = np.array([r1_init, r2_init, r3_init])
    
    try:
        result = minimize(
            radii_energy, x0, method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-15}
        )
        if result.fun < radii_energy(x0):
            return _make_mackay_icosahedron(result.x)
    except Exception:
        pass
    
    return pos0

# EVOLVE-BLOCK-END
