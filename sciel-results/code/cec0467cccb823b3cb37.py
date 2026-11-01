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
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def _check_min_dist(pos, min_d=0.5):
    """Check if all pairs are at least min_d apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d

def _optimize_radii(base_dirs, deadline):
    """
    Optimize the three shell radii using Nelder-Mead to find the best 
    starting geometry before full 3D relaxation.
    """
    from scipy.optimize import minimize
    
    center, s1, s2, s3 = base_dirs
    
    def radii_energy(radii):
        r1, r2, r3 = radii
        # Enforce minimum distance constraint softly via penalty
        first_shell = s1 * r1
        second_inner = s2 * r2
        second_outer = s3 * r3
        all_pos = np.vstack([center, first_shell, second_inner, second_outer])
        
        # Check min distance
        diff = all_pos[:, np.newaxis, :] - all_pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 0.5:
            penalty = 1000.0 * (0.5 - min_dist) ** 2
        else:
            penalty = 0.0
        
        return _lj_energy(all_pos) + penalty
    
    # Start from reasonable values
    x0 = np.array([1.05, 1.75, 1.95])
    
    try:
        result = minimize(
            radii_energy, x0, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-10}
        )
        return result.x
    except Exception:
        return x0

def construct():
    """Construct 55-atom LJ cluster near global minimum using multi-start stochastic search."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Create base Mackay icosahedron with reasonable initial radii
    base_pos = _make_mackay_icosahedron(1.05, 1.75, 1.95)
    
    best_pos = base_pos.copy()
    best_energy = _lj_energy(base_pos)
    
    # 2. Optimize shell radii first to get a better starting geometry
    if time.time() < deadline - 60.0:
        center, s1, s2, s3 = _get_mackay_icosahedron_coords()
        optimal_radii = _optimize_radii((center, s1, s2, s3), deadline)
        base_pos = _make_mackay_icosahedron(*optimal_radii)
        best_energy = _lj_energy(base_pos)
        best_pos = base_pos.copy()
    
    # 3. First, do a high-precision full 3D relaxation of the symmetric structure
    if time.time() < deadline - 60.0:
        pos_optimized, energy_optimized = _relax_lbfgs(base_pos, deadline, maxiter=5000, gtol=1e-12)
        if energy_optimized < best_energy:
            best_energy = energy_optimized
            best_pos = pos_optimized
        base_pos = pos_optimized  # Use this as the new base for perturbations
    
    # 4. Multi-start stochastic search with perturbed candidates
    rng = np.random.default_rng(42)
    
    # Define perturbation scales to try (from small to large)
    perturbation_scales = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2]
    
    # Reserve time for final polish
    polish_time = 20.0
    search_deadline = deadline - polish_time
    
    # Try perturbed starts
    for scale in perturbation_scales:
        if time.time() >= search_deadline - 10.0:
            break
        
        # Generate multiple perturbations at this scale for better coverage
        n_perturbations = 3 if scale < 0.1 else 2 if scale < 0.5 else 1
        
        for _ in range(n_perturbations):
            if time.time() >= search_deadline - 5.0:
                break
            
            # Generate a perturbed version
            noise = rng.normal(0, scale, size=base_pos.shape)
            perturbed_pos = base_pos + noise
            
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
            # Use adaptive iteration count based on perturbation scale
            if scale < 0.1:
                maxiter = 3000
                gtol = 1e-10
            elif scale < 0.5:
                maxiter = 2000
                gtol = 1e-9
            else:
                maxiter = 1500
                gtol = 1e-8
            
            pos_relaxed, energy_relaxed = _relax_lbfgs(perturbed_pos, search_deadline, maxiter=maxiter, gtol=gtol)
            
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed
    
    # 5. Final high-precision polish with very tight tolerance
    if time.time() < deadline - 5.0:
        pos_polished, energy_polished = _relax_lbfgs(best_pos, deadline, maxiter=10000, gtol=1e-14)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
    
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
# EVOLVE-BLOCK-END
