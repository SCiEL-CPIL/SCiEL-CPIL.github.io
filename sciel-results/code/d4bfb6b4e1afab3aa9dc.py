"""Lennard-Jones cluster global minimum, 55 atoms."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE = 148.0  # Safety margin before 150s kill

def _lj_energy(pos):
    """Compute total LJ energy for N atoms (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def _lj_energy_grad(pos):
    """Compute LJ energy and gradient (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    # dE/dpos_i = sum_{j!=i} 4 * (-12*r_ij^-14 + 6*r_ij^-8) * (pos_i - pos_j)
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def _min_distance(positions):
    """Compute minimum pairwise distance."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _get_mackay_directions():
    """
    Get the 55 unit direction vectors for a Mackay icosahedron M(2,2,2).
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edges of icosahedron (adjacent vertices)
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    
    # 30 edge midpoints (normalized)
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    # Total: 1 (center) + 12 (shell 1) + 30 (shell 2) + 12 (shell 3) = 55
    directions = np.zeros((55, 3))
    shell_ids = np.zeros(55, dtype=int)
    
    # Atom 0: center
    directions[0] = [0.0, 0.0, 0.0]
    shell_ids[0] = 0
    
    # Atoms 1-12: Shell 1 (icosahedron vertices)
    for i in range(12):
        directions[1+i] = verts[i]
        shell_ids[1+i] = 1
    
    # Atoms 13-42: Shell 2 (edge midpoints)
    for i in range(30):
        directions[13+i] = midpoints[i]
        shell_ids[13+i] = 2
    
    # Atoms 43-54: Shell 3 (icosahedron vertices, outer)
    for i in range(12):
        directions[43+i] = verts[i]
        shell_ids[43+i] = 3
    
    return directions, shell_ids

def _make_mackay_from_radii(radii):
    """Generate Mackay icosahedron positions given shell radii [r1, r2, r3]."""
    directions, shell_ids = _get_mackay_directions()
    
    pos = np.zeros((55, 3))
    for i in range(1, 55):
        shell = shell_ids[i]
        if shell == 1:
            pos[i] = directions[i] * radii[0]
        elif shell == 2:
            pos[i] = directions[i] * radii[1]
        elif shell == 3:
            pos[i] = directions[i] * radii[2]
    
    return pos

def _relax(positions, tol=1e-6, maxiter=300, deadline=None):
    """Relax positions using L-BFGS-B."""
    def objective(x):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError()
        pos = x.reshape(N, 3)
        e, g = _lj_energy_grad(pos)
        return e, g.flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'gtol': tol, 'maxiter': maxiter, 'maxls': 30, 'ftol': 1e-10})
        if result.success or result.fun < 1e6:
            return result.x.reshape(N, 3), result.fun
    except Exception:
        pass
    return positions, _lj_energy(positions)

def _radial_manifold_optimize(positions, directions, maxiter=300, deadline=None):
    """
    Optimize individual radial distances for all 54 non-center atoms
    on the icosahedral direction manifold.
    """
    n = len(positions)
    # Extract current radial distances for non-center atoms
    radii = np.zeros(n - 1)
    for i in range(1, n):
        r = np.linalg.norm(positions[i])
        radii[i-1] = max(r, 0.5)
    
    def objective(r_vec):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError()
        r_vec = np.maximum(r_vec, 0.5)
        pos = np.zeros((n, 3))
        pos[0] = [0.0, 0.0, 0.0]
        for i in range(1, n):
            pos[i] = directions[i] * r_vec[i-1]
        e, g = _lj_energy_grad(pos)
        # Compute gradient with respect to radii
        # dE/dr_i = sum over j of dE/dpos_i * dpos_i/dr_i = grad_i * directions[i]
        grad_r = np.zeros(n - 1)
        for i in range(1, n):
            grad_r[i-1] = np.dot(g[i], directions[i])
        return e, grad_r
    
    try:
        res = minimize(objective, radii, method='L-BFGS-B', jac=True,
                      options={'gtol': 1e-8, 'maxiter': maxiter, 'maxls': 30, 'ftol': 1e-10})
        best_r = np.maximum(res.x, 0.5)
        pos = np.zeros((n, 3))
        pos[0] = [0.0, 0.0, 0.0]
        for i in range(1, n):
            pos[i] = directions[i] * best_r[i-1]
        return pos, res.fun
    except Exception:
        return positions, _lj_energy(positions)

def _perturb(positions, scale=0.1, seed=None):
    """Add small random perturbation."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    return positions + rng.randn(*positions.shape) * scale

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE - elapsed()
    
    # Get Mackay directions
    directions, shell_ids = _get_mackay_directions()
    
    # Start from the standard Mackay icosahedron geometry
    # Use physically motivated initial radii for LJ (sigma=1)
    # Shell 1: ~1.0, Shell 2: ~1.8, Shell 3: ~2.6 (typical LJ cluster spacing)
    base_radii = np.array([1.0, 1.80, 2.60])
    base_pos = _make_mackay_from_radii(base_radii)
    
    best_energy = np.inf
    best_positions = None
    
    # Helper to validate and update best
    def update_best(pos, energy):
        nonlocal best_energy, best_positions
        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
            return
        min_d = _min_distance(pos)
        if min_d < 0.5:
            return
        if energy < best_energy:
            best_energy = energy
            best_positions = pos.copy()
    
    # Phase 1a: Optimize shell radii on the manifold first (fast, 3D optimization)
    try:
        opt_pos, opt_e = _radial_manifold_optimize(base_pos, directions, maxiter=100, deadline=start_time + DEADLINE)
        update_best(opt_pos, opt_e)
    except Exception:
        pass
    
    # Phase 1b: Full 3D relaxation from the radial-optimized structure
    if best_positions is not None:
        try:
            relaxed, e = _relax(best_positions, tol=1e-7, maxiter=200, deadline=start_time + DEADLINE)
            update_best(relaxed, e)
        except Exception:
            pass
    else:
        try:
            relaxed, e = _relax(base_pos, tol=1e-7, maxiter=200, deadline=start_time + DEADLINE)
            update_best(relaxed, e)
        except Exception:
            pass
    
    if best_positions is None:
        best_positions = base_pos.copy()
        best_energy = _lj_energy(base_pos)

    # Phase 2: Multi-start stochastic search with adaptive strategy
    rng = np.random.RandomState(42)
    
    # Adaptive perturbation: start small, increase if stuck
    base_scale = 0.05
    stuck_count = 0
    best_energy_snapshot = best_energy
    
    attempt = 0
    while remaining() > 5:
        # Adaptive scale: increase when stuck, decrease when improving
        if best_energy < best_energy_snapshot - 1e-4:
            # Improved, reset stuck counter and use smaller scale
            stuck_count = 0
            best_energy_snapshot = best_energy
            scale = base_scale
        else:
            stuck_count += 1
            # Gradually increase scale when stuck
            scale = base_scale * (1.0 + 0.3 * stuck_count)
            scale = min(scale, 3.0)  # Cap at 3.0
        
        # 70% chance to perturb current best, 30% to perturb original Mackay
        if rng.rand() < 0.7 and best_positions is not None:
            base_for_perturb = best_positions
        else:
            base_for_perturb = base_pos
            
        # Add Gaussian noise
        perturbed = base_for_perturb + rng.randn(*base_for_perturb.shape) * scale
        
        # Relax with tighter tolerance for more precise local minima
        try:
            relaxed, e = _relax(perturbed, tol=1e-7, maxiter=150, deadline=start_time + DEADLINE)
            update_best(relaxed, e)
        except Exception:
            pass
            
        attempt += 1
        
        # Every 8 attempts, try a large random restart from Mackay
        if attempt % 8 == 0 and remaining() > 10:
             large_scale = 2.5
             perturbed = base_pos + rng.randn(*base_pos.shape) * large_scale
             try:
                 relaxed, e = _relax(perturbed, tol=1e-7, maxiter=100, deadline=start_time + DEADLINE)
                 update_best(relaxed, e)
             except Exception:
                 pass

    # Phase 3: Final high-precision polish
    if remaining() > 2 and best_positions is not None:
        # First do a radial manifold polish (fast, keeps symmetry)
        try:
            polished_radial, e_radial = _radial_manifold_optimize(
                best_positions, directions, maxiter=200, deadline=start_time + DEADLINE - 1.0)
            if e_radial < best_energy:
                best_energy = e_radial
                best_positions = polished_radial.copy()
        except Exception:
            pass
        
        # Then full 3D high-precision polish
        try:
            relaxed_final, e_final = _relax(best_positions, tol=1e-12, maxiter=500, deadline=start_time + DEADLINE - 0.5)
            update_best(relaxed_final, e_final)
        except Exception:
            pass

    if best_positions is None:
        # Fallback
        best_positions = base_pos
        best_energy = _lj_energy(base_pos)

    return best_positions
