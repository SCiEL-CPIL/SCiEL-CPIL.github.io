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
    # Avoid division by zero or negative if r2 is 0
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0) # Avoid div by 0 in calculation, handled later
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    # Gradient calculation
    # dE/dr2 = 4 * (-6/r2^4 + 12/r2^7)
    # dE/dpos_i = sum_j dE/dr2_ij * 2*(pos_i - pos_j)
    # Let's use the explicit form:
    # grad_i = sum_j 4 * (-12/r2^7 + 6/r2^4) * (pos_i - pos_j)
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
    shell3_dir = verts_unit # Same directions as shell 1
    
    return center, shell1_dir, shell2_dir, shell3_dir

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms."""
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    
    first_shell = s1 * r1
    second_inner = s2 * r2
    second_outer = s3 * r3
    
    all_pos = np.vstack([center, first_shell, second_inner, second_outer])
    return all_pos

def _optimize_radii(deadline):
    """
    Optimize the three shell radii (r1, r2, r3) while keeping the 
    angular structure fixed. This provides a much better initial guess 
    for the full 3D relaxation.
    """
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    
    def radii_energy(radii):
        r1, r2, r3 = radii
        pos = np.vstack([
            center,
            s1 * r1,
            s2 * r2,
            s3 * r3
        ])
        return _lj_energy(pos)
    
    # Initial guess
    x0 = np.array([1.05, 1.75, 1.95])
    
    # Use Nelder-Mead for small 3D problem, it's robust
    from scipy.optimize import minimize
    
    result = minimize(radii_energy, x0, method='Nelder-Mead', 
                      options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
    
    return result.x

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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def _newton_polish(pos, deadline, maxiter=50, tol=1e-14):
    """
    Final high-precision polish using Newton-Raphson with a numerically 
    computed Hessian. Only used when we have a well-converged structure 
    from L-BFGS and need to squeeze out the last bits of precision.
    """
    from scipy.linalg import cho_factor, cho_solve
    
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    n = pos.shape[0]
    dim = n * 3
    
    # Precompute the Hessian using finite differences (central differences)
    # This is expensive but we only do it once per polish step
    def compute_hessian(pos_arr):
        """Compute Hessian via central finite differences on the gradient."""
        h = 1e-6
        H = np.zeros((dim, dim))
        grad0, _ = _lj_energy_grad(pos_arr)
        g0 = grad0.flatten()
        
        for i in range(dim):
            pos_plus = pos_arr.copy()
            pos_plus.flat[i] += h
            pos_minus = pos_arr.copy()
            pos_minus.flat[i] -= h
            
            _, g_plus = _lj_energy_grad(pos_plus)
            _, g_minus = _lj_energy_grad(pos_minus)
            
            H[i, :] = (g_plus.flatten() - g_minus.flatten()) / (2 * h)
        
        return H
    
    for it in range(maxiter):
        if time.time() > deadline:
            break
        
        e, g = _lj_energy_grad(pos)
        grad = g.flatten()
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            break
        
        # Compute Hessian
        H = compute_hessian(pos)
        
        # Solve H * dx = -grad
        try:
            lu, piv = cho_factor(H + 1e-12 * np.eye(dim))
            dx = cho_solve((lu, piv), -grad)
        except Exception:
            # Fallback to steepest descent if Hessian is not positive definite
            dx = -grad / (grad_norm * grad_norm) * 0.1
        
        # Line search
        alpha = 1.0
        for _ in range(20):
            new_pos = pos + alpha * dx.reshape(n, 3)
            new_e = _lj_energy(new_pos)
            if new_e < e:
                break
            alpha *= 0.5
        
        if new_e < best_energy:
            best_energy = new_e
            best_pos = new_pos.copy()
        
        pos = new_pos
        if abs(e - new_e) < 1e-16:
            break
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Optimize radii of the symmetric structure (fast, ~seconds)
    # This step significantly reduces the number of iterations needed for 3D relaxation
    try:
        if time.time() < deadline - 100.0:
            optimal_radii = _optimize_radii(deadline)
            r1, r2, r3 = optimal_radii
        else:
            r1, r2, r3 = 1.05, 1.75, 1.95
    except Exception:
        r1, r2, r3 = 1.05, 1.75, 1.95

    # Create initial positions with optimized radii
    init_pos = _make_mackay_icosahedron(r1, r2, r3)
    
    # 2. Full 3D Relaxation
    # We split the remaining time into two phases: 
    # Phase A: General relaxation with looser tolerance
    # Phase B: Precision polish with tighter tolerance
    
    best_pos = init_pos
    best_energy = _lj_energy(init_pos)
    
    # Phase A: ~100 seconds
    if time.time() < deadline - 40.0:
        pos_a, energy_a = _relax_lbfgs(init_pos, deadline, maxiter=2000, gtol=1e-10)
        if energy_a < best_energy:
            best_energy = energy_a
            best_pos = pos_a
    
    # Phase B: ~30 seconds, high precision
    if time.time() < deadline - 10.0:
        pos_b, energy_b = _relax_lbfgs(best_pos, deadline, maxiter=5000, gtol=1e-14)
        if energy_b < best_energy:
            best_energy = energy_b
            best_pos = pos_b
    
    # Phase C: Newton polish for final precision (only if time remains)
    if time.time() < deadline - 2.0:
        pos_c, energy_c = _newton_polish(best_pos, deadline, maxiter=30, tol=1e-15)
        if energy_c < best_energy:
            best_energy = energy_c
            best_pos = pos_c
    
    # Final Check: Minimum distance constraint
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        # If constraint is violated, scale up. 
        # This is a heuristic and likely not needed for the icosahedron,
        # but ensures validity.
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
        # Re-energy if scaled, though this breaks the minimum.
        # Better to just return if valid, else this is the best we can do.
        # For N=55 Icosa, min dist is > 1.0, so this won't trigger.

    return best_pos
# EVOLVE-BLOCK-END
