"""Lennard-Jones cluster global minimum, 55 atoms — Mackay icosahedron with expanded radial manifold."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_MARGIN = 5.0  # seconds before the 150 s wall-clock kill
TOTAL_BUDGET = 150.0


def _lj_energy_and_grad(pos):
    """Compute total LJ energy and gradient for N atoms. Vectorized."""
    # pos: (N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)  # avoid div by zero
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 ** 2

    # Energy: 4 * sum over i<j of (r^-12 - r^-6)
    # The matrix has each pair twice, so divide by 2
    pair_e = 4.0 * (r12 - r6)
    np.fill_diagonal(pair_e, 0.0)
    energy = 0.5 * np.sum(pair_e)

    # Gradient: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)
    # In terms of r2: r^14 = r2^7, r^8 = r2^4
    # dE/dx_i = 4*(-12*r2^-7 + 6*r2^-4) * diff[i,j]

    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))  # (N, N)
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)

    return energy, grad


def _get_icosahedral_directions():
    """
    Returns:
        verts: (12, 3) array of unit vectors to icosahedron vertices.
        edge_mids: (30, 3) array of unit vectors to icosahedron edge midpoints.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # Icosahedral vertices (12)
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    # Find edges: pairs of vertices with minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    # The 30 edges have the smallest distances
    triu_indices = np.triu_indices(12, k=1)
    triu_dists = dists[triu_indices]
    sorted_indices = np.argsort(triu_dists)
    edge_pairs = list(zip(triu_indices[0][sorted_indices[:30]], triu_indices[1][sorted_indices[:30]]))
    
    edge_mids = []
    for i, j in edge_pairs:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        edge_mids.append(mid)
    
    edge_mids = np.array(edge_mids)
    return verts, edge_mids


def _construct_mackay_from_params(params, verts, edge_mids):
    """
    Construct the 55-atom Mackay icosahedron from parameters.
    params: array of shape (54,) or (3,)
    If shape (3,): [r1, r2, r3] — simple radial manifold
    If shape (54,): per-atom radial offsets from base Mackay geometry
      - params[0:12]: radial scale for 12 inner-shell atoms (vertex dirs)
      - params[12:42]: radial scale for 30 middle-shell atoms (edge mid dirs)  
      - params[42:54]: radial scale for 12 outer-shell atoms (vertex dirs)
    Base radii: r1=1.12, r2=2.24, r3=3.36
    """
    if len(params) == 3:
        r1, r2, r3 = params
        center = np.zeros((1, 3))
        shell1 = verts * r1
        shell2 = edge_mids * r2
        shell3 = verts * r3
        all_pos = np.vstack([center, shell1, shell2, shell3])
        return all_pos
    else:
        # Per-atom radial parameters
        base_r1 = 1.12
        base_r2 = 2.24
        base_r3 = 3.36
        
        # params[0:12] are multipliers for inner shell
        inner_scales = 1.0 + params[0:12]  # small deviations
        # params[12:42] are multipliers for middle shell
        mid_scales = 1.0 + params[12:42]
        # params[42:54] are multipliers for outer shell
        outer_scales = 1.0 + params[42:54]
        
        center = np.zeros((1, 3))
        
        # Inner shell: 12 atoms at vertex directions with individual radii
        shell1_radii = base_r1 * inner_scales
        shell1 = (verts * shell1_radii[:, np.newaxis])
        
        # Middle shell: 30 atoms at edge mid directions with individual radii
        shell2_radii = base_r2 * mid_scales
        shell2 = (edge_mids * shell2_radii[:, np.newaxis])
        
        # Outer shell: 12 atoms at vertex directions with individual radii
        shell3_radii = base_r3 * outer_scales
        shell3 = (verts * shell3_radii[:, np.newaxis])
        
        all_pos = np.vstack([center, shell1, shell2, shell3])
        return all_pos


def _radial_energy_and_grad(params, verts, edge_mids):
    """
    Compute energy and gradient with respect to parameters.
    params: shape (3,) for simple radial or (54,) for per-atom radial.
    """
    pos = _construct_mackay_from_params(params, verts, edge_mids)
    energy, grad_cart = _lj_energy_and_grad(pos)
    
    if len(params) == 3:
        # Simple radial manifold
        grad_r1 = np.sum(grad_cart[1:13] * verts)
        grad_r2 = np.sum(grad_cart[13:43] * edge_mids)
        grad_r3 = np.sum(grad_cart[43:55] * verts)
        grad_params = np.array([grad_r1, grad_r2, grad_r3])
    else:
        # Per-atom radial manifold
        # pos[i] = base_r * scale_i * dir_i for i in each shell
        # d(pos[i])/d(params[k]) = base_r * dir_i for the corresponding atom
        # grad_params[k] = grad_cart[i] . d(pos[i])/d(params[k])
        
        base_r1 = 1.12
        base_r2 = 2.24
        base_r3 = 3.36
        
        grad_params = np.zeros(54)
        
        # Inner shell: atoms 1-12 (indices 1:13 in pos)
        for i in range(12):
            # pos[1+i] = base_r1 * (1+params[i]) * verts[i]
            # d(pos[1+i])/d(params[i]) = base_r1 * verts[i]
            grad_params[i] = np.dot(grad_cart[1+i], base_r1 * verts[i])
        
        # Middle shell: atoms 13-42 (indices 13:43 in pos)
        for i in range(30):
            # pos[13+i] = base_r2 * (1+params[12+i]) * edge_mids[i]
            # d(pos[13+i])/d(params[12+i]) = base_r2 * edge_mids[i]
            grad_params[12+i] = np.dot(grad_cart[13+i], base_r2 * edge_mids[i])
        
        # Outer shell: atoms 43-54 (indices 43:55 in pos)
        for i in range(12):
            # pos[43+i] = base_r3 * (1+params[42+i]) * verts[i]
            # d(pos[43+i])/d(params[42+i]) = base_r3 * verts[i]
            grad_params[42+i] = np.dot(grad_cart[43+i], base_r3 * verts[i])
    
    return energy, grad_params


def construct():
    start_time = time.time()
    
    # Get geometric directions
    verts, edge_mids = _get_icosahedral_directions()
    
    # Phase 1: Optimize simple 3-parameter radial manifold with multiple restarts
    r0 = np.array([1.12, 2.24, 3.36])
    best_radii = r0.copy()
    best_e_radial = np.inf
    
    # Try multiple initial radii to find the best symmetric solution
    initial_radii_list = [
        np.array([1.12, 2.24, 3.36]),
        np.array([1.10, 2.20, 3.30]),
        np.array([1.14, 2.28, 3.42]),
        np.array([1.08, 2.16, 3.24]),
        np.array([1.16, 2.32, 3.48]),
    ]
    
    for r_init in initial_radii_list:
        elapsed = time.time() - start_time
        if elapsed > TOTAL_BUDGET - DEADLINE_MARGIN:
            break
        try:
            res_radial = minimize(
                _radial_energy_and_grad,
                r_init,
                args=(verts, edge_mids),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-14, 'maxls': 100}
            )
            e = res_radial.fun
            if e < best_e_radial:
                best_e_radial = e
                best_radii = res_radial.x
        except Exception:
            pass
    
    # Construct the best symmetric configuration
    best_pos = _construct_mackay_from_params(best_radii, verts, edge_mids)
    best_energy = _lj_energy_and_grad(best_pos)[0]
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > TOTAL_BUDGET - DEADLINE_MARGIN:
        return best_pos
    
    # Phase 2: Expand to per-atom radial manifold (54 params) with multiple restarts
    # Initialize with zero deviations (i.e., symmetric solution)
    best_params = np.zeros(54)
    best_e_expanded = best_energy
    
    # Try multiple perturbations of the radial parameters with diverse scales
    np.random.seed(42)
    n_restarts = 8
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > TOTAL_BUDGET - DEADLINE_MARGIN:
            break
        
        if restart == 0:
            params0 = np.zeros(54)
        elif restart == 1:
            params0 = np.random.normal(0, 0.01, 54)
        elif restart == 2:
            params0 = np.random.normal(0, 0.02, 54)
        elif restart == 3:
            params0 = np.random.normal(0, 0.005, 54)
        elif restart == 4:
            params0 = np.random.normal(0, 0.03, 54)
        else:
            # Larger perturbations to explore different basins
            scale = 0.02 + 0.01 * restart
            params0 = np.random.normal(0, scale, 54)
        
        try:
            res_expanded = minimize(
                _radial_energy_and_grad,
                params0,
                args=(verts, edge_mids),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1500, 'gtol': 1e-15, 'maxls': 300}
            )
            e = res_expanded.fun
            if e < best_e_expanded:
                best_e_expanded = e
                best_params = res_expanded.x
        except Exception:
            pass
    
    expanded_pos = _construct_mackay_from_params(best_params, verts, edge_mids)
    e_exp = _lj_energy_and_grad(expanded_pos)[0]
    if e_exp < best_energy:
        best_pos = expanded_pos
        best_energy = e_exp
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > TOTAL_BUDGET - DEADLINE_MARGIN:
        return best_pos
    
    # Phase 3: Full Cartesian L-BFGS polish with very tight tolerance
    try:
        res_cart = minimize(
            _lj_energy_and_grad,
            best_pos,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 8000, 'gtol': 1e-16, 'maxls': 300}
        )
        if res_cart.fun < best_energy:
            best_pos = res_cart.x
            best_energy = res_cart.fun
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > TOTAL_BUDGET - DEADLINE_MARGIN:
        return best_pos
    
    # Phase 4: Final high-precision polish using BFGS (unconstrained, local)
    try:
        res_bfgs = minimize(
            _lj_energy_and_grad,
            best_pos,
            jac=True,
            method='BFGS',
            options={'maxiter': 2000, 'gtol': 1e-18, 'disp': False}
        )
        if res_bfgs.fun < best_energy:
            best_pos = res_bfgs.x
            best_energy = res_bfgs.fun
    except Exception:
        pass
    
    # Phase 5: One more L-BFGS polish after BFGS to ensure convergence
    elapsed = time.time() - start_time
    if elapsed < TOTAL_BUDGET - DEADLINE_MARGIN:
        try:
            res_final = minimize(
                _lj_energy_and_grad,
                best_pos,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-18, 'maxls': 200}
            )
            if res_final.fun < best_energy:
                best_pos = res_final.x
                best_energy = res_final.fun
        except Exception:
            pass
    
    # Phase 6: Ultra-precision final polish with even tighter tolerance
    elapsed = time.time() - start_time
    if elapsed < TOTAL_BUDGET - DEADLINE_MARGIN:
        try:
            res_ultra = minimize(
                _lj_energy_and_grad,
                best_pos,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'gtol': 1e-20, 'maxls': 100}
            )
            if res_ultra.fun < best_energy:
                best_pos = res_ultra.x
                best_energy = res_ultra.fun
        except Exception:
            pass
    
    # Final validation: ensure atoms are at least 0.5 apart
    # Compute minimum distance
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    min_dist = np.sqrt(np.min(r2))
    
    if min_dist < 0.5:
        # Fallback to the symmetric radial solution if validation fails
        best_pos = _construct_mackay_from_params(best_radii, verts, edge_mids)
    
    return best_pos
