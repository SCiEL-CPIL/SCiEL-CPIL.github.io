import numpy as np
from scipy.optimize import minimize
import time

N = 55

def _lj_energy_grad(pos):
    """
    Compute total LJ energy and gradient for a set of positions.
    Vectorized for speed using einsum.
    """
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    energy = 4.0 * (r12 - r6).sum() / 2.0
    
    # Gradient: dE/dpos_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (pos_i - pos_j)
    # In terms of r2: 4*(-12*r2^-7 + 6*r2^-4)
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    
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
    
    # Find edges (shortest distances between vertices)
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
    radii: shape (N-1,) for the 54 non-center atoms
    dirs: shape (N-1, 3) unit direction vectors
    """
    n_noncenter = len(radii)
    pos = np.zeros((n_noncenter + 1, 3))
    pos[1:] = dirs * radii[:, np.newaxis]
    
    e, g_pos = _lj_energy_grad(pos)
    g_r = np.sum(g_pos[1:] * dirs, axis=1)
    
    return e, g_r

def _optimize_radial_manifold(dirs, r0, deadline, maxiter=200):
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
            options={'maxiter': maxiter, 'gtol': 1e-8, 'maxls': 50, 'ftol': 1e-10}
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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50, 'ftol': 1e-12}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def _optimize_shell_radii_nm(dirs, deadline, maxiter_nm=80):
    """
    Optimize the three shell radii using Nelder-Mead.
    This finds a better starting point for the radial manifold optimization.
    """
    def objective(radii_3d):
        r1, r2, r3 = radii_3d
        r1 = max(r1, 0.5)
        r2 = max(r2, 0.5)
        r3 = max(r3, 0.5)
        
        # Build radii vector for all 54 non-center atoms
        radii = np.concatenate([
            np.full(12, r1),
            np.full(30, r2),
            np.full(12, r3)
        ])
        
        # Compute energy on radial manifold
        e, _ = _radial_energy_and_grad(radii, dirs)
        return e
    
    # Try multiple initial guesses for the three shell radii
    initial_guesses = [
        np.array([1.12, 1.75, 1.95]),
        np.array([1.0, 1.8, 2.6]),
        np.array([1.1, 1.9, 2.7]),
        np.array([0.95, 1.7, 2.5]),
        np.array([1.05, 1.85, 2.65]),
    ]
    
    best_radii = None
    best_energy = np.inf
    
    for guess in initial_guesses:
        if time.time() > deadline - 20.0:
            break
        
        try:
            result = minimize(
                objective, guess, method='Nelder-Mead',
                options={'maxiter': maxiter_nm, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_radii = result.x.copy()
        except Exception:
            continue
    
    if best_radii is None:
        # Fallback
        best_radii = np.array([1.12, 1.75, 1.95])
        best_energy = objective(best_radii)
    
    # Build full radii vector
    radii_full = np.concatenate([
        np.full(12, best_radii[0]),
        np.full(30, best_radii[1]),
        np.full(12, best_radii[2])
    ])
    
    return radii_full, best_energy

def construct():
    """
    Constructs the 55-atom LJ cluster by:
    1. Optimizing shell radii with Nelder-Mead to find a good starting point
    2. Optimizing on the radial manifold (54D) 
    3. Full 165D relaxation
    4. Multi-start with symmetry-breaking perturbations if time permits
    5. Final high-precision polish
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Get icosahedral directions
    center, s1, s2, s3 = _get_mackay_icosahedron_unit_vectors()
    dirs = np.vstack([s1, s2, s3])  # shape (54, 3)
    
    best_pos = None
    best_energy = np.inf
    
    # Step 1: Optimize shell radii with Nelder-Mead (fast, 3D)
    if time.time() < deadline - 100.0:
        r_nm, e_nm = _optimize_shell_radii_nm(dirs, deadline, maxiter_nm=60)
        best_energy = e_nm
        pos_nm = np.zeros((55, 3))
        pos_nm[1:] = dirs * r_nm[:, np.newaxis]
        best_pos = pos_nm.copy()
    
    # Step 2: Optimize on radial manifold (54D) from the Nelder-Mead result
    if time.time() < deadline - 50.0:
        r0 = r_nm if 'r_nm' in locals() else np.concatenate([
            np.full(12, 1.12), np.full(30, 1.75), np.full(12, 1.95)
        ])
        r_opt, e_radial = _optimize_radial_manifold(dirs, r0, deadline, maxiter=300)
        
        pos_radial = np.zeros((55, 3))
        pos_radial[1:] = dirs * r_opt[:, np.newaxis]
        
        if e_radial < best_energy:
            best_energy = e_radial
            best_pos = pos_radial.copy()
    
    # Step 3: Full 3D relaxation from radial manifold result
    if time.time() < deadline - 30.0 and best_pos is not None:
        pos_relaxed, energy_relaxed = _relax_lbfgs(best_pos, deadline, maxiter=400, gtol=1e-8)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed.copy()
    
    # Step 4: Multi-start search with symmetry-breaking perturbations
    rng = np.random.default_rng(42)
    candidates = []
    
    if best_pos is not None:
        candidates.append((best_energy, best_pos.copy()))
    
    for i in range(4):
        if time.time() > deadline - 25.0:
            break
        
        pos_seed = best_pos.copy() if best_pos is not None else _make_mackay_icosahedron(1.12, 1.75, 1.95)
        
        # Add random noise to break symmetry
        noise_scale = 0.02 + 0.015 * i
        pos_seed += rng.normal(0, noise_scale, size=pos_seed.shape)
        
        # Ensure minimum distance constraint (0.5)
        diff = pos_seed[:, np.newaxis, :] - pos_seed[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 0.5:
            scale_factor = 0.5 / min_dist * 1.01
            pos_seed *= scale_factor
        
        # Relax this candidate
        maxiter_this = 200 if i < 2 else 100
        pos_relaxed, energy_relaxed = _relax_lbfgs(pos_seed, deadline, maxiter=maxiter_this, gtol=1e-7)
        
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed.copy()
        
        candidates.append((energy_relaxed, pos_relaxed))
    
    # Step 5: Final high-precision polish
    if time.time() < deadline - 10.0 and best_pos is not None:
        pos_final, energy_final = _relax_lbfgs(
            best_pos, deadline, maxiter=500, gtol=1e-10
        )
        if energy_final < best_energy:
            best_energy = energy_final
            best_pos = pos_final
    
    # Final Safety Check: Minimum Distance
    if best_pos is not None:
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 0.5:
            scale_factor = 0.5 / min_dist * 1.01
            best_pos *= scale_factor
    
    if best_pos is None:
        # Fallback: return standard Mackay
        best_pos = _make_mackay_icosahedron(1.12, 1.75, 1.95)
    
    return best_pos
