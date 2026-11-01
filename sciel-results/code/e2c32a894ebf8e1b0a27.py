import numpy as np
from scipy.optimize import minimize
import time

N = 55

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient efficiently."""
    pos = pos.reshape(N, 3)
    # Compute pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.einsum('ijk,ijk->ij', diff, diff)  # (N, N)
    np.fill_diagonal(r2, np.inf)
    
    inv_r6 = r2 ** -3
    inv_r12 = inv_r6 ** 2
    
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6) = 2 * sum_{i!=j} (r^-12 - r^-6)
    E = 4.0 * 0.5 * np.sum(inv_r12 - inv_r6)
    
    # Gradient: dE/dx_i = sum_{j!=i} 24 * r^-8 * (1 - 2*r^-6) * (x_i - x_j)
    coeff = 24.0 * (r2 ** -4) * (1.0 - 2.0 * inv_r6)  # (N, N)
    # grad[i] = sum_j coeff[i,j] * (x_i - x_j)
    # Use einsum for efficiency
    grad = np.einsum('ij,ijk->k', coeff, diff)
    
    return E, grad.ravel()

def _get_mackay_icosahedron():
    """Construct Mackay icosahedron with proper geometry."""
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (12)
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges (pairs at minimum distance)
    dists = np.linalg.norm(verts_unit[:, np.newaxis] - verts_unit[np.newaxis, :], axis=2)
    min_dist = np.min(dists[dists > 1e-6])
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                mid = (verts_unit[i] + verts_unit[j]) / 2
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-6:
                    edges.append(mid / mid_norm)
    
    edges_unit = np.array(edges)
    
    # LJ equilibrium distance
    r_eq = 2.0 ** (1.0/6.0)  # ~1.122
    
    # Radii: inner shell ~1.0*r_eq, outer shell ~2.0*r_eq
    r_in = 1.05 * r_eq
    r_out = 2.05 * r_eq
    
    # Build positions
    pos = np.array([[0.0, 0.0, 0.0]])
    pos = np.vstack([pos, verts_unit * r_in])       # 12 inner
    pos = np.vstack([pos, verts_unit * r_out])       # 12 outer vertices
    pos = np.vstack([pos, edges_unit * r_out])       # 30 outer edges
    
    return pos[:N]

def _radial_manifold_optimize(pos0, deadline):
    """Optimize radial distances on the icosahedral direction manifold."""
    # Extract directions from initial positions (normalize)
    dirs = pos0.copy()
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    dirs = dirs / norms
    
    # Center atom stays at origin
    dirs[0] = np.array([0.0, 0.0, 0.0])
    
    def radial_to_pos(radii):
        """Convert radii to positions."""
        pos = dirs * radii[:, np.newaxis]
        pos[0] = [0.0, 0.0, 0.0]
        return pos
    
    def radial_energy_and_grad(radii):
        """Energy and gradient w.r.t. radii."""
        pos = radial_to_pos(radii)
        E, grad_pos = _lj_energy_and_grad(pos)
        # grad_r[i] = dot(grad_pos[i], dirs[i]) for i>0
        grad_r = np.zeros(N)
        grad_r[1:] = np.sum(grad_pos[1:] * dirs[1:], axis=1)
        return E, grad_r
    
    # Initial radii
    r0 = np.linalg.norm(pos0, axis=1)
    r0[0] = 0.0
    
    try:
        res = minimize(
            radial_energy_and_grad,
            r0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if res.fun < radial_energy_and_grad(r0)[0]:
            pos_opt = radial_to_pos(res.x)
            return pos_opt
    except Exception:
        pass
    
    return pos0

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin for 150s limit
    
    # Step 1: Construct Mackay icosahedron
    pos = _get_mackay_icosahedron()
    
    if time.time() > deadline - 30:
        return pos
    
    # Step 2: Radial manifold optimization (constrained to icosahedral directions)
    pos = _radial_manifold_optimize(pos, deadline)
    
    if time.time() > deadline - 20:
        return pos
    
    # Step 3: Full 3D L-BFGS relaxation
    best_pos = pos.copy()
    best_E, _ = _lj_energy_and_grad(pos)
    
    try:
        res = minimize(
            _lj_energy_and_grad,
            pos.ravel(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if res.fun < best_E:
            best_pos = res.x.reshape(N, 3)
            best_E = res.fun
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        return best_pos
    
    # Step 4: High-precision polish pass
    try:
        res2 = minimize(
            _lj_energy_and_grad,
            best_pos.ravel(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
        )
        if res2.fun < best_E:
            best_pos = res2.x.reshape(N, 3)
            best_E = res2.fun
    except Exception:
        pass
    
    return best_pos
