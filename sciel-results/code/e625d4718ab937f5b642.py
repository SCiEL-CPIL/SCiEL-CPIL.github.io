# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 55
DEADLINE = 145.0  # Safety margin before 150s kill

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
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def _get_mackay_directions():
    """
    Get the 55 unit direction vectors for a Mackay icosahedron M(2,2,2).
    Returns:
        directions: (55, 3) array of unit vectors (center is zero)
        shell_ids: (55,) array indicating which shell each atom belongs to
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

def _relax(positions, tol=1e-6, maxiter=300):
    """Relax positions using L-BFGS-B with analytical gradient."""
    def objective(x):
        pos = x.reshape(N, 3)
        e, g = _lj_energy_grad(pos)
        return e, g.flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'gtol': tol, 'maxiter': maxiter, 'maxls': 50, 'ftol': 1e-10})
        if result.success or result.fun < 1e6:
            return result.x.reshape(N, 3), result.fun
    except Exception:
        pass
    return positions, _lj_energy(positions)

def _rotate_shell(pos, shell_indices, angle, axis):
    """Rotate a specific set of atoms in pos by angle around axis."""
    c = np.cos(angle)
    s_ = np.sin(angle)
    K = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    R = c * np.eye(3) + s_ * K + (1 - c) * np.outer(axis, axis)
    new_pos = pos.copy()
    new_pos[shell_indices] = pos[shell_indices] @ R.T
    return new_pos

def _optimize_radii(radii_init, start_time, deadline):
    """
    Optimize shell radii using Nelder-Mead, with L-BFGS relaxation at each step.
    Returns best position, energy, and radii.
    """
    def objective(radii):
        if time.time() > deadline - 5.0:
            return 1e10
        radii_clipped = np.maximum(radii, 0.5)
        pos = _make_mackay_from_radii(radii_clipped)
        pos_relaxed, e = _relax(pos, tol=1e-5, maxiter=150)
        return e
    
    try:
        res = minimize(objective, radii_init, method='Nelder-Mead', 
                       options={'maxiter': 40, 'xatol': 1e-4, 'fatol': 1e-4})
        best_radii = np.maximum(res.x, 0.5)
        best_pos = _make_mackay_from_radii(best_radii)
        best_pos, best_e = _relax(best_pos, tol=1e-6, maxiter=200)
        return best_pos, best_e, best_radii
    except Exception:
        return None, np.inf, radii_init

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def remaining():
        return deadline - time.time()

    directions, shell_ids = _get_mackay_directions()
    
    # Phase 1: Optimize radii using Nelder-Mead from multiple starting points
    radius_candidates = [
        np.array([1.04, 1.76, 1.98]),  # Current best known
        np.array([1.0, 1.8, 2.6]),
        np.array([1.1, 1.9, 2.7]),
        np.array([0.95, 1.75, 2.05]),
    ]
    
    best_energy = np.inf
    best_positions = None
    
    for radii_init in radius_candidates:
        if remaining() < 20:
            break
        try:
            pos, e, radii = _optimize_radii(radii_init, start_time, deadline)
            if pos is not None and e < best_energy:
                best_energy = e
                best_positions = pos.copy()
        except Exception:
            continue
    
    # Fallback if all failed
    if best_positions is None:
        best_positions = _make_mackay_from_radii(np.array([1.04, 1.76, 1.98]))
        best_energy = _lj_energy(best_positions)
        best_positions, best_energy = _relax(best_positions, tol=1e-6, maxiter=200)

    # Phase 2: Shell twist exploration
    shell_1_idx = np.arange(1, 13)
    shell_2_idx = np.arange(13, 43)
    shell_3_idx = np.arange(43, 55)
    
    rng = np.random.RandomState(42)
    n_twists = 12
    for i in range(n_twists):
        if remaining() < 15:
            break
        
        angle_1 = rng.uniform(-0.08, 0.08)
        angle_2 = rng.uniform(-0.08, 0.08)
        angle_3 = rng.uniform(-0.08, 0.08)
        
        axis_1 = rng.randn(3)
        axis_2 = rng.randn(3)
        axis_3 = rng.randn(3)
        
        norm_1 = np.linalg.norm(axis_1)
        norm_2 = np.linalg.norm(axis_2)
        norm_3 = np.linalg.norm(axis_3)
        if norm_1 > 1e-10: axis_1 /= norm_1
        if norm_2 > 1e-10: axis_2 /= norm_2
        if norm_3 > 1e-10: axis_3 /= norm_3
        
        pos_try = best_positions.copy()
        pos_try = _rotate_shell(pos_try, shell_1_idx, angle_1, axis_1)
        pos_try = _rotate_shell(pos_try, shell_2_idx, angle_2, axis_2)
        pos_try = _rotate_shell(pos_try, shell_3_idx, angle_3, axis_3)
        
        pos_relaxed, e_relaxed = _relax(pos_try, maxiter=150, gtol=1e-5)
        
        if e_relaxed < best_energy:
            best_energy = e_relaxed
            best_positions = pos_relaxed.copy()

    # Phase 3: Perturbation-relax cycles
    for i in range(6):
        if remaining() < 10:
            break
        scale = 0.02 * (i + 1)
        perturbed = best_positions + rng.randn(*best_positions.shape) * scale
        relaxed, e = _relax(perturbed, tol=1e-5, maxiter=200)
        if e < best_energy:
            best_energy = e
            best_positions = relaxed.copy()

    # Phase 4: Per-atom radial refinement (break shell symmetry)
    if remaining() > 20:
        try:
            dirs = directions.copy()
            dirs[0] = np.array([1.0, 0.0, 0.0])  # dummy for center
            
            def radial_objective(radial_params):
                if time.time() > deadline - 5.0:
                    return 1e10
                pos = np.zeros((55, 3))
                pos[0] = [0.0, 0.0, 0.0]
                for i in range(1, 55):
                    r = max(radial_params[i-1], 0.5)
                    pos[i] = dirs[i] * r
                # Check min distance
                diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                dists = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) < 0.5:
                    return 1e10
                pos_relaxed, e = _relax(pos, tol=1e-6, maxiter=100)
                return e
            
            # Initialize radial params from current best
            radial_init = np.zeros(54)
            for i in range(1, 55):
                radial_init[i-1] = np.linalg.norm(best_positions[i])
            
            res = minimize(radial_objective, radial_init, method='Nelder-Mead',
                          options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_energy:
                radial_best = np.maximum(res.x, 0.5)
                pos_new = np.zeros((55, 3))
                pos_new[0] = [0.0, 0.0, 0.0]
                for i in range(1, 55):
                    pos_new[i] = dirs[i] * radial_best[i-1]
                pos_new, e_new = _relax(pos_new, tol=1e-8, maxiter=300)
                if e_new < best_energy:
                    best_energy = e_new
                    best_positions = pos_new
        except Exception:
            pass

    # Phase 5: Final high-precision polish
    if remaining() > 5:
        pos_final, e_final = _relax(best_positions, maxiter=500, gtol=1e-8)
        if e_final < best_energy:
            best_energy = e_final
            best_positions = pos_final
    
    if remaining() > 3:
        pos_final, e_final = _relax(best_positions, maxiter=1000, gtol=1e-10)
        if e_final < best_energy:
            best_energy = e_final
            best_positions = pos_final
    
    if remaining() > 1:
        pos_final, e_final = _relax(best_positions, maxiter=2000, gtol=1e-12)
        if e_final < best_energy:
            best_energy = e_final
            best_positions = pos_final

    return best_positions

# EVOLVE-BLOCK-END
