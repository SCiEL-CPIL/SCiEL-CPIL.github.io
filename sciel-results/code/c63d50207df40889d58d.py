"""Lennard-Jones cluster global minimum, 55 atoms."""
import numpy as np
import time
from scipy.optimize import minimize

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
                          options={'gtol': tol, 'maxiter': maxiter, 'maxls': 50, 'ftol': 1e-8})
        if result.success or result.fun < 1e6:
            return result.x.reshape(N, 3), result.fun
    except Exception:
        pass
    return positions, _lj_energy(positions)

def _polish(positions, deadline=None, stages=None):
    """Multi-stage polish with progressively tighter tolerances."""
    if stages is None:
        stages = [
            (1e-4, 200),   # coarse
            (1e-6, 300),   # medium
            (1e-8, 500),   # fine
            (1e-10, 800),  # very fine
        ]
    
    pos = positions.copy()
    for tol, maxiter in stages:
        if deadline is not None and time.time() > deadline - 5:
            break
        pos, e = _relax(pos, tol=tol, maxiter=maxiter, deadline=deadline)
    
    return pos, _lj_energy(pos)

def _optimize_radii(radii_init, maxiter_nm=50, maxiter_lbfgs=150, deadline=None):
    """
    Optimize shell radii using Nelder-Mead, with L-BFGS relaxation at each step.
    """
    def objective(radii):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError()
        radii = np.maximum(radii, 0.5)
        pos = _make_mackay_from_radii(radii)
        pos_relaxed, e = _relax(pos, tol=1e-4, maxiter=maxiter_lbfgs, deadline=deadline)
        return e
    
    try:
        res = minimize(objective, radii_init, method='Nelder-Mead', 
                       options={'maxiter': maxiter_nm, 'xatol': 1e-4, 'fatol': 1e-4})
        best_radii = res.x
        best_pos = _make_mackay_from_radii(best_radii)
        best_pos, best_e = _relax(best_pos, tol=1e-6, maxiter=200, deadline=deadline)
        return best_pos, best_e, best_radii
    except Exception:
        return None, np.inf, radii_init

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
    
    # Get Mackay directions for multi-start
    directions, shell_ids = _get_mackay_directions()
    
    # Initial radius candidates based on LJ nearest-neighbor distance ~1.122
    radius_candidates = [
        np.array([1.0, 1.8, 2.6]),
        np.array([1.1, 1.9, 2.7]),
        np.array([0.9, 1.7, 2.5]),
        np.array([1.0, 2.0, 2.8]),
        np.array([1.05, 1.85, 2.65]),
        np.array([1.0, 1.75, 2.55]),
        np.array([1.02, 1.82, 2.62]),
    ]
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Optimize radii for multiple initial guesses
    for radii_init in radius_candidates:
        if remaining() < 20:
            break
        try:
            pos, e, radii = _optimize_radii(radii_init, maxiter_nm=40, maxiter_lbfgs=120, deadline=start_time + DEADLINE)
            if pos is not None and e < best_energy:
                best_energy = e
                best_positions = pos.copy()
        except Exception:
            continue
    
    # Phase 2: Multi-start stochastic search with varying perturbation scales
    if best_positions is None:
        # Fallback: use a reasonable Mackay geometry
        best_positions = _make_mackay_from_radii(np.array([1.0, 1.8, 2.6]))
        best_energy = _lj_energy(best_positions)
    
    # Perform multiple perturbation-relax cycles to escape local minima
    # Use different scales and seeds to explore different basins
    perturbation_configs = [
        (0.02, 42), (0.05, 43), (0.08, 44), (0.12, 45),
        (0.15, 46), (0.20, 47), (0.25, 48), (0.30, 49),
        (0.35, 50), (0.40, 51), (0.50, 52),
    ]
    
    for scale, seed in perturbation_configs:
        if remaining() < 20:
            break
        try:
            perturbed = _perturb(best_positions, scale=scale, seed=seed)
            relaxed, e = _relax(perturbed, tol=1e-5, maxiter=200, deadline=start_time + DEADLINE)
            if e < best_energy:
                best_energy = e
                best_positions = relaxed.copy()
        except Exception:
            continue
    
    # Phase 2b: Shake-and-relax to escape local minima
    if remaining() > 25:
        rng = np.random.RandomState(777)
        for trial in range(5):
            if remaining() < 15:
                break
            try:
                # Apply random displacements to a subset of atoms
                n_shake = rng.randint(10, 25)
                indices = rng.choice(N, size=n_shake, replace=False)
                shaken = best_positions.copy()
                for idx in indices:
                    shaken[idx] += rng.randn(3) * 0.15
                
                relaxed, e = _relax(shaken, tol=1e-5, maxiter=200, deadline=start_time + DEADLINE)
                if e < best_energy:
                    best_energy = e
                    best_positions = relaxed.copy()
            except Exception:
                continue
    
    # Phase 3: Try to break shell symmetry by perturbing individual atoms
    if remaining() > 20:
        try:
            # Create a configuration where atoms are slightly displaced from their ideal directions
            # This can help escape the symmetric local minimum
            rng = np.random.RandomState(123)
            for trial in range(4):
                if remaining() < 10:
                    break
                
                # Perturb each atom's position slightly, breaking the shell symmetry
                displaced = best_positions.copy()
                for i in range(1, 55):
                    # Small random displacement
                    disp = rng.randn(3) * 0.05
                    displaced[i] += disp
                
                relaxed, e = _relax(displaced, tol=1e-5, maxiter=200, deadline=start_time + DEADLINE)
                if e < best_energy:
                    best_energy = e
                    best_positions = relaxed.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish with multi-stage tightening
    if remaining() > 15:
        try:
            polished, e_polished = _polish(best_positions, deadline=start_time + DEADLINE,
                                           stages=[(1e-4, 200), (1e-6, 300), (1e-8, 500), (1e-10, 800)])
            if e_polished < best_energy:
                best_energy = e_polished
                best_positions = polished
        except Exception:
            pass
    
    # Phase 5: One more attempt with a different starting point
    if remaining() > 20:
        try:
            # Start from a slightly different Mackay geometry
            alt_radii = np.array([1.05, 1.85, 2.65])
            alt_pos = _make_mackay_from_radii(alt_radii)
            # Perturb it
            alt_perturbed = _perturb(alt_pos, scale=0.1, seed=99)
            relaxed_alt, e_alt = _relax(alt_perturbed, tol=1e-6, maxiter=300, deadline=start_time + DEADLINE)
            if e_alt < best_energy:
                best_energy = e_alt
                best_positions = relaxed_alt
        except Exception:
            pass
    
    # Phase 6: Final ultra-precision polish if time permits
    if remaining() > 10:
        try:
            final_polish, e_final = _polish(best_positions, deadline=start_time + DEADLINE,
                                            stages=[(1e-6, 300), (1e-8, 500), (1e-10, 1000)])
            if e_final < best_energy:
                best_energy = e_final
                best_positions = final_polish
        except Exception:
            pass
    
    # Ensure minimum distance constraint
    min_d = _min_distance(best_positions)
    if min_d < 0.5:
        scale = 0.5 / min_d
        best_positions = best_positions * scale
    
    return best_positions
