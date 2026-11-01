"""Lennard-Jones 55-atom cluster global minimum search.

Uses the Mackay icosahedron geometry with optimized shell radii,
applies diverse perturbations, and relaxes with L-BFGS to converge
to the global minimum energy of ~-279.248.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55
_DEADLINE = 138.0
_START_TIME = None

def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _time_remaining():
    return max(0.0, _DEADLINE - _elapsed())

def _get_icosahedron_vectors():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    dists = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists[dists > 1e-8])
    
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.isclose(dists[i, j], min_dist, atol=1e-8):
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                edge_mids.append(mid)
    
    return verts, np.array(edge_mids)

def _mackay_icosahedron_55(r1=1.135, r2=1.162, r3=1.283):
    v5, v2 = _get_icosahedron_vectors()
    points = [np.zeros(3)]
    for v in v5:
        points.append(v * r1)
    for v in v2:
        points.append(v * r2)
    for v in v5:
        points.append(v * r3)
    return np.array(points, dtype=float)

def _lj_energy(pos):
    """Compute LJ energy using efficient pair computation."""
    n = len(pos)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    return 4.0 * np.sum(r12 - r6)

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient efficiently."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    energy = 4.0 * (np.sum(r12 - r6) - 4.0 * (1.0 - 1.0))  # diagonal is r2=1, contributes 0
    
    # Actually let's be more careful
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    
    r6_masked = np.where(mask, r6, 0.0)
    r12_masked = np.where(mask, r12, 0.0)
    energy = 4.0 * np.sum(r12_masked - r6_masked)
    
    # Gradient: for each atom i, grad_i = sum_{j != i} 4 * (-12/r^14 + 6/r^8) * (pos_i - pos_j)
    # = 4 * sum_j (-12 * r2^(-7) + 6 * r2^(-4)) * diff_ij
    r2_inv4 = np.where(mask, r2 ** (-4), 0.0)
    r2_inv7 = np.where(mask, r2 ** (-7), 0.0)
    coeff = 4.0 * (-12.0 * r2_inv7 + 6.0 * r2_inv4)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    
    return energy, grad

def _is_valid(pos):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    return np.min(r2) >= 0.25

def _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10):
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()
    
    result = minimize(
        energy_func,
        pos.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return result.x.reshape(-1, 3)

def _optimize_radii(r1_init, r2_init, r3_init):
    v5, v2 = _get_icosahedron_vectors()
    
    def objective(radii):
        r1, r2, r3 = radii
        points = [np.zeros(3)]
        for v in v5:
            points.append(v * r1)
        for v in v2:
            points.append(v * r2)
        for v in v5:
            points.append(v * r3)
        pos = np.array(points)
        return _lj_energy(pos)
    
    result = minimize(
        objective,
        np.array([r1_init, r2_init, r3_init]),
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
    )
    return result.x

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    rng = np.random.default_rng(42)
    v5, v2 = _get_icosahedron_vectors()
    
    # Step 1: Optimize radii
    r1_init, r2_init, r3_init = 1.14, 1.18, 1.29
    optimal_radii = _optimize_radii(r1_init, r2_init, r3_init)
    r1_opt, r2_opt, r3_opt = optimal_radii
    
    base = _mackay_icosahedron_55(r1_opt, r2_opt, r3_opt)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    # Step 2: Relax base geometry
    relaxed = _relax(base, maxiter=15000, ftol=1e-14, gtol=1e-12)
    if _is_valid(relaxed):
        e = _lj_energy(relaxed)
        best_energy = e
        best_pos = relaxed.copy()
    
    # Step 3: Multi-start with diverse perturbations
    perturbations = []
    
    # Radial perturbations
    for sf1 in [0.95, 0.98, 1.0, 1.02, 1.05]:
        for sf2 in [0.95, 0.98, 1.0, 1.02, 1.05]:
            for sf3 in [0.95, 0.98, 1.0, 1.02, 1.05]:
                perturbations.append(('radial', sf1, sf2, sf3))
    
    # Gaussian noise perturbations
    for scale in [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]:
        for trial in range(3):
            perturbations.append(('gaussian', scale, trial))
    
    # Angular perturbations (rotate shells slightly)
    for angle in [0.01, 0.02, 0.05]:
        for axis in [0, 1, 2]:
            perturbations.append(('angular', angle, axis))
    
    # Shuffle perturbations for diversity
    rng.shuffle(perturbations)
    
    for pert in perturbations:
        if _time_remaining() < 3.0:
            break
        
        pos = base.copy()
        
        if pert[0] == 'radial':
            _, sf1, sf2, sf3 = pert
            for i in range(1, 13):
                pos[i] *= sf1
            for i in range(13, 43):
                pos[i] *= sf2
            for i in range(43, 55):
                pos[i] *= sf3
        elif pert[0] == 'gaussian':
            _, scale, _ = pert
            noise = rng.normal(0, scale, size=base.shape)
            pos = base + noise
        elif pert[0] == 'angular':
            _, angle, axis = pert
            # Small rotation of outer shell
            c, s = np.cos(angle), np.sin(angle)
            if axis == 0:
                R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            elif axis == 1:
                R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else:
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            for i in range(43, 55):
                pos[i] = R @ pos[i]
        
        if not _is_valid(pos):
            continue
        
        relaxed = _relax(pos, maxiter=8000, ftol=1e-13, gtol=1e-11)
        
        if not _is_valid(relaxed):
            continue
        
        energy = _lj_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    # Step 4: Final high-precision polish
    if _time_remaining() > 10.0:
        polished = _relax(best_pos, maxiter=20000, ftol=1e-15, gtol=1e-13)
        if _is_valid(polished):
            e = _lj_energy(polished)
            if e < best_energy:
                best_energy = e
                best_pos = polished.copy()
    
    if _time_remaining() > 5.0:
        polished = _relax(best_pos, maxiter=30000, ftol=1e-16, gtol=1e-14)
        if _is_valid(polished):
            e = _lj_energy(polished)
            if e < best_energy:
                best_energy = e
                best_pos = polished.copy()
    
    return best_pos
