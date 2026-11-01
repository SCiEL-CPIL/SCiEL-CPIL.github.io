"""Lennard-Jones 55-atom global minimum: Symmetric shell optimization + L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

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

def _get_mackay_directions():
    """
    Returns the 55 unit direction vectors for the Mackay icosahedron shells.
    Shell 0: Center (1 atom, dir [0,0,0])
    Shell 1: 12 vertices
    Shell 2: 30 edge midpoints
    Shell 3: 12 face centers
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of an icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges (distance between adjacent vertices is 2)
    # In unit icosahedron, edge length is 2 / ||v||
    # Let's just find min distance
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
    
    # 30 edge midpoints
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    # 12 face centers
    # An icosahedron has 20 faces. The dual is a dodecahedron with 12 vertices.
    # The face centers of the icosahedron correspond to the vertices of the dodecahedron.
    # Alternatively, we can find them by grouping vertices into triangles.
    # Simpler: The face center direction is the normalized sum of the 3 vertex vectors of the face.
    # We need to find the 20 faces.
    # A face is a set of 3 vertices where all pairwise distances are min_dist.
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                for k in range(j + 1, 12):
                    if abs(dists[i, k] - min_dist) < 1e-6 and abs(dists[j, k] - min_dist) < 1e-6:
                        faces.append((i, j, k))
    
    face_centers = []
    for i, j, k in faces:
        fc = verts_unit[i] + verts_unit[j] + verts_unit[k]
        norm = np.linalg.norm(fc)
        if norm > 1e-6:
            face_centers.append(fc / norm)
    face_centers = np.array(face_centers)
    
    # Construct full set of 55 directions
    # Center
    center = np.array([[0.0, 0.0, 0.0]])
    
    return center, verts_unit, edge_mids, face_centers

def _build_symmetric_positions(radii):
    """
    Build 55 positions from 4 shell radii.
    radii = [r_center, r_shell1, r_shell2, r_shell3]
    """
    center, s1, s2, s3 = _get_mackay_directions()
    
    r0, r1, r2, r3 = radii
    
    pos0 = center * r0
    pos1 = s1 * r1
    pos2 = s2 * r2
    pos3 = s3 * r3
    
    return np.vstack([pos0, pos1, pos2, pos3])

def _symmetric_energy(radii):
    """Energy for symmetric configuration."""
    pos = _build_symmetric_positions(radii)
    return _lj_energy(pos)

def _relax_lbfgs(pos, deadline, maxiter=5000, gtol=1e-14):
    """Relax using L-BFGS-B with deadline guard."""
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        # Check time before starting
        if time.time() > deadline - 5:
            return best_pos, best_energy
            
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster using symmetric shell optimization + full relaxation."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Optimize the 4 shell radii
    # Initial guess based on known Mackay geometry
    # r1 ~ 1.05, r2 ~ 1.75, r3 ~ 1.95 (approx)
    # Center is 0.
    x0 = np.array([0.0, 1.05, 1.75, 1.95])
    
    # Use Nelder-Mead for global optimization of the 4D space
    # This is fast and robust for such low dimensions
    try:
        result_nm = minimize(
            _symmetric_energy, x0, method='Nelder-Mead',
            options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000}
        )
        radii_opt = result_nm.x
    except Exception:
        radii_opt = x0
    
    # 2. Generate positions from optimal radii
    pos_sym = _build_symmetric_positions(radii_opt)
    
    # 3. Full 3D L-BFGS relaxation from symmetric structure
    # This handles any slight symmetry breaking and converges to the local minimum
    if time.time() < deadline - 20.0:
        pos_relaxed, energy_relaxed = _relax_lbfgs(pos_sym, deadline, maxiter=1000, gtol=1e-12)
        if energy_relaxed < _lj_energy(pos_sym):
            pos_sym = pos_relaxed
    
    # 4. High-precision polish
    if time.time() < deadline - 5.0:
        pos_polished, energy_polished = _relax_lbfgs(pos_sym, deadline, maxiter=5000, gtol=1e-14)
        if energy_polished < _lj_energy(pos_sym):
            pos_sym = pos_polished
    
    # 5. Final min-distance check
    diff = pos_sym[:, np.newaxis, :] - pos_sym[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    if min_dist_found < 0.5:
        # Scale up to satisfy constraint
        scale = 0.5 / min_dist_found * 1.001
        pos_sym = pos_sym * scale
    
    return pos_sym

# EVOLVE-BLOCK-END
