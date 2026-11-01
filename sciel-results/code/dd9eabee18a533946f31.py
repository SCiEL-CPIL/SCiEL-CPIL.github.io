"""Lennard-Jones 55-atom cluster: Mackay icosahedron + radial manifold optimization + full relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
def _lj_energy(pos):
    """Compute total LJ energy (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero on diagonal
    r6 = r2 ** 3
    r12 = r6 * r6
    # Force magnitude factor: 4*(12/r^13 - 6/r^7) = 4/r^2 * (12/r^12 - 6/r^6)
    # Actually: dE/dr = 4*(-12/r^13 + 6/r^7) = 4*(6/r^7 - 12/r^13)
    # dE/dx_i = sum_j dE/dr_ij * (x_i - x_j)/r_ij
    # = sum_j 4*(6/r^7 - 12/r^13) * (x_i - x_j)/r
    # = sum_j 4*(6/r^8 - 12/r^14) * (x_i - x_j)
    # = sum_j 4*(6/r2^4 - 12/r2^7) * (x_i - x_j)
    # where r2 = r^2
    factor = 4.0 * (6.0 / (r2 ** 4) - 12.0 / (r2 ** 7))
    np.fill_diagonal(factor, 0.0)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    energy = 4.0 * np.sum(1.0 / (r6 * r6) - 1.0 / r6) / 2.0
    return energy, grad

def _make_mackay_icosahedron():
    """Construct Mackay icosahedron with 55 atoms.
    Returns positions (55, 3) and directions (55, 3) for radial optimization."""
    phi = (1 + np.sqrt(5)) / 2
    
    # icosahedron vertices (unit sphere)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edges (30 shortest pairs)
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    flat_dists = dists[~np.eye(12, dtype=bool)].flatten()
    min_dist = np.min(flat_dists)
    
    edge_pairs = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-5:
                edge_pairs.append((i, j))
    
    # Build directions for all 55 atoms
    # Atom 0: center
    # Atoms 1-12: inner shell (icosahedron vertices)
    # Atoms 13-24: outer shell vertices (icosahedron vertices)
    # Atoms 25-54: outer shell edge midpoints
    
    directions = np.zeros((55, 3))
    directions[0] = [0, 0, 0]  # center
    
    # Inner shell directions
    for i in range(12):
        directions[1 + i] = verts[i]
    
    # Outer shell vertex directions
    for i in range(12):
        directions[13 + i] = verts[i]
    
    # Outer shell edge midpoint directions
    for idx, (i, j) in enumerate(edge_pairs):
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        directions[25 + idx] = mid
    
    # Initial radii
    r_in = 1.1225  # ~2^(1/6)
    r_out = 2.245   # ~2 * r_in
    
    radii = np.zeros(55)
    radii[0] = 0.0
    radii[1:13] = r_in
    radii[13:25] = r_out
    radii[25:55] = r_out
    
    positions = directions * radii[:, np.newaxis]
    return positions, directions

def _radial_to_positions(directions, radii):
    """Convert directions and radii to positions."""
    # Center atom stays at origin
    pos = directions * radii[:, np.newaxis]
    pos[0] = [0, 0, 0]
    return pos

def _radial_energy_and_grad(directions, radii):
    """Compute energy and gradient with respect to radii (radial manifold)."""
    pos = _radial_to_positions(directions, radii)
    energy, full_grad = _lj_energy_grad(pos)
    
    # Project gradient onto radial directions
    # dE/dr_i = grad_i . direction_i
    radial_grad = np.sum(full_grad * directions, axis=1)
    radial_grad[0] = 0.0  # center is fixed
    
    return energy, radial_grad

def _optimize_radial_manifold(directions, initial_radii, deadline):
    """Optimize radial distances on the icosahedral manifold."""
    def obj_and_grad(radii):
        if time.time() > deadline:
            return 1e10, np.zeros_like(radii)
        e, g = _radial_energy_and_grad(directions, radii)
        # Penalize atoms that are too close
        pos = _radial_to_positions(directions, radii)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, np.inf)
        min_r2 = np.min(r2)
        if min_r2 < 0.25:  # distance < 0.5
            penalty = 1000.0 * (0.25 - min_r2)
            e += penalty
        return e, g
    
    x0 = initial_radii.copy()
    # Fix center radius to 0
    x0[0] = 0.0
    
    try:
        res = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
        if res.fun < 1e9:
            return res.x
    except Exception:
        pass
    
    return x0

def _relax_full(positions, deadline, max_iter=500):
    """Full 3D relaxation using L-BFGS-B."""
    def obj_and_grad(x):
        if time.time() > deadline:
            return 1e10, np.zeros_like(x)
        pos = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos)
        # Check minimum distance
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, np.inf)
        min_r2 = np.min(r2)
        if min_r2 < 0.25:
            penalty = 1000.0 * (0.25 - min_r2)
            e += penalty
        return e, g.flatten()
    
    x0 = positions.flatten()
    try:
        res = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
        if res.fun < 1e9:
            return res.x.reshape(-1, 3)
    except Exception:
        pass
    
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Construct Mackay icosahedron
    positions, directions = _make_mackay_icosahedron()
    
    # 2. Optimize radial distances on manifold
    if time.time() < deadline - 30:
        radii = _optimize_radial_manifold(directions, np.linalg.norm(positions, axis=1), deadline - 30)
        positions = _radial_to_positions(directions, radii)
    
    # 3. Full 3D relaxation
    if time.time() < deadline - 10:
        positions = _relax_full(positions, deadline - 10, max_iter=300)
    
    # 4. Final high-precision polish
    if time.time() < deadline - 2:
        positions = _relax_full(positions, deadline - 2, max_iter=200)
    
    return positions
# EVOLVE-BLOCK-END
