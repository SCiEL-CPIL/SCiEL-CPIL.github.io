"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + L-BFGS relaxation."""
import numpy as np
import time

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0  # each pair counted twice

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 3)
    r2 = np.sum(diff ** 2, axis=2)  # (n, n)
    np.fill_diagonal(r2, 1.0)  # avoid div by zero
    r = np.sqrt(r2)
    r6 = r2 ** 3
    r12 = r6 * r6
    
    # Energy: 4*(r^-12 - r^-6)
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    # Gradient: dE/dpos[i] = sum_j 24*(2/r^13 - 1/r^7) * (pos[i] - pos[j])
    # factor_ij = 24*(2/r^13 - 1/r^7) = 24*(2/r^14 * r - 1/r^7) ... 
    # Actually: dE/dr = 4*(-12*r^-13 + 6*r^-7)
    # grad_i = dE/dr * (pos[i]-pos[j])/r = 4*(-12/r^14 + 6/r^8) * (pos[i]-pos[j])
    # Let's use: factor = 4*(-12/r^14 + 6/r^8) = 24*(-2/r^14 + 1/r^8)
    # r^14 = r2^7, r^8 = r2^4
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    # grad[i] = sum_j factor[i,j] * (pos[i] - pos[j])
    # = sum_j factor[i,j] * pos[i] - sum_j factor[i,j] * pos[j]
    # = pos[i] * sum_j factor[i,j] - (factor * pos)[sum over j]
    factor_pos = factor[:, :, np.newaxis] * pos[np.newaxis, :, :]
    grad = -np.sum(factor_pos, axis=1) + pos * np.sum(factor, axis=1, keepdims=True)
    
    return energy, grad

def _make_mackay_icosahedron():
    """Construct a two-shell Mackay icosahedron for 55 atoms.
    1 center + 12 first shell + 30 edge-midpoints + 12 vertex-like = 55
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges of icosahedron
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
    
    # 30 edge midpoints (unit vectors)
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    # Radii: tune for LJ minimum ~1.122
    r1 = 1.05       # first shell
    r2 = 1.75       # edge midpoints (second shell inner)
    r3 = 1.95       # vertex-like (second shell outer)
    
    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    second_inner = edge_mids * r2
    second_outer = verts_unit * r3
    
    all_pos = np.vstack([center, first_shell, second_inner, second_outer])
    return all_pos

def _relax_lbfgs(pos, deadline=None):
    """Relax using L-BFGS-B with deadline guard."""
    from scipy.optimize import minimize
    
    start_time = time.time()
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
            options={'maxiter': 500, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.success or result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Check deadline
    if deadline is not None and time.time() > deadline - 5.0:
        return best_pos, best_energy
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Generate multiple Mackay variants with slightly different radii
    candidates = []
    
    # Base Mackay
    base = _make_mackay_icosahedron()
    candidates.append(base)
    
    # Perturbed radii variants
    for r1, r2, r3 in [(1.0, 1.7, 1.9), (1.1, 1.8, 2.0), (1.05, 1.72, 1.92),
                        (1.08, 1.78, 1.96), (1.02, 1.68, 1.88)]:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(verts, axis=1, keepdims=True)
        verts_unit = verts / norms
        
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
        edge_mids = []
        for i, j in edges:
            mid = (verts_unit[i] + verts_unit[j]) / 2.0
            norm = np.linalg.norm(mid)
            if norm > 1e-6:
                edge_mids.append(mid / norm)
        edge_mids = np.array(edge_mids)
        
        center = np.array([[0.0, 0.0, 0.0]])
        first_shell = verts_unit * r1
        second_inner = edge_mids * r2
        second_outer = verts_unit * r3
        candidates.append(np.vstack([center, first_shell, second_inner, second_outer]))
    
    # Add a few random perturbations of the base Mackay
    np.random.seed(42)
    for _ in range(3):
        perturbed = base + np.random.randn(55, 3) * 0.05
        candidates.append(perturbed)
    
    # Relax each candidate, keep the best
    best_energy = float('inf')
    best_pos = base.copy()
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 15.0:
            break
        relaxed, energy = _relax_lbfgs(cand, deadline=deadline)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed
    
    # Final check: ensure minimum distance >= 0.5
    # If any pair is too close, nudge them apart
    min_dist_found = float('inf')
    for i in range(55):
        for j in range(i+1, 55):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < min_dist_found:
                min_dist_found = d
    
    if min_dist_found < 0.5:
        # Simple fix: scale up slightly
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
    
    return best_pos
# EVOLVE-BLOCK-END
