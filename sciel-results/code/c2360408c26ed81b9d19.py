"""Sutton-Chen cluster global minima: structure-based global search with deterministic prototypes."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Wall-clock deadline with safety margin
_DEADLINE = time.time() + 280.0


def _check_deadline():
    return time.time() > _DEADLINE


def _sc_energy(pos, n_exp, m_exp, c):
    """Compute Sutton-Chen energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    dist = np.maximum(dist, 1e-10)
    r_inv_n = np.sum(dist ** (-n_exp), axis=1)
    r_inv_m = np.sum(dist ** (-m_exp), axis=1)
    pair_energy = 0.5 * np.sum(r_inv_n)
    cohesion = c * np.sum(np.sqrt(np.maximum(r_inv_m, 0.0)))
    return pair_energy - cohesion


def _sc_gradient(pos, n_exp, m_exp, c):
    """Compute Sutton-Chen gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(np.maximum(dist_sq, 1e-20))
    # Avoid division by zero on diagonal
    inv_dist = np.zeros_like(dist)
    mask = dist > 1e-10
    inv_dist[mask] = 1.0 / dist[mask]
    
    # r^-n_exp and r^-m_exp
    r_inv_n = inv_dist ** n_exp
    r_inv_m = inv_dist ** m_exp
    
    # Gradient of pair term: d/dr [0.5 * r^-n_exp] = -0.5*n_exp*r^(-n_exp-1)
    # Force on i from j: -dV/dr_ij * (r_i - r_j)/r_ij
    # V_pair = 0.5 * r^-n_exp, dV/dr = -0.5*n_exp*r^(-n_exp-1)
    # grad_i V_pair = -dV/dr * (r_i-r_j)/r = 0.5*n_exp*r^(-n_exp-1) * (r_i-r_j)/r = 0.5*n_exp*r^(-n_exp-2)*(r_i-r_j)
    
    # Cohesion: V_coh = -c*sqrt(r_inv_m_sum)
    # dV_coh/d(r_ij^-m) = -c * 0.5 / sqrt(r_inv_m_sum_i)
    # d(r_ij^-m)/dr_ij = -m * r^(-m-1)
    # dV_coh/dr_ij = -c * 0.5 / sqrt(S_i) * (-m * r^(-m-1)) = c*m*0.5*r^(-m-1)/sqrt(S_i)
    # grad_i V_coh = -dV_coh/dr_ij * (r_i-r_j)/r_ij = -c*m*0.5*r^(-m-2)*(r_i-r_j)/sqrt(S_i)
    
    S_m = np.sum(r_inv_m, axis=1)  # sum over j for each i
    sqrt_S_m = np.sqrt(np.maximum(S_m, 1e-30))
    
    # Pair contribution to gradient
    # For each pair (i,j), contribution to grad_i: 0.5*n_exp*r^(-n_exp-2)*(r_i-r_j)
    # Sum over j
    pair_grad = 0.5 * n_exp * np.sum(r_inv_n * inv_dist[:, :, np.newaxis] * diff, axis=1)
    
    # Cohesion contribution
    # grad_i = sum_j [ -c*m*0.5 * r_ij^(-m-2) * (r_i - r_j) / sqrt(S_i) ]
    # = -c*m*0.5/sqrt(S_i) * sum_j [ r_ij^(-m-2) * (r_i - r_j) ]
    coh_factor = -c * m * 0.5 / sqrt_S_m[:, np.newaxis]
    coh_grad = coh_factor * np.sum(r_inv_m * inv_dist[:, :, np.newaxis] * diff, axis=1)
    
    grad = pair_grad + coh_grad
    energy = _sc_energy(pos, n_exp, m_exp, c)
    return energy, grad


def _generate_fcc_cluster(n, a=1.0):
    """Generate n atoms in FCC-like arrangement."""
    # FCC lattice points in a cubic cell of side a
    # Generate points in FCC structure, take first n
    points = []
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype=float)
    # Generate lattice points
    max_coord = int(np.ceil((n / 4) ** (1/3))) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b * a
                    points.append(p)
    points = np.array(points)
    # Center the cluster
    points -= np.mean(points, axis=0)
    # Sort by distance from center and take n closest
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return points[order[:n]]


def _generate_hcp_cluster(n, a=1.0):
    """Generate n atoms in HCP-like arrangement."""
    points = []
    # HCP: ABAB stacking
    a_hcp = a
    c_hcp = a * np.sqrt(8.0 / 3.0)
    max_coord = int(np.ceil((n / 2) ** (1/3))) + 2
    for layer in range(-max_coord, max_coord + 1):
        z = layer * c_hcp
        if layer % 2 == 0:
            # A layer: hexagonal
            for i in range(-max_coord, max_coord + 1):
                for j in range(-max_coord, max_coord + 1):
                    x = a_hcp * (i + 0.5 * (j % 2))
                    y = a_hcp * np.sqrt(3.0) / 2.0 * j
                    points.append([x, y, z])
        else:
            # B layer: offset
            for i in range(-max_coord, max_coord + 1):
                for j in range(-max_coord, max_coord + 1):
                    x = a_hcp * (i + 0.5 * (j % 2)) + a_hcp * 0.5
                    y = a_hcp * np.sqrt(3.0) / 2.0 * j + a_hcp * np.sqrt(3.0) / 6.0
                    points.append([x, y, z])
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return points[order[:n]]


def _generate_icosahedron(n, radius=1.0):
    """Generate atoms on/within an icosahedral shell structure."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    verts *= radius
    
    if n <= 12:
        # Just take n vertices
        return verts[:n]
    
    # For larger n, create shells
    points = [np.zeros(3)]  # center
    points.extend(list(verts))
    
    # Second shell: octahedron vertices scaled
    octa = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    octa /= np.linalg.norm(octa[0])
    # Scale to fit between first and second icosahedral shell
    octa *= radius * 0.75
    points.extend(list(octa))
    
    # If still need more, add more icosahedral vertices at larger radius
    if len(points) < n:
        verts2 = verts * radius * 1.5
        points.extend(list(verts2))
    
    points = np.array(points[:n])
    return points


def _generate_mackay_icosahedron(n, a=1.0):
    """Generate Mackay icosahedral cluster."""
    # Mackay icosahedra: 13, 55, 147 atoms
    # For 55: center + 12 (first shell) + 42 (second shell)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    points = [np.zeros(3)]
    # First shell: 12 icosahedron vertices
    points.extend([v * a for v in verts])
    
    # Second shell: 42 atoms
    # These are at the midpoints of edges and face centers of the icosahedron
    # Edge midpoints: 30 edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1:  # connected vertices
                edges.append((i, j))
    
    # Add edge midpoints (scaled appropriately)
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid) * a * 1.5
        points.append(mid)
    
    # Face centers: 20 faces
    # Find triangular faces
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if 0.9 < d1 < 1.1 and 0.9 < d2 < 1.1 and 0.9 < d3 < 1.1:
                    face_center = (verts[i] + verts[j] + verts[k]) / 3.0
                    face_center = face_center / np.linalg.norm(face_center) * a * 1.4
                    faces.append(face_center)
    
    points.extend(faces)
    
    # If we need more, add third shell
    if len(points) < n:
        verts3 = verts * a * 2.0
        points.extend([v for v in verts3])
    
    points = np.array(points[:n])
    return points


def _generate_bcc_cluster(n, a=1.0):
    """Generate BCC-like cluster."""
    points = []
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=float)
    max_coord = int(np.ceil((n / 2) ** (1/3))) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b * a
                    points.append(p)
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=0)
    order = np.argsort(dists)
    return points[order[:n]]


def _generate_random_compact(n, rng, spacing=0.72):
    """Generate a random compact cluster."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we didn't get enough, fill with random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _relax_cluster(positions, kind, n_exp, m_exp, c, max_iter=2000):
    """Relax a cluster using L-BFGS-B with our own energy/gradient."""
    def fun_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _sc_energy_gradient(pos, n_exp, m_exp, c)
        return e, g.flatten()
    
    x0 = positions.flatten()
    result = minimize(fun_and_grad, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'gtol': 1e-8})
    if result.success or result.fun < np.inf:
        return result.fun, result.x.reshape(-1, 3)
    # Fallback to the provided relax function
    energy, pos = relax(positions, 'sutton_chen', kind)
    return energy, pos


def _sc_energy_gradient(pos, n_exp, m_exp, c):
    """Combined energy and gradient."""
    e = _sc_energy(pos, n_exp, m_exp, c)
    g = _sc_gradient(pos, n_exp, m_exp, c)[1]
    return e, g


def construct(instance):
    """Structure-based global search for Sutton-Chen cluster minima."""
    size = instance["n"]
    kind = instance["kind"]
    n_exp = instance["n_exp"]
    m_exp = instance["m_exp"]
    c = instance["c"]
    
    # Parameters per kind
    params = {
        '12-6': (12, 6, 144.41),
        '9-6': (9, 6, 39.432),
        '10-8': (10, 8, 34.408)
    }
    if kind in params:
        n_exp, m_exp, c = params[kind]
    
    rng = np.random.default_rng(size)
    
    # Generate candidate structures
    candidates = []
    
    # 1. FCC
    candidates.append(_generate_fcc_cluster(size, a=0.72))
    # 2. HCP
    candidates.append(_generate_hcp_cluster(size, a=0.72))
    # 3. Icosahedral
    candidates.append(_generate_icosahedron(size, radius=0.72 * size ** (1/3) * 0.5))
    # 4. Mackay icosahedral
    candidates.append(_generate_mackay_icosahedron(size, a=0.72))
    # 5. BCC
    candidates.append(_generate_bcc_cluster(size, a=0.72))
    
    # 6-10. Random compact structures
    for _ in range(5):
        candidates.append(_generate_random_compact(size, rng, spacing=0.72))
    
    # Relax each candidate and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, cand in enumerate(candidates):
        if _check_deadline():
            break
        try:
            energy, positions = _relax_cluster(cand, kind, n_exp, m_exp, c, max_iter=1500)
            if np.isfinite(energy) and energy < best_energy:
                # Verify no atoms too close
                if _check_validity(positions):
                    best_energy = energy
                    best_positions = positions
        except Exception:
            continue
    
    # If deadline is close, do a final polish with the provided relax
    if best_positions is not None and not _check_deadline():
        try:
            energy, positions = relax(best_positions, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy and _check_validity(positions):
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback: use the first candidate relaxed
        try:
            energy, positions = relax(candidates[0], 'sutton_chen', kind)
            best_positions = positions
        except Exception:
            best_positions = candidates[0]
    
    return best_positions


def _check_validity(positions):
    """Check that positions are valid (finite, no atoms too close)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 0.35:
                return False
    return True
# EVOLVE-BLOCK-END
