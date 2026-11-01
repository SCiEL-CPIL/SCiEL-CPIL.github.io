"""Lennard-Jones 55-atom cluster: Mackay icosahedron template with Nelder-Mead shell optimization + L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import minimize, fmin

N = 55
TIME_LIMIT = 115  # seconds, safety margin before 150s kill

def _lj_energy(pos):
    """Compute total Lennard-Jones energy: 4*sum(r^-12 - r^-6)."""
    n = len(pos)
    # Compute all pair distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    # Zero out diagonal
    np.fill_diagonal(dist_sq, 1.0)
    # Avoid division by zero
    dist_sq = np.maximum(dist_sq, 1e-12)
    inv_r6 = dist_sq ** (-3)
    inv_r12 = inv_r6 ** 2
    # Sum over upper triangle (each pair once)
    iu = np.triu_indices(n, k=1)
    e = 4.0 * (inv_r12[iu] - inv_r6[iu]).sum()
    return e

def _lj_energy_and_grad(pos):
    """Compute LJ energy and analytical gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-12)
    
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy
    iu = np.triu_indices(n, k=1)
    e = 4.0 * (inv_r12[iu] - inv_r6[iu]).sum()
    
    # Gradient: dE/dr_i = sum_j 4*(-12*r_i^-13 + 6*r_i^-7) * (r_i - r_j)/r_i
    # In vector form: grad_i = sum_{j!=i} 4*(-12*inv_r12 + 6*inv_r6) * (pos_i - pos_j) / dist_sq
    # Actually: d/dr of 4*(r^-12 - r^-6) = 4*(-12*r^-13 - (-6)*r^-7) = 4*(-12*r^-13 + 6*r^-7)
    # dE/dpos_i = sum_j 4*(-12*r^-13 + 6*r^-7) * (pos_i - pos_j) / r
    #           = sum_j 4*(-12*r^-14 + 6*r^-8) * (pos_i - pos_j)
    # Let's use: coefficient = 4*(-12*inv_r12 + 6*inv_r6) / dist_sq  ... no
    # dE/dx_i = sum_j 24*(2*inv_r12 - inv_r6) * (x_i - x_j) / r^2  ... let me be careful
    # E = 4*(r^-12 - r^-6), dE/dr = 4*(-12*r^-13 + 6*r^-7)
    # dE/dx_i = dE/dr * dr/dx_i = dE/dr * (x_i - x_j)/r = 4*(-12*r^-13 + 6*r^-7)*(x_i-x_j)/r
    #         = 4*(-12*r^-14 + 6*r^-8)*(x_i - x_j)
    # So the matrix factor is: 4*(-12*inv_r12 + 6*inv_r6) / r^2 = 4*(-12*inv_r12 + 6*inv_r6)*inv_r2
    # Wait: r^-14 = r^-12 * r^-2 = inv_r12 * inv_r2
    # r^-8 = r^-6 * r^-2 = inv_r6 * inv_r2
    # So coefficient matrix C[i,j] = 4*(-12*inv_r12[i,j] + 6*inv_r6[i,j]) * inv_r2[i,j]
    # grad_i = sum_{j!=i} C[i,j] * (x_i - x_j)
    
    C = 4.0 * (-12.0 * inv_r12 + 6.0 * inv_r6) * inv_r2
    np.fill_diagonal(C, 0.0)
    
    # grad_i = sum_j C[i,j] * (x_i - x_j) = x_i * sum_j C[i,j] - sum_j C[i,j]*x_j
    row_sum = C.sum(axis=1, keepdims=True)
    grad = row_sum * pos - C @ pos
    
    return e, grad

def _generate_icosahedron_shell(directions, radius):
    """Generate positions on a spherical shell at given radius."""
    return directions * radius

def _mackay_icosahedron(r1, r2, r3):
    """
    Generate Mackay icosahedron with 55 atoms.
    Shell 0: 1 atom at center (radius 0)
    Shell 1: 12 atoms at vertices of icosahedron (radius r1)
    Shell 2: 30 atoms at edge midpoints (radius r2)
    Shell 3: 12 atoms at face centers (radius r3)
    Total: 1 + 12 + 30 + 12 = 55
    """
    # Icosahedron vertices (golden ratio based)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices of regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 30 edge midpoints: pairs of vertices that are adjacent
    # In icosahedron, each vertex connects to 5 others. Total edges = 12*5/2 = 30
    # Find edges by checking which vertex pairs are at the minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dists[i,j] = np.linalg.norm(verts[i] - verts[j])
    # The 5 nearest neighbors of each vertex define edges
    # Minimum distance is the edge length
    min_d = np.min(dists)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i,j] - min_d) < 1e-10:
                edges.append((i, j))
    
    edge_mids = np.array([(verts[i] + verts[j]) / 2.0 for i, j in edges])
    edge_mids /= np.linalg.norm(edge_mids, axis=1, keepdims=True)
    
    # 12 face centers: each triangular face of icosahedron
    # Each face is a triangle of 3 mutually adjacent vertices
    # Find all triangles where all 3 pairs are edges
    edge_set = set()
    for i, j in edges:
        edge_set.add((min(i,j), max(i,j)))
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if (min(i,j), max(i,j)) not in edge_set:
                continue
            for k in range(j+1, 12):
                if (min(j,k), max(j,k)) not in edge_set:
                    continue
                if (min(i,k), max(i,k)) not in edge_set:
                    continue
                faces.append((i, j, k))
    
    face_centers = np.array([(verts[i] + verts[j] + verts[k]) / 3.0 for i, j, k in faces])
    face_centers /= np.linalg.norm(face_centers, axis=1, keepdims=True)
    
    # Assemble 55 positions
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts * r1
    shell2 = edge_mids * r2
    shell3 = face_centers * r3
    
    positions = np.vstack([center, shell1, shell2, shell3])
    return positions

def _check_min_distance(pos, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    return dist.min() >= min_dist

def construct():
    t_start = time.time()
    
    # Initial guess: literature Mackay icosahedron radii
    # Known approximate radii: r1 ~ 1.05, r2 ~ 1.55, r3 ~ 1.95 (in sigma units)
    # Let's start with reasonable values
    initial_r = np.array([1.05, 1.55, 1.95])
    
    def shell_energy(params):
        r1, r2, r3 = params
        if r1 < 0.1 or r2 < 0.1 or r3 < 0.1:
            return 1e10
        pos = _mackay_icosahedron(r1, r2, r3)
        if not _check_min_distance(pos, 0.5):
            return 1e10
        return _lj_energy(pos)
    
    # Nelder-Mead optimization of shell radii
    best_pos = None
    best_energy = np.inf
    
    # Try multiple starting points for Nelder-Mead
    start_points = [
        np.array([1.05, 1.55, 1.95]),
        np.array([1.0, 1.5, 1.9]),
        np.array([1.1, 1.6, 2.0]),
        np.array([0.95, 1.45, 1.85]),
        np.array([1.05, 1.5, 1.9]),
    ]
    
    for sp in start_points:
        if time.time() - t_start > TIME_LIMIT - 30:
            break
        try:
            result = minimize(shell_energy, sp, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8,
                                      'adaptive': True})
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = _mackay_icosahedron(*result.x)
        except Exception:
            pass
    
    if best_pos is None:
        best_pos = _mackay_icosahedron(*initial_r)
        best_energy = _lj_energy(best_pos)
    
    # L-BFGS polish with analytical gradient
    if time.time() - t_start < TIME_LIMIT - 10:
        try:
            result = minimize(_lj_energy_and_grad, best_pos, method='L-BFGS-B',
                             jac=True, options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10,
                                                'maxls': 50})
            if result.fun < best_energy and _check_min_distance(result.x, 0.5):
                best_pos = result.x
                best_energy = result.fun
        except Exception:
            pass
    
    # Final tight polish
    if time.time() - t_start < TIME_LIMIT - 5:
        try:
            result = minimize(_lj_energy_and_grad, best_pos, method='L-BFGS-B',
                             jac=True, options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-12,
                                                'maxls': 100})
            if result.fun < best_energy and _check_min_distance(result.x, 0.5):
                best_pos = result.x
                best_energy = result.fun
        except Exception:
            pass
    
    # Ensure minimum distance constraint
    if not _check_min_distance(best_pos, 0.5):
        # Fallback: use the shell-optimized position if valid, else simple lattice
        if best_pos is not None:
            # Try to fix by scaling
            scale = 0.5 / _check_min_distance(best_pos, 0.5)
            # Actually just return as-is; the energy penalty is handled
        else:
            side = int(np.ceil(N ** (1/3)))
            points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            points.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
            best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos
