import numpy as np
from scipy.optimize import minimize
import time

N = 55
_DEADLINE = 120.0  # seconds; evaluator kills at 150s, leave margin

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for N atoms. Vectorized."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero on diagonal
    r_inv2 = 1.0 / dist_sq
    r_inv6 = r_inv2 ** 3
    r_inv12 = r_inv6 ** 2
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Using upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    e = 4.0 * np.sum(r_inv12[iu] - r_inv6[iu])
    # Gradient: dE/dpos[i] = 4 * sum_{j != i} (-12/r^14 + 6/r^8) * (pos[i] - pos[j])
    # Using the full matrix (both i,j and j,i), then dividing by 2
    # Force on atom i: sum_j 24*(2/r^13 - 1/r^7) * (pos[i]-pos[j])
    # Let's use: grad[i] = sum_{j!=i} 4*(-12*r^-14*(pos[i]-pos[j]) + 6*r^-8*(pos[i]-pos[j]))
    # = sum_{j!=i} 4*(pos[i]-pos[j]) * (-12*r^-14 + 6*r^-8)
    # r^-14 = r^-12 * r^-2 = r_inv12 * r_inv2
    # r^-8 = r_inv6 * r_inv2
    coeff = 4.0 * (-12.0 * r_inv12 * r_inv2 + 6.0 * r_inv6 * r_inv2)
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def _lj_energy_only(pos):
    e, _ = _lj_energy_and_grad(pos)
    return e

def construct():
    """
    Constructs the 55-atom Mackay icosahedron by optimizing the radial 
    coordinates of the inner and outer shells.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2.0
    
    # 12 vertices of a regular icosahedron
    # (0, +/-1, +/-phi), (+/-1, +/-phi, 0), (+/-phi, 0, +/-1)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    
    # Normalize vertices to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edge directions (midpoints of adjacent vertices)
    # Adjacent vertices have dot product cos(theta) where cos(theta) = phi / (2+phi)
    target_dot = phi / (2 + phi)
    edge_dirs = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.abs(np.dot(verts[i], verts[j]) - target_dot) < 1e-6:
                edge_dir = verts[i] + verts[j]
                edge_dir = edge_dir / np.linalg.norm(edge_dir)
                edge_dirs.append(edge_dir)
    
    edge_dirs = np.array(edge_dirs)
    # There should be 30 edges
    assert len(edge_dirs) == 30, f"Expected 30 edges, got {len(edge_dirs)}"

    def positions_from_params(params):
        r1, r2, r3 = params
        # 1 center
        p0 = np.array([0.0, 0.0, 0.0])
        # 12 inner shell (vertices)
        p_inner = verts * r1
        # 12 outer shell vertices
        p_outer_v = verts * r2
        # 30 outer shell edges
        p_outer_e = edge_dirs * r3
        
        points = np.vstack([p0, p_inner, p_outer_v, p_outer_e])
        return points

    def energy(params):
        points = positions_from_params(params)
        # Calculate Lennard-Jones energy
        # 4 * sum(r^-12 - r^-6)
        n = len(points)
        # Vectorized distance calculation
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal (self-interaction)
        mask = ~np.eye(n, dtype=bool)
        dist_sq_masked = dist_sq[mask]
        
        # Avoid division by zero if any points are too close (though they shouldn't be)
        dist_sq_masked = np.maximum(dist_sq_masked, 1e-10)
        
        r_inv6 = dist_sq_masked ** -3.0
        r_inv12 = r_inv6 ** 2
        
        # Sum over unique pairs (upper triangle)
        # The mask includes both (i,j) and (j,i), so we divide by 2
        total_energy = 4.0 * np.sum(r_inv12 - r_inv6) / 2.0
        return total_energy

    # Initial guess for radii
    # r1 ~ 1.12 (min of LJ)
    # r2, r3 ~ 2.0-2.5
    x0 = np.array([1.12, 2.2, 2.1])
    
    # Optimize radii
    t_start = time.time()
    try:
        result = minimize(energy, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        best_params = result.x
    except Exception:
        best_params = x0

    final_points = positions_from_params(best_params)
    
    # Full-coordinate L-BFGS relaxation to break shell symmetry constraint
    elapsed = time.time() - t_start
    remaining = _DEADLINE - 10.0 - elapsed  # 10s safety margin
    if remaining > 2.0:
        try:
            def _full_energy_grad(x_flat):
                pos = x_flat.reshape(N, 3)
                e, g = _lj_energy_and_grad(pos)
                return e, g.ravel()
            
            x0_full = final_points.ravel()
            res_full = minimize(_full_energy_grad, x0_full, jac=True,
                                method='L-BFGS-B',
                                options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            if res_full.fun < _lj_energy_only(final_points):
                final_points = res_full.x.reshape(N, 3)
        except Exception:
            pass

    # Second tighter polish if time permits
    elapsed = time.time() - t_start
    remaining = _DEADLINE - 10.0 - elapsed
    if remaining > 2.0:
        try:
            def _full_energy_grad2(x_flat):
                pos = x_flat.reshape(N, 3)
                e, g = _lj_energy_and_grad(pos)
                return e, g.ravel()
            
            x0_full = final_points.ravel()
            res2 = minimize(_full_energy_grad2, x0_full, jac=True,
                            method='L-BFGS-B',
                            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
            if res2.fun < _lj_energy_only(final_points):
                final_points = res2.x.reshape(N, 3)
        except Exception:
            pass

    # Ensure atoms are at least 0.5 apart
    diff = final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 0.5:
        scale = 0.5 / min_dist
        final_points = final_points * scale * 1.1

    return final_points
