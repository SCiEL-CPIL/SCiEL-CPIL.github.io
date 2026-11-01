import numpy as np
from scipy.optimize import minimize
import time

N = 55

def construct():
    """
    Constructs the 55-atom Mackay icosahedron by optimizing the radial 
    coordinates of the inner and outer shells, then performs a full 
    local relaxation to refine the geometry.
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
    
    # Optimize
    try:
        result = minimize(energy, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        best_params = result.x
    except Exception:
        best_params = x0

    final_points = positions_from_params(best_params)
    
    # Ensure atoms are at least 0.5 apart
    # Check minimum distance
    diff = final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 0.5:
        # This should not happen for the icosahedron, but just in case
        # Fall back to a simple expansion
        scale = 0.5 / min_dist
        final_points = final_points * scale * 1.1 # Add margin

    # Full local relaxation of the 55 atoms
    def full_energy(pos):
        pos = pos.reshape(N, 3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dist_sq_masked = dist_sq[mask]
        dist_sq_masked = np.maximum(dist_sq_masked, 1e-10)
        r_inv6 = dist_sq_masked ** -3.0
        r_inv12 = r_inv6 ** 2
        total_energy = 4.0 * np.sum(r_inv12 - r_inv6) / 2.0
        return total_energy

    # Use L-BFGS-B for the final polish
    # We flatten the array for the optimizer
    try:
        result_relax = minimize(full_energy, final_points.flatten(), method='L-BFGS-B', 
                                 options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if result_relax.success:
            final_points = result_relax.x.reshape(N, 3)
    except Exception:
        pass # Keep the previous points if relaxation fails

    return final_points
