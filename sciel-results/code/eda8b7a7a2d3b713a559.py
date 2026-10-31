"""Construct LJ38 global minimum: FCC truncated octahedron with local relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _generate_fcc_38(scale=1.1225, dist=1.0):
    """
    Generate 38 FCC lattice points forming a truncated octahedron.
    'scale' controls the nearest-neighbor distance (since r_nn = scale in this parametrization).
    'dist' controls the anisotropic distortion along the z-axis.
    """
    # In the standard FCC conventional cell, if we set the lattice constant a such that
    # the nearest neighbor distance is 'scale', then a = scale * sqrt(2).
    a = np.sqrt(2) * scale
    
    # Generate a dense grid of FCC points
    # Basis vectors for the FCC lattice points in the conventional cubic cell:
    # The points are at (i*a + dx, j*a + dy, k*a + dz)
    # where (dx, dy, dz) are in {(0,0,0), (a/2,a/2,0), (a/2,0,a/2), (0,a/2,a/2)}
    
    # We need to cover a range sufficient to find the 38 closest points.
    # The truncated octahedron of 38 atoms extends roughly to the 3rd shell.
    # Max coordinate in units of a is roughly 2 or 3.
    max_coord = 4
    points_list = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # 4 FCC basis positions
                for dx, dy, dz in [(0, 0, 0), (a/2, a/2, 0), (a/2, 0, a/2), (0, a/2, a/2)]:
                    x = i * a + dx
                    y = j * a + dy
                    z = (k * a + dz) * dist
                    points_list.append([x, y, z])
    
    points = np.array(points_list)
    
    # Compute distances from origin
    dists = np.linalg.norm(points, axis=1)
    
    # Sort by distance and take the 38 closest
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    # Center the points (they should already be centered around origin due to symmetry,
    # but just in case)
    points = points - np.mean(points, axis=0)
    
    return points


def _lj_energy_and_grad(positions):
    """Compute total LJ energy and gradient for a set of positions."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    
    r = np.sqrt(r2)
    r6 = r2 ** 3
    r12 = r6 ** 2
    
    # Energy
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6) / 2.0
    
    # Gradient
    # dE/dx_i = sum_j 4 * (-12 * r^-14 + 6 * r^-8) * (x_i - x_j)
    r_inv_8 = 1.0 / r6
    r_inv_14 = r_inv_8 / (r2 * r2) # r^-14 = r^-8 * r^-6
    
    coeff = 4.0 * (-12.0 * r_inv_14 + 6.0 * r_inv_8)
    np.fill_diagonal(coeff, 0.0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def construct():
    """
    Optimize rigid-body parameters (scale, dist) and then relax atom positions.
    """
    
    # 1. Optimize rigid body parameters
    def energy_params(params):
        s, d = params
        pts = _generate_fcc_38(scale=s, dist=d)
        e, _ = _lj_energy_and_grad(pts)
        return e
    
    # Initial guess: 
    # Optimal LJ distance is 2^(1/6) ~ 1.1225. 
    # The 'scale' parameter in our generator corresponds to the nearest neighbor distance.
    # So we search around 1.1225.
    # Dist should be close to 1.0 for a cubic lattice.
    
    best_energy = np.inf
    best_params = [1.1225, 1.0]
    
    # Grid search
    # Scale range: 1.05 to 1.20
    # Dist range: 0.95 to 1.05
    scales = np.linspace(1.05, 1.20, 15)
    dists = np.linspace(0.95, 1.05, 11)
    
    for s in scales:
        for d in dists:
            e = energy_params([s, d])
            if e < best_energy:
                best_energy = e
                best_params = [s, d]
                
    # Refine with Nelder-Mead
    res_params = minimize(energy_params, best_params, method='Nelder-Mead', 
                          options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500, 'adaptive': True})
    
    optimal_scale = res_params.x[0]
    optimal_dist = res_params.x[1]
    
    # 2. Generate the optimized rigid body
    positions = _generate_fcc_38(scale=optimal_scale, dist=optimal_dist)
    
    # 3. Local relaxation using L-BFGS-B
    x0 = positions.flatten()
    
    def objective(x):
        pos = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos)
        return e, g.flatten()
    
    # Use L-BFGS-B for robustness
    res = minimize(objective, x0, method='L-BFGS-B', jac=True, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if res.success:
        positions = res.x.reshape(-1, 3)
    else:
        # If relaxation fails, stick with the rigid body
        pass
        
    # 4. Verify minimum separation >= 0.5
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    
    if min_r < 0.5:
        # Revert to rigid body if relaxation caused overlap
        positions = _generate_fcc_38(scale=optimal_scale, dist=optimal_dist)
        # Check again
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(np.min(r2))
        if min_r < 0.5:
            # Fallback: simple cubic seed scaled
            side = int(np.ceil(N ** (1 / 3)))
            pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            pts.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
            positions = np.array(pts[:N], dtype=float) * 1.12

    return positions
# EVOLVE-BLOCK-END
