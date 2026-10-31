import numpy as np
from scipy.optimize import minimize
import time

N = 38

def _lj_energy_and_grad(pos):
    """
    Calculate Lennard-Jones energy and gradient for a set of positions.
    pos: (N, 3) array of coordinates.
    Returns: energy, gradient (N, 3)
    """
    # Compute pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist_sq, 1.0)
    
    # Inverse distances
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6) over pairs. 
    # The matrix dist_sq contains each pair twice (i,j and j,i). 
    # So sum over all i!=j is 2 * sum over i<j.
    # The standard formula 4 * sum_{i<j} (r^-12 - r^-6) is equivalent to 
    # 2 * sum_{i!=j} (r^-12 - r^-6).
    # Let's compute sum over all i!=j and multiply by 2.
    # energy_matrix = 4 * (inv_r12 - inv_r6)
    # Total energy = 0.5 * sum(energy_matrix) because of double counting?
    # Standard: E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # Sum_{i!=j} (r_ij^-12 - r_ij^-6) = 2 * Sum_{i<j} ...
    # So E = 2 * Sum_{i!=j} (r_ij^-12 - r_ij^-6)
    
    pair_energy = inv_r12 - inv_r6
    # Zero out diagonal (though it's 0 anyway since 1-1=0)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 2.0 * np.sum(pair_energy)
    
    # Gradient
    # dE/dr_ij = 4 * (-12 r^-13 + 6 r^-7) = 24 * (2 r^-13 - r^-7)
    # dE/dx_i = sum_{j!=i} dE/dr_ij * d r_ij / d x_i
    # r_ij = |r_i - r_j|
    # d r_ij / d x_i = (x_i - x_j) / r_ij
    # So dE/dx_i = sum_{j!=i} [ 24 * (2 r^-13 - r^-7) * (x_i - x_j) / r ]
    # = sum_{j!=i} 24 * (2 r^-14 - r^-8) * (x_i - x_j)
    # Let's use the matrix form.
    # grad_matrix[i, j] = dE / d r_i (contribution from pair i,j)
    # Factor = 24 * (2 * inv_r12 * inv_r2 - inv_r6 * inv_r2) * diff[i,j] ?
    # Let's re-derive carefully.
    # E = 2 * sum_{i!=j} (r^-12 - r^-6)  <-- Wait, E = 4 sum_{i<j} ... = 2 sum_{i!=j} ...
    # dE/dx_k = 2 * sum_{i!=j, (i,j) involves k} d/dx_k (r_ij^-12 - r_ij^-6)
    # If k=i: d/dr (r^-12 - r^-6) * dr/dx_i = (-12 r^-13 + 6 r^-7) * (x_i - x_j)/r
    # = (-12 r^-14 + 6 r^-8) * (x_i - x_j)
    # Multiply by 2: 2 * (-12 r^-14 + 6 r^-8) * (x_i - x_j) = 24 * (-2 r^-14 + r^-8) * (x_i - x_j)
    # Let's check signs.
    # LJ: V = 4(r^-12 - r^-6). F = -dV/dr = 4(12r^-13 - 6r^-7).
    # dV/dx_i = 4(-12r^-13 + 6r^-7) * (x_i-x_j)/r = 4(-12r^-14 + 6r^-8)(x_i-x_j).
    # Total E = 4 sum_{i<j} V_ij.
    # dE/dx_i = sum_{j!=i} dV_ij/dx_i.
    # dV_ij/dx_i = 4(-12r^-14 + 6r^-8)(x_i-x_j).
    # So grad_i = 4 * sum_{j!=i} (-12 inv_r14 + 6 inv_r8) * diff[i,j]
    # where inv_r14 = inv_r2^7, inv_r8 = inv_r2^4.
    
    inv_r8 = inv_r2 ** 4
    inv_r14 = inv_r2 ** 7
    
    # Gradient factor for each pair i,j acting on i
    # grad_factor[i,j] = 4 * (-12 * inv_r14 + 6 * inv_r8)
    grad_factor = 4.0 * (-12.0 * inv_r14 + 6.0 * inv_r8)
    
    # Zero out diagonal
    np.fill_diagonal(grad_factor, 0.0)
    
    # grad[i] = sum_j grad_factor[i,j] * diff[i,j]
    grad = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _make_fcc_cluster():
    """
    Generate an FCC lattice fragment of 38 atoms.
    """
    # FCC basis vectors
    v1 = np.array([1.0, 1.0, 0.0])
    v2 = np.array([1.0, 0.0, 1.0])
    v3 = np.array([0.0, 1.0, 1.0])
    
    points = []
    # Generate a cube of FCC lattice points
    # Range -4 to 4 should be enough to get 38 closest to origin
    for n1 in range(-4, 5):
        for n2 in range(-4, 5):
            for n3 in range(-4, 5):
                p = n1 * v1 + n2 * v2 + n3 * v3
                points.append(p)
                
    points = np.array(points)
    
    # Calculate distances to origin
    dists = np.linalg.norm(points, axis=1)
    
    # Sort by distance
    sorted_indices = np.argsort(dists)
    
    # Take the first 38
    cluster = points[sorted_indices[:N]]
    
    # Center the cluster
    centroid = np.mean(cluster, axis=0)
    cluster = cluster - centroid
    
    # Scale to approximate LJ minimum
    # Nearest neighbor distance in this FCC lattice is sqrt(2) (e.g. (1,1,0) to (0,0,0))
    # LJ min is 2^(1/6) ~ 1.122
    # Scale factor: 1.122 / sqrt(2)
    scale = (2.0 ** (1.0/6.0)) / np.sqrt(2.0)
    cluster = cluster * scale
    
    return cluster

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate initial guess
    pos = _make_fcc_cluster()
    
    # Initial energy
    e0, g0 = _lj_energy_and_grad(pos)
    best_pos = pos.copy()
    best_e = e0
    
    # Perform L-BFGS minimization
    try:
        res = minimize(
            _lj_energy_and_grad, 
            pos, 
            jac=True, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-12}
        )
        
        if res.success or res.fun < best_e:
            best_pos = res.x
            best_e = res.fun
            
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_pos
        
    return best_pos
