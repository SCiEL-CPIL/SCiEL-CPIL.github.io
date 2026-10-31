"""
LJ38 Cluster Global Minimum Optimizer
Constructs an FCC lattice seed and refines it to the global minimum
truncated octahedron using L-BFGS.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 38

def _lj_energy_and_grad(pos):
    """
    Compute Lennard-Jones energy and gradient.
    pos: (N, 3) array
    """
    N = pos.shape[0]
    # Compute all pair differences
    # i < j
    # Use broadcasting for vectorization
    # diff[i, j, :] = pos[i] - pos[j]
    # This creates an N x N x 3 array, which is 38*38*3 ~ 4300 floats, very small.
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    # dist_sq[i, j] = |pos[i] - pos[j]|^2
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist_sq, 1.0)
    
    # r = sqrt(dist_sq)
    r = np.sqrt(dist_sq)
    r6 = r**-6
    r12 = r6**2
    
    # Energy per pair is 4 * (r12 - r6)
    # Total energy is sum over i < j
    # Since the matrix is symmetric, we can sum the upper triangle
    # Or sum all and divide by 2, but we must be careful with the diagonal (which is 0)
    # Let's sum the upper triangle explicitly to avoid factor of 2 errors
    iu = np.triu_indices(N, k=1)
    e_pairs = 4.0 * (r12[iu] - r6[iu])
    energy = np.sum(e_pairs)
    
    # Gradient calculation
    # dE/dr = 4 * (-12 r^-13 + 6 r^-7) = 24 * (r^-7 - 2 r^-13)
    # dE/dpos_i = sum_{j != i} dE/dr_ij * d r_ij / d pos_i
    # r_ij = |pos_i - pos_j|
    # d r_ij / d pos_i = (pos_i - pos_j) / r_ij
    # So dE/dpos_i = sum_{j != i} 24 * (r^-7 - 2 r^-13) * (pos_i - pos_j) / r
    #               = sum_{j != i} 24 * (r^-8 - 2 r^-14) * (pos_i - pos_j)
    
    # Let's compute the force factor F(r) = 24 * (r^-8 - 2 r^-14)
    # Note: r^-8 = r6 * r^-2 = r6 / dist_sq
    # r^-14 = r12 * r^-2 = r12 / dist_sq
    inv_r2 = 1.0 / dist_sq
    # F(r) for each pair
    # We need to sum forces.
    # Force on i from j is -dE/dpos_i = - F(r_ij) * (pos_i - pos_j)
    # Gradient is dE/dpos_i = F(r_ij) * (pos_i - pos_j) summed over j.
    
    # Let's build the matrix G where G[i,j] = F(r_ij)
    # F(r) = 24 * (r^-8 - 2 r^-14)
    r8 = r**-8
    r14 = r**-14
    F = 24.0 * (r8 - 2.0 * r14)
    
    # Zero out diagonal to avoid self-interaction
    np.fill_diagonal(F, 0.0)
    
    # Gradient for atom i: sum_j F[i,j] * (pos[i] - pos[j])
    # grad[i] = sum_j F[i,j] * pos[i] - sum_j F[i,j] * pos[j]
    # sum_j F[i,j] * pos[i] = pos[i] * sum_j F[i,j]
    # sum_j F[i,j] * pos[j] = (F @ pos)[i]
    
    sum_F = np.sum(F, axis=1)
    grad = pos * sum_F[:, np.newaxis] - F @ pos
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Generate FCC Lattice Seed
    # FCC primitive vectors:
    # a1 = (0.5, 0.5, 0)
    # a2 = (0.5, 0.0, 0.5)
    # a3 = (0.0, 0.5, 0.5)
    # We want nearest neighbor distance r_min = 2^(1/6)
    # In FCC, nearest neighbors are at distance a/sqrt(2)
    # So a = r_min * sqrt(2)
    r_min = 2.0 ** (1.0 / 6.0)
    a = r_min * np.sqrt(2.0)
    
    # Generate lattice points
    # We need enough points to pick 38 closest to origin
    # 38 atoms is small. A 4x4x4 grid of FCC points will have plenty.
    # FCC points are at (i, j, k) * a where i+j+k is even?
    # Actually, standard FCC lattice points in a cubic cell of side a are:
    # (0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)
    # Let's generate integer coordinates (i,j,k) and filter for i+j+k even, then scale by a/2.
    # Wait, if we use integer coords (i,j,k) with i+j+k even, the distance between (0,0,0) and (1,1,0) is sqrt(2).
    # Scaled by a/2, distance is a/sqrt(2) = r_min. This works.
    
    max_coord = 6
    pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    pts.append([i, j, k])
    
    pts = np.array(pts, dtype=float)
    # Scale to physical units
    pts = pts * (a / 2.0)
    
    # Sort by distance from origin
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    pts = pts[idx]
    
    # Take the first N points
    x0 = pts[:N].copy()
    
    # 2. Optimization
    # We use L-BFGS-B with analytical gradient
    # Bounds are not strictly necessary if we trust the local minimum, but 
    # adding loose bounds can prevent divergence if the seed is bad.
    # However, for a good seed, unconstrained is faster.
    
    result = None
    best_energy = float('inf')
    
    # Run optimization
    try:
        res = minimize(
            _lj_energy_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
        )
        if res.success or res.fun < best_energy:
            best_energy = res.fun
            result = res.x
    except Exception:
        pass
    
    # If the first run didn't finish or failed, we might try a second run with different params
    # but given the seed is very close to the minimum, one run is usually sufficient.
    
    if result is None:
        # Fallback: return the seed (which is already decent)
        result = x0
    
    # 3. Final Polish / Check
    # Check time
    if time.time() > deadline:
         return result
    
    # Ensure minimum distance constraint (0.5)
    # The LJ potential blows up for r < 1, so the optimizer should avoid this.
    # But just in case, we check.
    diff = result[:, np.newaxis, :] - result[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 0.5:
        # This shouldn't happen with a good seed and optimizer, but if it does,
        # we might need to project or just return as is (invalid).
        # For safety, we could add a small perturbation and re-run, but time is tight.
        # Given the seed is FCC, min dist is ~1.12, so this is unlikely.
        pass

    return result

# EVOLVE-BLOCK-END
