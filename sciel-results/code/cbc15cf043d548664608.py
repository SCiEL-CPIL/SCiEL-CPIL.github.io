"""Rewritten program using robust multi-start FCC lattice construction with local relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    mask = ~np.eye(len(pos), dtype=bool)
    r2 = r2[mask]
    r6 = r2 ** 3
    r12 = r6 ** 2
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6)
    return energy

def _lj_grad(pos):
    """Compute gradient of total Lennard-Jones energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    r2 = np.where(r2 < 1e-12, 1e-12, r2)
    r6 = r2 ** 3
    r12 = r6 ** 2
    coeff = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _minimize_lbfgs(pos, max_iter=2000, tol=1e-10):
    """L-BFGS minimization with tight tolerances."""
    try:
        res = minimize(
            _lj_energy,
            pos,
            jac=_lj_grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
        )
        return res.x
    except Exception:
        return pos

def _generate_fcc_cluster(n_atoms, r_nn=2**(1/6)):
    """Generate an FCC cluster of n_atoms closest to the origin."""
    scale = r_nn / np.sqrt(2.0)
    max_coord = 10  # Ensure we have enough points
    
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append((i, j, k))
    
    pts = np.array(points, dtype=float)
    pts *= scale
    
    # Sort by distance to origin
    dists = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Take the first n_atoms points
    cluster = pts[sorted_indices[:n_atoms]].copy()
    
    # Center the cluster at the origin
    center = np.mean(cluster, axis=0)
    cluster -= center
    
    return cluster

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate the ideal FCC truncated octahedron
    base_cluster = _generate_fcc_cluster(N)
    
    # Initial relaxation of the perfect lattice
    # This ensures we start from a good local minimum
    best_pos = _minimize_lbfgs(base_cluster, max_iter=500, tol=1e-8)
    best_energy = _lj_energy(best_pos)
    
    # 2. Multi-start with perturbations to escape potential shallow local minima
    # or to ensure we converge to the exact global minimum geometry
    rng = np.random.RandomState(42)
    
    # Define a list of perturbation scales
    # Small perturbations to polish, larger ones to escape if stuck
    scales = [0.01, 0.05, 0.1, 0.2]
    
    for scale in scales:
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            
            perturbed = best_pos + rng.normal(0, scale, size=best_pos.shape)
            
            # Relax the perturbed configuration
            relaxed = _minimize_lbfgs(perturbed, max_iter=1000, tol=1e-10)
            energy = _lj_energy(relaxed)
            
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed
    
    # 3. Final high-precision polish
    # Use a very tight tolerance to ensure we are at the bottom of the well
    if time.time() < deadline - 5.0:
        final_pos = _minimize_lbfgs(best_pos, max_iter=5000, tol=1e-12)
        final_energy = _lj_energy(final_pos)
        if final_energy < best_energy:
            best_pos = final_pos
            best_energy = final_energy

    return best_pos
# EVOLVE-BLOCK-END
