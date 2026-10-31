"""Rewritten program using direct FCC lattice construction with local relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time

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

def _minimize_lbfgs(pos, max_iter=1000, tol=1e-10):
    """L-BFGS minimization using scipy."""
    from scipy.optimize import minimize
    res = minimize(
        _lj_energy,
        pos,
        jac=_lj_grad,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
    )
    return res.x

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    # The Lennard-Jones minimum distance is 2^(1/6)
    r_min = 2.0 ** (1.0 / 6.0)
    
    # Construct FCC lattice fragment
    # FCC lattice: points (i,j,k) where i+j+k is even
    # Nearest neighbor distance in this lattice is sqrt(2) * scale
    # We want nearest neighbor distance = r_min
    # So scale = r_min / sqrt(2)
    scale = r_min / np.sqrt(2.0)
    max_coord = 6
    
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
    
    # Take the first N points
    cluster = pts[sorted_indices[:N]].copy()
    
    # Relax the cluster
    try:
        relaxed = _minimize_lbfgs(cluster, max_iter=500, tol=1e-8)
        energy_relaxed = _lj_energy(relaxed)
        energy_initial = _lj_energy(cluster)
        
        if energy_relaxed < energy_initial:
            cluster = relaxed
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return cluster
    
    # Try perturbations to escape local minima
    best_energy = _lj_energy(cluster)
    best_pos = cluster.copy()
    
    rng = np.random.RandomState(42)
    
    for attempt in range(30):
        if time.time() > deadline - 10.0:
            break
        
        # Small random perturbation
        perturbed = best_pos + rng.normal(0, 0.01, size=best_pos.shape)
        
        try:
            relaxed = _minimize_lbfgs(perturbed, max_iter=300, tol=1e-8)
            energy = _lj_energy(relaxed)
            
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed
        except Exception:
            continue
    
    return best_pos
# EVOLVE-BLOCK-END
