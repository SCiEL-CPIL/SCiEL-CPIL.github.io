"""Rewritten program using direct FCC lattice construction with local relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
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

def _minimize_lbfgs(pos, max_iter=1000, tol=1e-10):
    """L-BFGS minimization using scipy."""
    res = minimize(
        _lj_energy,
        pos,
        jac=_lj_grad,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
    )
    return res.x

def _make_fcc_positions(scale, aniso_x=1.0, aniso_y=1.0, aniso_z=1.0, max_coord=6):
    """
    Generate FCC lattice points with anisotropic scaling.
    Returns array of shape (M, 3).
    """
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append((i * aniso_x, j * aniso_y, k * aniso_z))
    
    pts = np.array(points, dtype=float)
    pts *= scale
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    r_min = 2.0 ** (1.0 / 6.0)
    base_scale = r_min / np.sqrt(2.0)
    
    best_energy = float('inf')
    best_pos = None
    
    # 1. Grid search over shape parameters (anisotropy) and scale
    # The truncated octahedron is close to cubic, but surface relaxation might favor slight distortions.
    # We search a small grid around the ideal FCC parameters.
    
    scales = np.linspace(0.95 * base_scale, 1.05 * base_scale, 5)
    anisotropies = np.linspace(0.9, 1.1, 5)
    
    for s in scales:
        for ax in anisotropies:
            for ay in anisotropies:
                # Keep volume roughly constant or just explore shape
                # Let's fix az to 1.0 for simplicity in this first pass, or vary it slightly
                az = 1.0
                try:
                    if time.time() > deadline - 30.0:
                        break
                    
                    pts = _make_fcc_positions(s, ax, ay, az, max_coord=5)
                    # Center the cluster
                    center = np.mean(pts, axis=0)
                    pts -= center
                    
                    # Sort by distance to origin and take N
                    dists = np.sum(pts**2, axis=1)
                    sorted_indices = np.argsort(dists)
                    cluster = pts[sorted_indices[:N]].copy()
                    
                    # Quick relaxation
                    relaxed = _minimize_lbfgs(cluster, max_iter=200, tol=1e-6)
                    energy = _lj_energy(relaxed)
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed
                except Exception:
                    continue
            if time.time() > deadline - 30.0:
                break
        if time.time() > deadline - 30.0:
            break
    
    if best_pos is None:
        # Fallback to original logic
        s = base_scale
        pts = _make_fcc_positions(s, 1.0, 1.0, 1.0, max_coord=6)
        pts -= np.mean(pts, axis=0)
        dists = np.sum(pts**2, axis=1)
        sorted_indices = np.argsort(dists)
        best_pos = pts[sorted_indices[:N]].copy()
        best_energy = _lj_energy(best_pos)
    
    # 2. Refine the best candidate with longer relaxation and perturbations
    current_pos = best_pos
    current_energy = best_energy
    
    # High precision polish
    try:
        polished = _minimize_lbfgs(current_pos, max_iter=2000, tol=1e-12)
        e_polished = _lj_energy(polished)
        if e_polished < current_energy:
            current_energy = e_polished
            current_pos = polished
    except Exception:
        pass
    
    # 3. Local search with perturbations
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 5.0:
        # Random perturbation
        # Scale of perturbation: try a few different magnitudes
        sigma = rng.uniform(0.01, 0.1)
        perturbed = current_pos + rng.normal(0, sigma, size=current_pos.shape)
        
        try:
            relaxed = _minimize_lbfgs(perturbed, max_iter=500, tol=1e-8)
            energy = _lj_energy(relaxed)
            
            if energy < current_energy:
                current_energy = energy
                current_pos = relaxed
        except Exception:
            continue
    
    return current_pos
# EVOLVE-BLOCK-END
