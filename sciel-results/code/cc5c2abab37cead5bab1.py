"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses a multi-start stochastic search strategy. Each candidate is initialized
by perturbing the known Mackay icosahedron geometry with Gaussian noise or
radial variations, then relaxed using L-BFGS to converge to the global minimum.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55

def _get_icosahedron_vectors():
    """
    Returns 12 unit vectors along the 5-fold axes (icosahedron vertices)
    and 30 unit vectors along the 2-fold axes (icosahedron edge midpoints).
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [-phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, -1],
    ], dtype=float)
    
    # Normalize to unit length
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 30 edge midpoints
    dists = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists[dists > 1e-8])
    
    edge_mids = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(dists[i,j], min_dist, atol=1e-8):
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                edge_mids.append(mid)
                
    edge_mids = np.array(edge_mids)
    return verts, edge_mids

def _mackay_icosahedron_55(r1=1.135, r2=1.162, r3=1.283):
    """
    Construct the 55-atom two-shell Mackay icosahedron.
    1 center, 12 at 5-fold (r1), 30 at 2-fold (r2), 12 at 5-fold (r3).
    """
    v5, v2 = _get_icosahedron_vectors()
    
    points = [np.zeros(3)]
    
    # Add 12 atoms at 5-fold axes, radius r1
    for v in v5:
        points.append(v * r1)
        
    # Add 30 atoms at 2-fold axes, radius r2
    for v in v2:
        points.append(v * r2)
        
    # Add 12 atoms at 5-fold axes, radius r3
    for v in v5:
        points.append(v * r3)
        
    points = np.array(points, dtype=float)
    return points

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    return energy

def _lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    
    # Gradient: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)/r
    # Simplified: grad = 4 * sum_j (-12*r2^-7 + 6*r2^-4) * diff
    r2_inv4 = r2_safe ** (-4)
    r2_inv7 = r2_safe ** (-7)
    coeff = 4 * (-12 * r2_inv7 + 6 * r2_inv4)
    coeff = np.where(mask, coeff, 0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _is_valid(pos):
    """Check if all inter-atomic distances are >= 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _relax(pos, maxiter=2000, ftol=1e-10, gtol=1e-8):
    """Relax positions using L-BFGS-B with analytical gradient."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()
        
    try:
        result = minimize(
            energy_func,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
        )
        return result.x.reshape(-1, 3)
    except Exception:
        return pos

def construct():
    """
    Construct the 55-atom LJ cluster using multi-start stochastic search
    from the Mackay icosahedron geometry.
    """
    rng = np.random.default_rng(42)
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    best_energy = float('inf')
    best_pos = _mackay_icosahedron_55()
    
    # Generate a list of diverse initial configurations
    initial_configs = []
    
    # 1. Exact Mackay geometry
    initial_configs.append(_mackay_icosahedron_55(1.135, 1.162, 1.283))
    
    # 2. Variations in shell radii
    radii_variants = [
        (1.130, 1.160, 1.280),
        (1.140, 1.165, 1.290),
        (1.125, 1.155, 1.275),
        (1.145, 1.170, 1.295),
        (1.100, 1.150, 1.300),
        (1.150, 1.150, 1.250),
    ]
    for r1, r2, r3 in radii_variants:
        initial_configs.append(_mackay_icosahedron_55(r1, r2, r3))
        
    # 3. Gaussian noise perturbations on the base Mackay geometry
    base_mackay = _mackay_icosahedron_55(1.135, 1.162, 1.283)
    noise_scales = [0.01, 0.02, 0.05, 0.1, 0.2]
    for scale in noise_scales:
        for _ in range(5):  # 5 trials per scale
            noise = rng.normal(0, scale, size=base_mackay.shape)
            initial_configs.append(base_mackay + noise)
            
    # 4. Random rotations of the whole cluster (shouldn't change energy, but good for variety if symmetries break)
    # Actually, LJ is isotropic, so global rotation doesn't change energy. 
    # Instead, let's add random perturbations to specific shells.
    for _ in range(10):
        pos = base_mackay.copy()
        # Perturb inner shell (indices 1-12)
        idx_inner = np.arange(1, 13)
        pos[idx_inner] += rng.normal(0, 0.05, size=(12, 3))
        # Perturb middle shell (indices 13-42)
        idx_mid = np.arange(13, 43)
        pos[idx_mid] += rng.normal(0, 0.05, size=(30, 3))
        # Perturb outer shell (indices 43-54)
        idx_outer = np.arange(43, 55)
        pos[idx_outer] += rng.normal(0, 0.05, size=(12, 3))
        initial_configs.append(pos)

    # Shuffle to ensure diverse sampling order
    rng.shuffle(initial_configs)
    
    # Relax each candidate
    for pos in initial_configs:
        if time.time() - start_time > time_limit:
            break
            
        # Check validity before relaxation to save time
        if not _is_valid(pos):
            continue
            
        relaxed = _relax(pos)
        
        if not _is_valid(relaxed):
            continue
            
        energy = _lj_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
            
            # If we get very close to the known minimum, we can stop early
            # Known minimum is approx -279.248
            if best_energy < -279.24:
                # Do a final high-precision relaxation
                refined = _relax(best_pos, maxiter=5000, ftol=1e-12, gtol=1e-10)
                if _is_valid(refined):
                    e_refined = _lj_energy(refined)
                    if e_refined < best_energy:
                        best_energy = e_refined
                        best_pos = refined
                break

    return best_pos
