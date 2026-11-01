"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses the known Mackay icosahedron geometry with optimized shell radii,
applies diverse perturbations (Gaussian, radial, angular), and relaxes
with L-BFGS to converge to the global minimum energy of ~-279.248.
"""

import numpy as np
from scipy.optimize import minimize

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

def _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10):
    """Relax positions using L-BFGS-B with analytical gradient."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()
        
    result = minimize(
        energy_func,
        pos.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return result.x.reshape(-1, 3)

def construct():
    """
    Construct the 55-atom LJ cluster by perturbing the Mackay icosahedron
    and relaxing with L-BFGS.
    """
    rng = np.random.default_rng(42)
    
    # Use standard Mackay icosahedron radii
    # These are close to the optimal values for the global minimum
    r1, r2, r3 = 1.135, 1.162, 1.283
    base = _mackay_icosahedron_55(r1, r2, r3)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    # Helper to update best
    def update_best(pos):
        nonlocal best_energy, best_pos
        if not _is_valid(pos):
            return
        e = _lj_energy(pos)
        if e < best_energy:
            best_energy = e
            best_pos = pos.copy()
            
    # 1. Relax the exact Mackay geometry (very good start)
    relaxed = _relax(base, maxiter=20000, ftol=1e-14, gtol=1e-12)
    update_best(relaxed)
    
    # 2. Multi-start with Gaussian noise
    # We use a range of noise scales to explore the energy landscape
    # Small noise refines, large noise escapes local minima
    noise_scales = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    
    for scale in noise_scales:
        # Multiple trials per scale to improve robustness
        for _ in range(3):
            noise = rng.normal(0, scale, size=base.shape)
            pos = base + noise
            # Relax with moderate iteration count
            relaxed = _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10)
            update_best(relaxed)
            
    # 3. Radial perturbations
    # Perturb the shell radii to find better packing
    for _ in range(20):
        r1_p = r1 * (1 + rng.normal(0, 0.05))
        r2_p = r2 * (1 + rng.normal(0, 0.05))
        r3_p = r3 * (1 + rng.normal(0, 0.05))
        pos = _mackay_icosahedron_55(r1_p, r2_p, r3_p)
        # Add small noise to break symmetry
        noise = rng.normal(0, 0.01, size=pos.shape)
        pos = pos + noise
        relaxed = _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10)
        update_best(relaxed)
        
    # 4. Angular perturbations
    for _ in range(10):
        pos = base.copy()
        for i in range(1, len(pos)):
            angle = rng.normal(0, 0.05)
            axis = rng.normal(size=3)
            axis = axis / (np.linalg.norm(axis) + 1e-10)
            c = np.cos(angle)
            s = np.sin(angle)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = c * np.eye(3) + s * K + (1 - c) * np.outer(axis, axis)
            pos[i] = R @ pos[i]
        relaxed = _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10)
        update_best(relaxed)
        
    # 5. Final high-precision refinement of the best candidate
    if best_energy < float('inf'):
        refined = _relax(best_pos, maxiter=50000, ftol=1e-15, gtol=1e-13)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined

    return best_pos
