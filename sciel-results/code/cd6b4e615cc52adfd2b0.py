"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses the known Mackay icosahedron geometry with optimized shell radii,
applies diverse perturbations, and relaxes with L-BFGS to converge to the
global minimum energy of ~-279.248.
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

def _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10, deadline=None):
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
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 100}
        )
        return result.x.reshape(-1, 3)
    except Exception:
        return pos

def _polish(pos, deadline=None):
    """
    High-precision polish pass with progressively tighter tolerances.
    """
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    # Multi-stage polish: progressively tighter
    stages = [
        {'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200},
        {'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 500},
        {'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 1000},
    ]
    
    x0 = best_pos.flatten()
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        try:
            result = minimize(
                lambda x: _lj_energy_and_grad(x.reshape(-1, 3)),
                x0,
                jac=True,
                method='L-BFGS-B',
                options=stage_opts
            )
            new_pos = result.x.reshape(-1, 3)
            if _is_valid(new_pos):
                e = _lj_energy(new_pos)
                if e < best_energy:
                    best_energy = e
                    best_pos = new_pos
                    x0 = result.x
        except Exception:
            pass
    
    return best_pos

def _quick_radii_search(r1_range, r2_range, r3_range, n_samples=5):
    """
    Quick grid search over shell radii to find good starting points.
    Returns list of (r1, r2, r3, energy) sorted by energy.
    """
    candidates = []
    r1_vals = np.linspace(r1_range[0], r1_range[1], n_samples)
    r2_vals = np.linspace(r2_range[0], r2_range[1], n_samples)
    r3_vals = np.linspace(r3_range[0], r3_range[1], n_samples)
    
    for r1 in r1_vals:
        for r2 in r2_vals:
            for r3 in r3_vals:
                if r1 <= 0 or r2 <= 0 or r3 <= 0:
                    continue
                pos = _mackay_icosahedron_55(r1, r2, r3)
                if _is_valid(pos):
                    e = _lj_energy(pos)
                    candidates.append((r1, r2, r3, e))
    
    candidates.sort(key=lambda x: x[3])
    return candidates

def construct():
    """
    Construct the 55-atom LJ cluster by perturbing the Mackay icosahedron
    and relaxing with L-BFGS.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # 1. Quick radii search to find good starting geometries
    radii_candidates = _quick_radii_search(
        (1.10, 1.18), (1.14, 1.22), (1.25, 1.33), n_samples=6
    )
    
    # Take top 3 radii sets
    top_radii = radii_candidates[:3]
    
    best_energy = float('inf')
    best_pos = None
    
    # 2. Relax the best radii geometry first
    for r1, r2, r3, e0 in top_radii:
        if time.time() > deadline - 60.0:
            break
        base = _mackay_icosahedron_55(r1, r2, r3)
        if not _is_valid(base):
            continue
        relaxed = _relax(base, maxiter=5000, deadline=deadline)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
    
    # If no valid result from radii search, fall back to default
    if best_pos is None:
        base = _mackay_icosahedron_55(1.14, 1.18, 1.29)
        relaxed = _relax(base, maxiter=5000, deadline=deadline)
        if _is_valid(relaxed):
            best_energy = _lj_energy(relaxed)
            best_pos = relaxed.copy()
        else:
            best_pos = base.copy()
            best_energy = _lj_energy(base)
    
    # Use the best relaxed position as the base for perturbations
    base = best_pos.copy()
    
    # 3. Multi-start with structured and Gaussian perturbations
    # Use smaller noise scales for more trials since we have a good base
    noise_scales = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    
    for scale in noise_scales:
        if time.time() > deadline - 25.0:
            break
        n_trials = 5 if scale <= 0.05 else 3
        for trial in range(n_trials):
            if time.time() > deadline - 20.0:
                break
            
            if trial % 2 == 0:
                # Gaussian noise
                noise = rng.normal(0, scale, size=base.shape)
            else:
                # Structured: shell-specific noise
                noise = np.zeros_like(base)
                # Center atom: small noise
                noise[0] = rng.normal(0, scale * 0.5, size=3)
                # First shell (12 atoms): moderate noise
                noise[1:13] = rng.normal(0, scale, size=(12, 3))
                # Second shell (30 atoms): moderate noise
                noise[13:43] = rng.normal(0, scale, size=(30, 3))
                # Third shell (12 atoms): larger noise
                noise[43:55] = rng.normal(0, scale * 1.5, size=(12, 3))
            
            pos = base + noise
            
            if not _is_valid(pos):
                continue
                
            relaxed = _relax(pos, maxiter=3000, deadline=deadline)
            
            if not _is_valid(relaxed):
                continue
                
            energy = _lj_energy(relaxed)
            
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
                base = relaxed.copy()  # Update base for subsequent perturbations

    # 4. Final high-precision polish
    if best_energy < float('inf') and time.time() < deadline - 5.0:
        refined = _polish(best_pos, deadline=deadline)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined
    
    return best_pos
