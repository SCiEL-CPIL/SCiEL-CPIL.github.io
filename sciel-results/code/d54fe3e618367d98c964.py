"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses the known Mackay icosahedron geometry with optimized shell radii,
applies diverse perturbations (Gaussian, radial, angular), and relaxes
with L-BFGS to converge to the global minimum energy of ~-279.248.
Uses differential evolution population-based search to escape local minima.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55
_DEADLINE = 145.0  # Safety margin before 150s kill
_START_TIME = None

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

def _check_deadline():
    """Check if we've exceeded the deadline."""
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE



def construct():
    """
    Construct the 55-atom LJ cluster using differential evolution population
    search on the Mackay icosahedron family, with L-BFGS relaxation.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    rng = np.random.default_rng(42)
    
    # Use standard Mackay icosahedron radii
    base = _mackay_icosahedron_55(1.135, 1.162, 1.283)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    # Strategy 1: Relax the base geometry directly
    relaxed = _relax(base, maxiter=5000, ftol=1e-12, gtol=1e-10)
    if _is_valid(relaxed):
        e = _lj_energy(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
    
    # Strategy 2: Differential evolution population-based search
    # Each individual is a 55-atom structure; fitness = energy after L-BFGS relaxation
    pop_size = 12
    max_generations = 20
    F = 0.8  # Mutation scale
    CR = 0.9  # Crossover rate
    
    # Initialize population with diverse perturbations of the Mackay icosahedron
    population = []
    energies = []
    
    # First individual: the base (already relaxed)
    population.append(best_pos.copy())
    energies.append(best_energy)
    
    # Add diverse initial individuals
    noise_scales_init = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    for i, scale in enumerate(noise_scales_init):
        if _check_deadline():
            break
        noise = rng.normal(0, scale, size=base.shape)
        pos = base + noise
        relaxed = _relax(pos, maxiter=3000, ftol=1e-10, gtol=1e-8)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            population.append(relaxed.copy())
            energies.append(e)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
    
    # Ensure population is at least pop_size
    while len(population) < pop_size and not _check_deadline():
        noise = rng.normal(0, 0.1, size=base.shape)
        pos = base + noise
        relaxed = _relax(pos, maxiter=3000, ftol=1e-10, gtol=1e-8)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            population.append(relaxed.copy())
            energies.append(e)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
    
    # Differential evolution loop
    for gen in range(max_generations):
        if _check_deadline():
            break
        
        # Sort by energy (best first)
        idx_sorted = np.argsort(energies)
        population = [population[i] for i in idx_sorted]
        energies = [energies[i] for i in idx_sorted]
        
        # Early exit if very close to theoretical minimum
        if best_energy < -279.248:
            break
        
        # Generate offspring
        new_population = []
        new_energies = []
        
        for i in range(len(population)):
            if _check_deadline():
                break
            
            # Mutation: DE/rand/1
            a, b, c = rng.choice(len(population), size=3, replace=False)
            mutant = population[a] + F * (population[b] - population[c])
            
            # Crossover: binomial
            trial = np.zeros_like(mutant)
            j_rand = rng.integers(0, N)
            for j in range(N):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = population[i][j]
            
            # Evaluate trial
            relaxed = _relax(trial, maxiter=3000, ftol=1e-10, gtol=1e-8)
            if not _is_valid(relaxed):
                # Keep parent if trial is invalid
                new_population.append(population[i])
                new_energies.append(energies[i])
                continue
            
            e_trial = _lj_energy(relaxed)
            
            # Selection: keep better of trial and parent
            if e_trial < energies[i]:
                new_population.append(relaxed)
                new_energies.append(e_trial)
                if e_trial < best_energy:
                    best_energy = e_trial
                    best_pos = relaxed.copy()
            else:
                new_population.append(population[i])
                new_energies.append(energies[i])
        
        # Update population
        if new_population:
            population = new_population
            energies = new_energies
    
    # Final fine-tuning pass with extremely tight tolerances
    if best_energy < float('inf') and not _check_deadline():
        refined = _relax(best_pos, maxiter=10000, ftol=1e-14, gtol=1e-12)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined
    
    return best_pos
