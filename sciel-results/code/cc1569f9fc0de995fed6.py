"""Lennard-Jones cluster global minimum, 55 atoms.

Strategy:
- Start from a cubic lattice seed (as required).
- Use a dual-temperature simulated annealing schedule embedded within L-BFGS:
  1. High-temperature stochastic perturbation phase: add random noise to escape local minima.
  2. Gradually reduce temperature while doing L-BFGS relaxations.
  3. Final zero-temperature deterministic polish with tight tolerance.
- Use multiple restarts with different random seeds, keeping the best result.
- Enforce minimum distance constraint (0.5) via projection after each step.
- Deadline guard: track elapsed time and return best result with safety margin.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 55
MIN_DIST = 0.5
DEADLINE = 115.0  # safety margin from 120s limit


def lj_energy(pos):
    """Compute total Lennard-Jones energy for all pairs."""
    n = len(pos)
    if n < 2:
        return 0.0
    # Compute all pair distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    # Mask self-interactions and near-zero distances
    mask = ~np.eye(n, dtype=bool)
    dist_sq = np.where(mask, dist_sq, np.inf)
    dist_sq = np.maximum(dist_sq, 1e-12)
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    # Subtract double-counted self terms (none since masked)
    return energy


def lj_gradient(pos):
    """Compute gradient of total Lennard-Jones energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    mask = ~np.eye(n, dtype=bool)
    dist_sq = np.where(mask, dist_sq, np.inf)
    dist_sq = np.maximum(dist_sq, 1e-12)
    
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # dE/dr = 4 * (-12 * r^-13 + 6 * r^-7) = 4 * (-12 * r^-12/r + 6 * r^-6/r)
    # dE/dx_i = sum_j 4 * (-12 * r_ij^-12 + 6 * r_ij^-6) * (x_i - x_j) / r_ij
    # = sum_j 4 * (-12 * r_ij^-13 + 6 * r_ij^-7) * (x_i - x_j)
    # Using inv_r2: r^-13 = r^-12 * r^-1 = inv_r12 / sqrt(dist_sq)
    # Better: factor = 4 * (-12 * inv_r12 + 6 * inv_r6) / sqrt(dist_sq)
    
    sqrt_dist = np.sqrt(dist_sq)
    sqrt_dist = np.where(sqrt_dist > 1e-12, sqrt_dist, 1.0)
    
    # factor_ij = 4 * (-12 * r^-12 + 6 * r^-6) / r = 4 * (-12 * inv_r12 + 6 * inv_r6) / r
    factor = 4.0 * (-12.0 * inv_r12 + 6.0 * inv_r6) / sqrt_dist
    factor = np.where(mask, factor, 0.0)
    
    # Gradient: grad_i = sum_j factor_ij * (pos_i - pos_j)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    return grad


def enforce_min_dist(pos, min_dist=MIN_DIST):
    """Project positions to satisfy minimum distance constraint."""
    for _ in range(20):  # iterative projection
        changed = False
        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[j] - pos[i]
                dist = np.linalg.norm(d)
                if dist < min_dist and dist > 1e-12:
                    # Push apart
                    correction = (min_dist - dist) / (2.0 * dist) * d
                    pos[i] -= correction
                    pos[j] += correction
                    changed = True
                elif dist <= 1e-12:
                    # Randomly separate
                    rand_dir = np.random.randn(3)
                    rand_dir /= np.linalg.norm(rand_dir)
                    pos[i] -= 0.25 * rand_dir
                    pos[j] += 0.25 * rand_dir
                    changed = True
        if not changed:
            break
    return pos


def dual_temp_anneal(start_pos, rng, time_budget):
    """
    Dual-temperature simulated annealing within L-BFGS framework.
    
    Phase 1: High temperature - large stochastic perturbations + light L-BFGS
    Phase 2: Medium temperature - smaller perturbations + moderate L-BFGS  
    Phase 3: Low temperature - tiny perturbations + tight L-BFGS
    Phase 4: Zero temperature - pure L-BFGS polish
    """
    pos = start_pos.copy()
    current_energy = lj_energy(pos)
    best_pos = pos.copy()
    best_energy = current_energy
    
    # Temperature schedule
    t_start = time.time()
    
    # Define phases: (fraction_of_time, temp_scale, lbfgs_tol, lbfgs_maxiter)
    phases = [
        (0.35, 0.8, 1e-3, 30),   # High temp: big moves
        (0.25, 0.4, 1e-4, 50),   # Medium temp
        (0.20, 0.15, 1e-5, 80),  # Low temp
        (0.20, 0.0, 1e-7, 200),  # Zero temp: polish
    ]
    
    for frac, temp_scale, tol, maxiter in phases:
        phase_end = t_start + frac * time_budget
        while time.time() < phase_end:
            # Add stochastic perturbation
            if temp_scale > 0:
                noise = rng.normal(0, temp_scale, size=pos.shape)
                pos_new = pos + noise
                pos_new = enforce_min_dist(pos_new)
                
                # Evaluate energy of perturbed position
                e_new = lj_energy(pos_new)
                
                # Metropolis acceptance
                if e_new < current_energy:
                    pos = pos_new
                    current_energy = e_new
                else:
                    # Accept with probability exp(-(e_new - current_energy)/T)
                    # Use effective temperature
                    T_eff = temp_scale * 5.0  # scale temperature
                    if T_eff > 1e-10:
                        delta_e = e_new - current_energy
                        if delta_e < 100.0:  # avoid overflow
                            accept_prob = np.exp(-delta_e / max(T_eff, 1e-10))
                            if rng.random() < accept_prob:
                                pos = pos_new
                                current_energy = e_new
            
            # L-BFGS relaxation
            try:
                result = minimize(
                    lj_energy,
                    pos,
                    jac=lj_gradient,
                    method='L-BFGS-B',
                    options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol}
                )
                if result.success or result.fun < current_energy:
                    pos = result.x
                    current_energy = result.fun
            except Exception:
                pass
            
            # Enforce constraints
            pos = enforce_min_dist(pos)
            current_energy = lj_energy(pos)
            
            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = pos.copy()
            
            # Check global deadline
            if time.time() > DEADLINE:
                break
        
        if time.time() > DEADLINE:
            break
    
    return best_pos, best_energy


def construct():
    """Construct 55-atom Lennard-Jones cluster configuration."""
    t0 = time.time()
    
    # Generate seed: cubic lattice fragment
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    seed_pos = np.array(points[:N], dtype=float) * 1.12
    
    # Ensure minimum distance for seed
    seed_pos = enforce_min_dist(seed_pos)
    
    best_pos = seed_pos.copy()
    best_energy = lj_energy(best_pos)
    
    # Time budget for search
    elapsed = time.time() - t0
    remaining = DEADLINE - elapsed - 5.0  # leave 5s for final polish
    if remaining < 10.0:
        return best_pos
    
    # Multiple restarts with different random seeds
    num_restarts = max(1, int(remaining / 8.0))  # ~8s per restart
    num_restarts = min(num_restarts, 12)  # cap restarts
    
    for restart in range(num_restarts):
        if time.time() > DEADLINE - 5.0:
            break
        
        # Create perturbed start for this restart
        rng = np.random.RandomState(42 + restart)
        
        if restart == 0:
            # First restart: use seed directly
            start = seed_pos.copy()
        else:
            # Subsequent restarts: perturb the seed
            noise_scale = 0.5 + 0.3 * restart
            start = seed_pos + rng.normal(0, noise_scale, size=seed_pos.shape)
            start = enforce_min_dist(start)
        
        # Allocate time budget for this restart
        restart_budget = min(remaining / (num_restarts - restart), 25.0)
        
        try:
            pos, energy = dual_temp_anneal(start, rng, restart_budget)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        except Exception:
            pass
        
        # Update remaining time
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 5.0
        if remaining < 5.0:
            break
    
    # Final high-precision polish on best configuration
    if time.time() < DEADLINE - 3.0:
        try:
            result = minimize(
                lj_energy,
                best_pos,
                jac=lj_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            if result.fun < best_energy:
                best_pos = result.x
                best_energy = result.fun
        except Exception:
            pass
    
    # Final constraint enforcement
    best_pos = enforce_min_dist(best_pos)
    
    return best_pos
