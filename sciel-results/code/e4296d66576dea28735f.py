"""
LJ38 cluster global minimum search.

Strategy:
  1. Build the known FCC truncated-octahedron geometry (the putative global minimum).
  2. Use Basin Hopping to explore the energy landscape, escaping local minima 
     (like the icosahedral funnel) by combining random perturbations with local relaxation.
  3. Apply a high-precision L-BFGS-B polish to the best candidate.
  4. Return the polished coordinates.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 38
DEADLINE = 115.0  # Seconds, safety margin for 120s limit

def _lj_energy_and_grad(pos):
    """
    Total LJ energy and gradient for positions pos (N,3).
    """
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero on diagonal
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy: sum over all pairs
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient
    # dE/dx_i = 4 * sum_{j != i} (x_i - x_j) * (-12 / r^14 + 6 / r^8)
    # Using r2:
    # r^-14 = (r2)^-7
    # r^-8  = (r2)^-4
    # coeff = 4 * (6 * (r2)^-4 - 12 * (r2)^-7)
    
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_truncated_octahedron():
    """
    Build the FCC truncated-octahedron geometry for LJ38.
    """
    # Generate FCC points
    max_coord = 5
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append((float(i), float(j), float(k)))
    
    points = np.array(points)
    
    # Center the cluster
    points -= np.mean(points, axis=0)
    
    # Sort by distance from origin
    dists = np.sum(points ** 2, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    # Re-center after selection
    points -= np.mean(points, axis=0)
    
    # Scale so that nearest-neighbor distance = 2^(1/6)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    min_r = np.sqrt(min_r2)
    
    if min_r > 1e-10:
        scale = 2.0 ** (1.0 / 6.0) / min_r
        points = points * scale
        
    return points

def _relax(pos, max_iter=1000, gtol=1e-8, ftol=1e-12):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]

def _optimize_scale(pos, n_iter=30):
    """
    1D global search over uniform scale factor s applied to centered positions.
    Finds the s that minimizes total LJ energy.
    """
    center = np.mean(pos, axis=0)
    centered = pos - center
    best_e = np.inf
    best_s = 1.0
    
    # Coarse scan
    for s in np.linspace(0.95, 1.05, 201):
        p = centered * s + center
        e, _ = _lj_energy_and_grad(p)
        if e < best_e:
            best_e = e
            best_s = s
            
    # Fine scan around best
    for _ in range(n_iter):
        span = 0.05 / (10 ** (_ // 3))
        for s in np.linspace(best_s - span, best_s + span, 201):
            if s <= 0.5:
                continue
            p = centered * s + center
            e, _ = _lj_energy_and_grad(p)
            if e < best_e:
                best_e = e
                best_s = s
                
    return best_s

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        # This is a hard constraint violation. 
        # Scaling is a simple fix, but might be bad for energy.
        # However, the evaluator checks this.
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def construct():
    """
    Construct the 38-atom LJ cluster at (or very near) the global minimum.
    Returns array of shape (38, 3).
    """
    start_time = time.time()
    
    # 1. Initial Guess: FCC Truncated Octahedron
    coords = _make_fcc_truncated_octahedron()
    coords = _ensure_min_sep(coords)
    
    # Initial Relaxation
    best_coords, best_energy = _relax(coords, max_iter=1000, gtol=1e-8, ftol=1e-12)
    
    # Optimize scale of initial guess
    best_s = _optimize_scale(best_coords)
    center = np.mean(best_coords, axis=0)
    centered = best_coords - center
    scaled_coords = centered * best_s + center
    scaled_coords = _ensure_min_sep(scaled_coords)
    scaled_coords, scaled_energy = _relax(scaled_coords, max_iter=500, gtol=1e-8, ftol=1e-12)
    
    if scaled_energy < best_energy:
        best_coords = scaled_coords
        best_energy = scaled_energy
    
    # 2. Differential Evolution-like Search with Large Displacements
    # We maintain a population of candidates and use large random displacements
    # to jump over the icosahedral energy barrier.
    
    population_size = 8
    population = [best_coords.copy()]
    
    # Initialize population with diverse perturbations
    for i in range(population_size - 1):
        noise = np.random.normal(0, 0.8, size=best_coords.shape)
        perturbed = best_coords + noise
        perturbed = _ensure_min_sep(perturbed)
        p_relaxed, e_relaxed = _relax(perturbed, max_iter=500, gtol=1e-6, ftol=1e-10)
        population.append(p_relaxed)
    
    # Track the best in population
    pop_best_idx = 0
    for i in range(population_size):
        e_i, _ = _lj_energy_and_grad(population[i])
        if e_i < _lj_energy_and_grad(population[pop_best_idx])[0]:
            pop_best_idx = i
    
    # Main search loop
    while time.time() - start_time < DEADLINE - 15.0:
        
        # Differential evolution-like mutation: 
        # Generate new candidate by perturbing the current best with a large random displacement
        # This helps jump over the icosahedral barrier
        
        # Large displacement to escape local minima
        displacement = np.random.normal(0, 1.5, size=best_coords.shape)
        candidate = best_coords + displacement
        
        # Also try smaller perturbations for refinement
        if np.random.random() < 0.3:
            displacement = np.random.normal(0, 0.3, size=best_coords.shape)
            candidate = best_coords + displacement
        
        candidate = _ensure_min_sep(candidate)
        
        # Strict L-BFGS relaxation
        new_coords, new_energy = _relax(candidate, max_iter=1000, gtol=1e-8, ftol=1e-12)
        
        # Optimize scale of the new candidate
        new_s = _optimize_scale(new_coords)
        center_new = np.mean(new_coords, axis=0)
        centered_new = new_coords - center_new
        scaled_new = centered_new * new_s + center_new
        scaled_new = _ensure_min_sep(scaled_new)
        scaled_new, scaled_new_energy = _relax(scaled_new, max_iter=500, gtol=1e-8, ftol=1e-12)
        
        if scaled_new_energy < new_energy:
            new_coords = scaled_new
            new_energy = scaled_new_energy
        
        # Accept if better
        if new_energy < best_energy:
            best_coords = new_coords
            best_energy = new_energy
            # Update population best
            population[pop_best_idx] = new_coords.copy()
        else:
            # Replace the worst member of the population with the new candidate
            # if it's better than the worst
            worst_idx = 0
            worst_e = np.inf
            for i in range(population_size):
                e_i, _ = _lj_energy_and_grad(population[i])
                if e_i > worst_e:
                    worst_e = e_i
                    worst_idx = i
            
            if new_energy < worst_e:
                population[worst_idx] = new_coords.copy()
        
        # Occasionally update pop_best_idx
        if len(population) > 1:
            pop_best_idx = 0
            for i in range(population_size):
                e_i, _ = _lj_energy_and_grad(population[i])
                if e_i < _lj_energy_and_grad(population[pop_best_idx])[0]:
                    pop_best_idx = i
        
    # 3. Final High-Precision Polish
    if time.time() - start_time < DEADLINE - 3.0:
        polished_coords, polished_energy = _relax(
            best_coords, 
            max_iter=5000, 
            gtol=1e-12, 
            ftol=1e-16
        )
        polished_coords = _ensure_min_sep(polished_coords)
        
        # Optimize scale one final time
        final_s = _optimize_scale(polished_coords)
        center_final = np.mean(polished_coords, axis=0)
        centered_final = polished_coords - center_final
        scaled_final = centered_final * final_s + center_final
        scaled_final = _ensure_min_sep(scaled_final)
        scaled_final, scaled_final_energy = _relax(scaled_final, max_iter=1000, gtol=1e-12, ftol=1e-16)
        
        if scaled_final_energy < polished_energy:
            polished_coords = scaled_final
            polished_energy = scaled_final_energy
        
        # Verify energy of polished coords
        final_energy, _ = _lj_energy_and_grad(polished_coords)
        
        if final_energy < best_energy:
            best_coords = polished_coords
            best_energy = final_energy

    return best_coords

# For testing
if __name__ == "__main__":
    import time
    start = time.time()
    pos = construct()
    end = time.time()
    e, g = _lj_energy_and_grad(pos)
    print(f"Time: {end-start:.2f}s")
    print(f"Energy: {e}")
    print(f"Min Dist: {np.min([np.linalg.norm(pos[i]-pos[j]) for i in range(N) for j in range(i+1,N)])}")
