# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.spatial.distance import pdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _generate_random_compact(n, rng, min_dist=0.6):
    """Generate a random compact cluster with minimum distance constraints."""
    points = []
    # Estimate radius based on density
    # Volume ~ n * (min_dist/2)^3 * packing_factor
    # R ~ (n)^(1/3) * scale
    radius = 0.8 * n ** (1.0 / 3.0) + 1.0
    max_attempts = 5000 * n
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            attempts += 1
            continue
        if not points:
            points.append(p)
        else:
            # Vectorized distance check for speed
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) >= min_dist:
                points.append(p)
            else:
                attempts += 1
    if len(points) < n:
        # Fallback: simple cubic grid if packing fails (should be rare)
        points = []
        side = int(np.ceil(n ** (1/3))) + 1
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    if len(points) < n:
                        points.append(np.array([i, j, k]) * 0.7)
        points = np.array(points[:n])
        points -= points.mean(axis=0)
    else:
        points = np.array(points)
        points -= points.mean(axis=0)
    return points


def _basin_hopping_worker(initial_pos, kind, rng_seed, time_limit, max_iter=100):
    """
    Perform basin hopping from initial_pos within time_limit.
    Uses k-atom perturbation for structural diversity.
    """
    start_time = time.time()
    current_energy, current_pos = relax(initial_pos, 'sutton_chen', kind)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    rng = np.random.default_rng(rng_seed)
    n = current_pos.shape[0]
    
    # Adaptive step size
    step_size = 0.3
    # k-atom perturbation: number of atoms to move
    k_atoms = max(2, int(0.1 * n)) 
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb: Move k random atoms
        perturbed = current_pos.copy()
        if k_atoms < n:
            indices = rng.choice(n, size=k_atoms, replace=False)
            perturbed[indices] += rng.normal(0, step_size, (k_atoms, 3))
        else:
            perturbed += rng.normal(0, step_size, current_pos.shape)
        
        # Relax
        new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            step_size *= 0.95
        else:
            delta_e = new_energy - current_energy
            # Simple cooling schedule
            temp = 0.5 * np.exp(-i / 20.0) + 0.01
            if rng.random() < np.exp(-delta_e / temp):
                current_energy = new_energy
                current_pos = new_pos
                step_size *= 1.05
            else:
                step_size *= 0.95
        
        # Track best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
            
        # Check minimum distance
        if n > 1:
            min_dist = pdist(current_pos).min()
            if min_dist < 0.35:
                # Push apart or restart locally
                current_pos += rng.normal(0, 0.1, current_pos.shape)
                current_energy, current_pos = relax(current_pos, 'sutton_chen', kind)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """
    Parallel multi-start basin-hopping search for Sutton-Chen cluster global minima.
    """
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    n = instance["n"]
    kind = instance["kind"]
    
    # Generate diverse initial structures
    rng_master = np.random.default_rng(42)
    
    # We will run 4 parallel workers.
    # Each worker will run basin hopping from a random dense packing.
    # To increase diversity, we can also mix in some geometric motifs if time allows,
    # but random dense packing is often sufficient and more robust.
    
    initial_structures = []
    for i in range(4):
        initial_structures.append(_generate_random_compact(n, rng_master, min_dist=0.6))
    
    results = []
    
    # Use ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, start_pos in enumerate(initial_structures):
            worker_time_limit = (deadline - time.time()) / 4 - 5
            if worker_time_limit < 10:
                worker_time_limit = 10
            seed = 100 + i
            futures.append(executor.submit(_basin_hopping_worker, start_pos, kind, seed, worker_time_limit, max_iter=200))
        
        for future in futures:
            try:
                energy, pos = future.result()
                if np.isfinite(energy):
                    results.append((energy, pos))
            except Exception as e:
                pass
    
    if not results:
        return _generate_random_compact(n, rng_master)
    
    # Find the best result
    best_energy, best_pos = min(results, key=lambda x: x[0])
    
    # Final check for validity
    if n > 1:
        min_dist = pdist(best_pos).min()
        if min_dist < 0.35:
            # Try to fix by small perturbations and relax
            rng_fix = np.random.default_rng(123)
            for _ in range(10):
                best_pos += rng_fix.normal(0, 0.05, best_pos.shape)
                min_dist = pdist(best_pos).min()
                if min_dist >= 0.35:
                    break
            try:
                best_energy, best_pos = relax(best_pos, 'sutton_chen', kind)
            except:
                pass
    
    return best_pos
# EVOLVE-BLOCK-END
