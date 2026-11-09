"""Improved program for Sutton-Chen cluster global minima search using Genetic Algorithm with Parallel Relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def _center_and_scale(positions):
    """Center the cluster at the origin and scale to unit RMS radius."""
    com = np.mean(positions, axis=0)
    centered = positions - com
    rms = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    if rms > 1e-6:
        scaled = centered / rms
    else:
        scaled = centered
    return scaled

def _initialize_population(n, rng, pop_size):
    """Initialize a population of clusters using dense random packing."""
    pop = []
    # Estimate radius for close packing
    radius = (n / (4.0/3.0 * np.pi * 0.74)) ** (1/3.0) * 1.2
    
    for _ in range(pop_size):
        points = []
        attempts = 0
        max_attempts = n * 500
        min_dist = 0.8
        
        while len(points) < n and attempts < max_attempts:
            u = rng.random(3)
            r = radius * (u[0] ** (1/3.0))
            theta = 2 * np.pi * u[1]
            phi = np.arccos(2 * u[2] - 1)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            p = np.array([x, y, z])
            
            # Check distance to existing points
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
            attempts += 1
        
        # Fill remaining if needed
        while len(points) < n:
            u = rng.random(3)
            r = radius * (u[0] ** (1/3.0))
            theta = 2 * np.pi * u[1]
            phi = np.arccos(2 * u[2] - 1)
            points.append(np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ]))
            
        pop.append(np.array(points))
        
    return pop

def _evaluate_individual(positions, potential, kind):
    """Relax and evaluate an individual. Returns (energy, positions)."""
    try:
        E, x = relax(positions, potential, kind)
        if np.isfinite(E):
            # Quick validity check: min distance
            min_d = np.inf
            for a in range(len(x)):
                for b in range(a+1, len(x)):
                    d = np.linalg.norm(x[a] - x[b])
                    if d < min_d:
                        min_d = d
            if min_d < 0.35:
                return np.inf, positions
            return E, x
        else:
            return np.inf, positions
    except Exception:
        return np.inf, positions

def construct(instance):
    """Genetic Algorithm optimization for Sutton-Chen cluster global minima."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Adaptive parameters based on cluster size
    if size < 50:
        pop_size = 20
        generations = 80
    else:
        pop_size = 12
        generations = 40
        
    start_time = time.time()
    time_limit = 45.0 # Allow ~75s per instance for 6 instances in 300s total
    
    # Initialize population
    population = _initialize_population(size, rng, pop_size)
    
    # Evaluate initial population in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_evaluate_individual, pos, potential, kind) for pos in population]
        results = [f.result() for f in as_completed(futures)]
        
    # Sort by energy
    results.sort(key=lambda x: x[0])
    current_pop = [r[1] for r in results]
    current_energies = [r[0] for r in results]
    
    best_energy = current_energies[0]
    best_positions = current_pop[0]
    
    # GA Loop
    for gen in range(generations):
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        half = pop_size // 2
        parents = current_pop[:half]
        
        # Generate offspring
        offspring = []
        for _ in range(half):
            # Select parents from top half
            idx1 = rng.integers(0, half)
            idx2 = rng.integers(0, half)
            p1 = parents[idx1]
            p2 = parents[idx2]
            
            # Crossover: Arithmetic blend of centered/scaled structures
            p1_c = _center_and_scale(p1)
            p2_c = _center_and_scale(p2)
            
            alpha = rng.random()
            child = alpha * p1_c + (1 - alpha) * p2_c
            
            # Mutation: Gaussian noise with cooling
            kick = 0.5 * (1.0 - gen / generations) + 0.1
            child = child + rng.normal(0, kick, child.shape)
            
            # Center back to origin
            child -= np.mean(child, axis=0)
            
            offspring.append(child)
            
        # Evaluate offspring in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_evaluate_individual, pos, potential, kind) for pos in offspring]
            new_results = [f.result() for f in as_completed(futures)]
            
        # Combine parents and offspring
        # We keep the energy values for parents from the previous step
        combined = []
        for i in range(half):
            combined.append((current_energies[i], current_pop[i]))
        for e, x in new_results:
            combined.append((e, x))
            
        # Sort and keep best pop_size
        combined.sort(key=lambda x: x[0])
        current_pop = [x for e, x in combined[:pop_size]]
        current_energies = [e for e, x in combined[:pop_size]]
        
        # Update best
        if current_energies[0] < best_energy:
            best_energy = current_energies[0]
            best_positions = current_pop[0]
            
    # Fallback if all failed
    if best_positions is None or not np.isfinite(best_energy):
        x0 = _initialize_population(size, rng, 1)[0]
        E, x = _evaluate_individual(x0, potential, kind)
        if np.isfinite(E):
            best_energy = E
            best_positions = x
            
    return best_positions
# EVOLVE-BLOCK-END
