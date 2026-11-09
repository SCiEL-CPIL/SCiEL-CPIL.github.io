"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax

# Constants for Dzugutov potential
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _generate_icosahedron(n):
    """Generate a cluster starting with a central icosahedron and adding atoms."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit length
    verts /= np.linalg.norm(verts[0])
    
    points = [np.zeros(3)] # Center
    points.extend(verts.tolist()) # 12 vertices, total 13
    
    if n <= 13:
        return np.array(points[:n])

    # Greedy expansion for remaining atoms
    current_points = np.array(points)
    min_dist = 0.8 # Slightly larger than 0.5 to allow relax
    
    # Estimate radius for shell
    R1 = 1.0
    R2 = np.sqrt(2)
    
    rng = np.random.default_rng(42 + n)
    
    while len(current_points) < n:
        # Generate a batch of random candidates in a shell
        batch_size = 100
        r_min = R2 * 0.9
        r_max = R2 * 1.5
        batch = []
        for _ in range(batch_size):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            r = rng.uniform(r_min, r_max)
            p = v * r
            batch.append(p)
        
        # Try to add as many as possible
        added = 0
        for p in batch:
            if len(current_points) >= n:
                break
            # Check distance against all current points
            dists = np.linalg.norm(current_points - p, axis=1)
            if np.min(dists) >= min_dist:
                current_points = np.vstack([current_points, [p]])
                added += 1
                
        if added == 0:
            # If stuck, expand search radius
            r_min = r_max
            r_max *= 1.1
            if r_max > 10:
                # Fallback: just add random points in a large sphere
                for _ in range(n - len(current_points)):
                    v = rng.normal(size=3)
                    v /= np.linalg.norm(v)
                    r = rng.uniform(0, 5)
                    p = v * r
                    current_points = np.vstack([current_points, [p]])
                break

    return current_points[:n]

def _generate_fcc(n):
    """Generate a truncated FCC cluster."""
    rng = np.random.default_rng(123 + n)
    points = []
    a = 1.0
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]
    
    # Generate points in a grid
    N = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            for k in range(-N, N+1):
                for b in basis:
                    p = np.array([i, j, k]) + np.array(b)
                    candidates.append(p)
    
    candidates = np.array(candidates)
    # Sort by distance from center
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)
    candidates = candidates[indices]
    
    # Select closest n points
    points = candidates[:n]
    
    # Center it
    centroid = np.mean(points, axis=0)
    points -= centroid
    
    return points

def _generate_random_packing(n, seed):
    """Generate a random compact packing."""
    rng = np.random.default_rng(seed)
    points = []
    radius = 1.1 * n ** (1.0 / 3.0)
    min_dist = 0.8
    attempts = 0
    max_attempts = n * 100
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
        
    # If we didn't get enough, just fill with random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def construct(instance):
    """
    Parallel relaxation of diverse structural templates.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    # Safety margin: 330s limit, return by 320s to be safe
    deadline = start_time + 320.0 
    
    # 1. Generate initial candidates
    candidates = []
    
    # Icosahedral templates (best guess for Dzugutov)
    try:
        p_ico = _generate_icosahedron(size)
        candidates.append(p_ico)
        # Slightly deformed icosahedron
        rng_def = np.random.default_rng(size + 1)
        p_def = p_ico + rng_def.normal(0, 0.1, size=p_ico.shape)
        candidates.append(p_def)
    except Exception:
        pass
        
    # FCC templates
    try:
        p_fcc = _generate_fcc(size)
        candidates.append(p_fcc)
    except Exception:
        pass
        
    # Random packings (diversity)
    for seed in [1, 2, 3, 4, 5, 6]:
        try:
            p_rand = _generate_random_packing(size, seed)
            candidates.append(p_rand)
        except Exception:
            pass

    best_energy = np.inf
    best_pos = None
    
    # 2. Parallel Relaxation
    # Use 4 threads as allowed by task
    max_workers = 4
    
    def _relax_task(pos):
        try:
            # relax returns (energy, positions) based on typical API patterns in such suites
            # If it returns just positions, we might need to compute energy, but task says 
            # "chem_clusters.relax(positions, 'dzugutov')" may be called.
            # Usually these helpers return the relaxed structure or (energy, structure).
            # Let's assume it returns (energy, structure) or structure.
            # Looking at similar tasks, often it returns the relaxed positions.
            # However, the score calculation requires energy. 
            # The task says "Energies are recomputed exactly... by the evaluator".
            # So we just need to return positions. But to pick the best, we need energy.
            # Does relax return energy? The prompt says: "chem_clusters.relax(positions, 'dzugutov') ... may be called".
            # It doesn't explicitly state the return type.
            # Common pattern in these 'chem_clusters' suites: returns (energy, positions).
            # If it only returns positions, we can't compare energies directly without a separate energy function.
            # The prompt mentions `dzugutov_energy_gradient` is available, implying an energy function exists or is part of the suite.
            # Let's assume relax returns (energy, relaxed_pos). If it returns just pos, this will fail.
            # Given "score 3.8450" current program uses `energy, relaxed_pos = relax(...)`, we assume this signature is correct.
            energy, relaxed_pos = relax(pos, potential, parameter)
            return energy, relaxed_pos
        except Exception as e:
            return np.inf, pos # Return original on failure

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_relax_task, pos): i for i, pos in enumerate(candidates)}
        
        for future in as_completed(futures):
            if time.time() > deadline - 10:
                break
            try:
                energy, pos = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except Exception:
                continue

    # 3. Refinement (Basin Hopping)
    if best_pos is not None:
        rng = np.random.default_rng(size + 100)
        current_energy = best_energy
        current_pos = best_pos.copy()
        
        # Use remaining time for refinement
        # Estimate time per relax. If we have 30s left and 4 threads, we can do ~12 relaxations in parallel batches.
        # Let's do a few rounds of parallel perturbations.
        
        max_refine_rounds = 5
        perturbations_per_round = 4
        
        for round_num in range(max_refine_rounds):
            if time.time() > deadline - 5:
                break
                
            # Generate perturbations
            sigma = 0.2 * np.exp(-round_num * 0.2)
            perturbations = []
            for _ in range(perturbations_per_round):
                noise = rng.normal(0, sigma, size=current_pos.shape)
                perturbations.append(current_pos + noise)
            
            # Relax in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_relax_task, p): i for i, p in enumerate(perturbations)}
                for future in as_completed(futures):
                    if time.time() > deadline - 2:
                        break
                    try:
                        energy, pos = future.result()
                        if energy < current_energy:
                            current_energy = energy
                            current_pos = pos
                            best_energy = energy
                            best_pos = pos
                    except Exception:
                        continue
                        
            # If we found a better energy, we might want to continue from there (done above)
            # If not, we might want to increase sigma or stop.
            
    if best_pos is None:
        # Fallback
        if candidates:
            return candidates[0]
        else:
            return np.array([np.array([i, 0, 0]) for i in range(size)])

    return best_pos
# EVOLVE-BLOCK-END
