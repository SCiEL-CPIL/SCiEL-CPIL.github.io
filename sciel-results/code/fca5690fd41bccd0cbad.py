"""Parallel multi-start basin-hopping with diverse motifs for Sutton-Chen cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.spatial.distance import pdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _generate_random_compact(n, rng, min_dist=0.5):
    """Generate a random compact cluster with minimum distance constraints."""
    points = []
    radius = 0.7 * n ** (1.0 / 3.0) + 1.0
    max_attempts = 2000 * n
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            attempts += 1
            continue
        if not points:
            points.append(p)
        else:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) >= min_dist:
                points.append(p)
            else:
                attempts += 1
    if len(points) < n:
        # Fallback: simple cubic grid
        points = []
        side = int(np.ceil(n ** (1/3))) + 1
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    if len(points) < n:
                        points.append(np.array([i, j, k]) * 0.8)
        points = np.array(points[:n])
        points -= points.mean(axis=0)
    else:
        points = np.array(points)
        points -= points.mean(axis=0)
    return points


def _generate_icosahedral(n, rng):
    """Generate an icosahedral-like cluster (Mackay shells)."""
    phi = (1 + 5 ** 0.5) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    k = 1
    while len(points) < n:
        for v in verts:
            if len(points) < n:
                points.append(k * v)
        if len(points) < n:
            for i in range(12):
                v1 = verts[i]
                v2 = verts[(i + 1) % 12]
                for t in np.linspace(0.2, 0.8, 5):
                    if len(points) < n:
                        p = k * (v1 * (1 - t) + v2 * t)
                        p = p / np.linalg.norm(p) * k
                        points.append(p)
        k += 1
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _generate_fcc(n, rng):
    """Generate an FCC-like cluster."""
    points = []
    a = 1.0
    max_coord = int(np.ceil(n ** (1/3) / 2)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
                if len(points) >= n * 2:
                    break
            if len(points) >= n * 2:
                break
        if len(points) >= n * 2:
            break
    points = np.array(points)
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points -= points.mean(axis=0)
    return points


def _generate_hcp(n, rng):
    """Generate an HCP-like cluster."""
    points = []
    a = 1.0
    c_over_a = 1.633
    max_coord = int(np.ceil(n ** (1/3) / 2)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                p = np.array([i + 0.5 * (j % 2), j * np.sqrt(3) / 2, k * c_over_a]) * a
                points.append(p)
                if len(points) >= n * 2:
                    break
            if len(points) >= n * 2:
                break
        if len(points) >= n * 2:
            break
    points = np.array(points)
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points -= points.mean(axis=0)
    return points


def _basin_hopping_worker(initial_pos, kind, rng_seed, time_limit, max_iter=30):
    """
    Perform basin hopping from initial_pos within time_limit.
    Returns (energy, positions).
    """
    start_time = time.time()
    current_energy, current_pos = relax(initial_pos, 'sutton_chen', kind)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    rng = np.random.default_rng(rng_seed)
    step_size = 0.25
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Relax
        new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
        
        # Metropolis acceptance with adaptive temperature
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            step_size *= 0.95
        else:
            delta_e = new_energy - current_energy
            temp = 0.1 * (1.0 - i / max_iter) + 0.02
            if rng.random() < np.exp(-delta_e / temp):
                current_energy = new_energy
                current_pos = new_pos
                step_size *= 1.05
            else:
                step_size *= 0.9
        
        # Track best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
            
        # Check minimum distance
        min_dist = pdist(current_pos).min()
        if min_dist < 0.35:
            current_pos += rng.normal(0, 0.1, current_pos.shape)
            current_energy, current_pos = relax(current_pos, 'sutton_chen', kind)
    
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
    
    initial_structures = []
    
    # Random compact
    for i in range(3):
        initial_structures.append(_generate_random_compact(n, rng_master, min_dist=0.6))
    
    # Icosahedral
    initial_structures.append(_generate_icosahedral(n, rng_master))
    
    # FCC
    initial_structures.append(_generate_fcc(n, rng_master))
    
    # HCP
    initial_structures.append(_generate_hcp(n, rng_master))
    
    # More random
    for i in range(2):
        initial_structures.append(_generate_random_compact(n, rng_master, min_dist=0.5))
    
    # Total 8 initial structures
    
    # Select 4 diverse starts for 4 parallel workers
    selected_starts = [
        initial_structures[0],  # Random 1
        initial_structures[3],  # Icosa
        initial_structures[4],  # FCC
        initial_structures[5],  # HCP
    ]
    
    results = []
    
    # Use ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, start_pos in enumerate(selected_starts):
            worker_time_limit = (deadline - time.time()) / 4 - 5
            if worker_time_limit < 5:
                worker_time_limit = 5
            seed = 100 + i
            futures.append(executor.submit(_basin_hopping_worker, start_pos, kind, seed, worker_time_limit, max_iter=50))
        
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
    min_dist = pdist(best_pos).min()
    if min_dist < 0.35:
        for _ in range(10):
            best_pos += np.random.default_rng(123).normal(0, 0.05, best_pos.shape)
            min_dist = pdist(best_pos).min()
            if min_dist >= 0.35:
                break
        try:
            best_energy, best_pos = relax(best_pos, 'sutton_chen', kind)
        except:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
