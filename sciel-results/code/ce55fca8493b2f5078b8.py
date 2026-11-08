# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.spatial.distance import pdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _generate_random_compact(n, rng, min_dist=0.6):
    """Generate a random compact cluster with minimum distance constraints."""
    points = []
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
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) >= min_dist:
                points.append(p)
            else:
                attempts += 1
    if len(points) < n:
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


def _generate_icosahedral(n, rng=None):
    """Generate an icosahedral shell cluster."""
    if rng is None:
        rng = np.random.default_rng(42)
    # icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    points = [np.zeros(3)]  # center
    # Shell 1: 12 vertices
    if n > 1:
        points.extend([v * 1.44 for v in verts])
    
    # Shell 2: 42 vertices (octahedral + tetrahedral sites)
    if n > 13:
        # Octahedral sites
        oct_sites = np.array([
            [1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],
            [1,0,1],[1,0,-1],[-1,0,1],[-1,0,-1],
            [0,1,1],[0,1,-1],[0,-1,1],[0,-1,-1]
        ], dtype=float)
        oct_sites /= np.linalg.norm(oct_sites[0])
        for v in oct_sites:
            if len(points) < n:
                points.append(v * 2.1)
        # Tetrahedral sites
        tet_sites = np.array([
            [1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],
            [-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]
        ], dtype=float)
        tet_sites /= np.linalg.norm(tet_sites[0])
        for v in tet_sites:
            if len(points) < n:
                points.append(v * 2.2)
    
    # Shell 3: fill remaining with random points on sphere
    if len(points) < n:
        radius = 2.8
        while len(points) < n:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * radius
            points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _generate_fcc_cluster(n, rng=None):
    """Generate an FCC-like cluster."""
    if rng is None:
        rng = np.random.default_rng(42)
    points = []
    # FCC lattice: atoms at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) * a
    a = 1.0  # lattice spacing
    # Generate points in a cube and select n closest to center
    side = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                for offset in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]]) * a
                    candidates.append(p)
    candidates = np.array(candidates)
    # Sort by distance from origin
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    points = candidates[order[:n]]
    points -= points.mean(axis=0)
    return points


def _generate_hcp_cluster(n, rng=None):
    """Generate an HCP-like cluster."""
    if rng is None:
        rng = np.random.default_rng(42)
    points = []
    a = 1.0  # in-plane spacing
    c = 1.633 * a  # HCP c/a ratio
    # Generate hexagonal layers
    layer_spacing = c / 2
    max_radius = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for layer in range(-max_radius, max_radius+1):
        z = layer * layer_spacing
        for i in range(-max_radius, max_radius+1):
            for j in range(-max_radius, max_radius+1):
                x = i * a + (j % 2) * a / 2
                y = j * a * np.sqrt(3) / 2
                p = np.array([x, y, z])
                candidates.append(p)
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    points = candidates[order[:n]]
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
                current_pos += rng.normal(0, 0.1, current_pos.shape)
                current_energy, current_pos = relax(current_pos, 'sutton_chen', kind)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """
    Parallel multi-start basin-hopping search for Sutton-Chen cluster global minima.
    Uses diverse structural motifs (icosahedral, FCC, HCP, random) as seeds.
    """
    start_time = time.time()
    deadline = start_time + 290  # safety margin
    
    n = instance["n"]
    kind = instance["kind"]
    
    # Generate diverse initial structures from different motifs
    rng_master = np.random.default_rng(42)
    
    initial_structures = []
    # Motif 0: icosahedral shell
    initial_structures.append(_generate_icosahedral(n, rng_master))
    # Motif 1: FCC-like
    initial_structures.append(_generate_fcc_cluster(n, rng_master))
    # Motif 2: HCP-like
    initial_structures.append(_generate_hcp_cluster(n, rng_master))
    # Motif 3: random compact
    initial_structures.append(_generate_random_compact(n, rng_master, min_dist=0.6))
    
    results = []
    
    # Use ThreadPoolExecutor with 4 workers
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
