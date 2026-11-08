# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.spatial.distance import pdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _generate_random_compact(n, rng, min_dist=0.65):
    """Generate a random compact cluster with minimum distance constraints."""
    points = []
    radius = 0.85 * n ** (1.0 / 3.0) + 1.5
    max_attempts = 8000 * n
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
                        points.append(np.array([i, j, k]) * 0.75)
        points = np.array(points[:n])
        points -= points.mean(axis=0)
    else:
        points = np.array(points)
        points -= points.mean(axis=0)
    return points


def _generate_icosahedral(n, rng):
    """Generate an icosahedral-like cluster."""
    points = []
    # Center atom
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # Icosahedral vertices (12 atoms)
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    scale = 1.0
    for v in ico_vertices:
        if len(points) < n:
            points.append(np.array(v, dtype=float) * scale)
    
    # Second shell: 30 atoms (edge midpoints of icosahedron)
    if len(points) < n:
        # Add atoms at various positions to fill up
        rng_local = np.random.default_rng(42)
        while len(points) < n:
            r = 1.5 * scale * (1 + 0.3 * rng_local.random())
            theta = rng_local.random() * 2 * np.pi
            z = 2 * rng_local.random() - 1
            xy = np.sqrt(1 - z*z)
            p = np.array([r * xy * np.cos(theta), r * xy * np.sin(theta), r * z])
            # Check minimum distance
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.55:
                    points.append(p)
            else:
                points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _generate_fcc_cluster(n, rng):
    """Generate an FCC-like cluster."""
    points = []
    # FCC lattice points within a sphere
    a = 1.0  # lattice constant
    max_coord = int(np.ceil(n ** (1/3))) + 2
    fcc_points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: corners and face centers
                fcc_points.append(np.array([i, j, k]) * a)
                fcc_points.append(np.array([i + 0.5, j + 0.5, k]) * a)
                fcc_points.append(np.array([i + 0.5, j, k + 0.5]) * a)
                fcc_points.append(np.array([i, j + 0.5, k + 0.5]) * a)
    
    # Keep only points within a sphere
    fcc_points = np.array(fcc_points)
    center = fcc_points.mean(axis=0)
    fcc_points -= center
    norms = np.linalg.norm(fcc_points, axis=1)
    max_r = 1.2 * n ** (1/3)
    mask = norms <= max_r
    fcc_points = fcc_points[mask]
    
    # Shuffle and take n points
    idx = rng.permutation(len(fcc_points))
    points = fcc_points[idx[:min(n, len(fcc_points))]]
    
    if len(points) < n:
        # Fill remaining with random points
        while len(points) < n:
            p = rng.uniform(-max_r, max_r, 3)
            if np.linalg.norm(p) <= max_r:
                if len(points) > 0:
                    dists = np.linalg.norm(points - p, axis=1)
                    if np.min(dists) >= 0.5:
                        points = np.vstack([points, p])
                else:
                    points = p.reshape(1, 3)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _generate_hcp_cluster(n, rng):
    """Generate an HCP-like cluster."""
    points = []
    a = 1.0
    c = np.sqrt(8/3) * a
    max_coord = int(np.ceil(n ** (1/3))) + 2
    
    hcp_points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # HCP: two interpenetrating triangular lattices
                x = a * (i + j/2)
                y = a * np.sqrt(3)/2 * j
                z = c/2 * k
                hcp_points.append(np.array([x, y, z]))
                # Second sublattice
                x2 = a * (i + j/2 + 1/3)
                y2 = a * np.sqrt(3)/2 * (j + 2/3)
                z2 = c/2 * k + c/2
                hcp_points.append(np.array([x2, y2, z2]))
    
    hcp_points = np.array(hcp_points)
    center = hcp_points.mean(axis=0)
    hcp_points -= center
    norms = np.linalg.norm(hcp_points, axis=1)
    max_r = 1.2 * n ** (1/3)
    mask = norms <= max_r
    hcp_points = hcp_points[mask]
    
    idx = rng.permutation(len(hcp_points))
    points = hcp_points[idx[:min(n, len(hcp_points))]]
    
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-max_r, max_r, 3)
            if np.linalg.norm(p) <= max_r:
                if len(points) > 0:
                    dists = np.linalg.norm(points - p, axis=1)
                    if np.min(dists) >= 0.5:
                        points = np.vstack([points, p])
                else:
                    points = p.reshape(1, 3)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _basin_hopping_worker(initial_pos, kind, rng_seed, time_limit, max_iter=150):
    """
    Perform basin hopping from initial_pos within time_limit.
    Uses adaptive k-atom perturbation for structural diversity.
    """
    start_time = time.time()
    current_energy, current_pos = relax(initial_pos, 'sutton_chen', kind)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    rng = np.random.default_rng(rng_seed)
    n = current_pos.shape[0]
    
    # Adaptive step size and k-atom count
    step_size = 0.35
    k_atoms = max(2, int(0.12 * n))
    
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
        
        # Metropolis acceptance with adaptive temperature
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            step_size *= 0.92
        else:
            delta_e = new_energy - current_energy
            # Cooling schedule
            temp = 0.4 * np.exp(-i / 15.0) + 0.02
            if rng.random() < np.exp(-delta_e / temp):
                current_energy = new_energy
                current_pos = new_pos
                step_size *= 1.08
            else:
                step_size *= 0.92
        
        # Track best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
            
        # Check minimum distance and fix if needed
        if n > 1:
            min_dist = pdist(current_pos).min()
            if min_dist < 0.35:
                # Push atoms apart
                for _ in range(5):
                    current_pos += rng.normal(0, 0.08, current_pos.shape)
                    min_dist = pdist(current_pos).min()
                    if min_dist >= 0.35:
                        break
                current_energy, current_pos = relax(current_pos, 'sutton_chen', kind)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """
    Parallel multi-start basin-hopping search for Sutton-Chen cluster global minima.
    Uses diverse initial structures: random, icosahedral, FCC, and HCP motifs.
    """
    start_time = time.time()
    deadline = start_time + 285  # 15s safety margin
    
    n = instance["n"]
    kind = instance["kind"]
    
    # Generate diverse initial structures using different motifs
    rng_master = np.random.default_rng(42)
    
    initial_structures = []
    
    # Random compact
    initial_structures.append(_generate_random_compact(n, rng_master, min_dist=0.65))
    
    # Icosahedral-like
    try:
        initial_structures.append(_generate_icosahedral(n, rng_master))
    except Exception:
        initial_structures.append(_generate_random_compact(n, rng_master, min_dist=0.65))
    
    # FCC-like
    try:
        initial_structures.append(_generate_fcc_cluster(n, rng_master))
    except Exception:
        initial_structures.append(_generate_random_compact(n, rng_master, min_dist=0.65))
    
    # HCP-like
    try:
        initial_structures.append(_generate_hcp_cluster(n, rng_master))
    except Exception:
        initial_structures.append(_generate_random_compact(n, rng_master, min_dist=0.65))
    
    results = []
    
    # Use ThreadPoolExecutor with 4 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, start_pos in enumerate(initial_structures):
            worker_time_limit = (deadline - time.time()) / 4 - 5
            if worker_time_limit < 15:
                worker_time_limit = 15
            seed = 100 + i
            futures.append(executor.submit(_basin_hopping_worker, start_pos, kind, seed, worker_time_limit, max_iter=200))
        
        for future in futures:
            try:
                energy, pos = future.result()
                if np.isfinite(energy):
                    results.append((energy, pos))
            except Exception:
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
            except Exception:
                pass
    
    return best_pos
# EVOLVE-BLOCK-END
