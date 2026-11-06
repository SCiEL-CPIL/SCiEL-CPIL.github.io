"""Morse cluster global minima: parallel basin hopping with diverse structural seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 285.0  # 15s safety margin before 300s kill


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy(pos, rho):
    """Compute Morse energy for positions."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e10)
    mask = dist > 0.5
    dist_masked = np.where(mask, dist, 1e10)
    x = rho * (1.0 - dist_masked)
    e = np.exp(x)
    pair_e = e * (e - 2.0)
    pair_e[~mask] = 0.0
    np.fill_diagonal(pair_e, 0.0)
    return float(np.sum(pair_e) * 0.5)


def _morse_gradient(pos, rho):
    """Compute Morse gradient for positions."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e10)
    mask = dist > 0.5
    dist_masked = np.where(mask, dist, 1e10)
    x = rho * (1.0 - dist_masked)
    e = np.exp(x)
    de_dr = -2.0 * rho * e * (e - 1.0)
    de_dr[~mask] = 0.0
    np.fill_diagonal(de_dr, 0.0)
    unit_vec = diff / dist_masked[:, :, np.newaxis]
    unit_vec[~mask] = 0.0
    grad = np.sum(de_dr[:, :, np.newaxis] * unit_vec, axis=1)
    return grad


def _enforce_min_dist(pos, min_dist=0.5, max_iter=100):
    """Ensure all pairs are at least min_dist apart."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        # Find closest pair
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1e10)
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        if dist[min_idx] >= min_dist:
            break
        # Push apart
        i, j = min_idx
        if i > j:
            i, j = j, i
        direction = pos[i] - pos[j]
        d = np.linalg.norm(direction)
        if d < 1e-10:
            direction = np.array([1.0, 0.0, 0.0])
            d = 1.0
        direction /= d
        needed = (min_dist - d) / 2.0 + 0.01
        pos[i] += direction * needed
        pos[j] -= direction * needed
    return pos


def _local_minimize(pos, potential, rho, max_iter=300, tol=1e-10):
    """Local minimization using gradient-based method."""
    pos = pos.copy()
    n = len(pos)
    if n < 2:
        return 0.0, pos
    
    energy = _morse_energy(pos, rho)
    step = 0.02
    
    for it in range(max_iter):
        if _time_left() < 10.0:
            break
        grad = _morse_gradient(pos, rho)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        new_pos = pos - step * grad
        new_energy = _morse_energy(new_pos, rho)
        
        if new_energy < energy:
            pos = new_pos
            energy = new_energy
            step = min(step * 1.05, 0.3)
        else:
            step *= 0.5
            if step < 1e-8:
                break
    
    return energy, pos


def _generate_fcc(n, spacing=1.0):
    """Generate FCC lattice points."""
    points = []
    s = spacing
    basis = [
        (0, 0, 0),
        (0, 0.5*s, 0.5*s),
        (0.5*s, 0, 0.5*s),
        (0.5*s, 0.5*s, 0),
    ]
    max_coord = int(np.ceil(n ** (1/3))) + 3
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                for b in basis:
                    p = (i*s + b[0], j*s + b[1], k*s + b[2])
                    points.append(p)
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    return points


def _generate_bcc(n, spacing=1.0):
    """Generate BCC lattice points."""
    points = []
    s = spacing
    basis = [
        (0, 0, 0),
        (0.5*s, 0.5*s, 0.5*s),
    ]
    max_coord = int(np.ceil(n ** (1/3))) + 3
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                for b in basis:
                    p = (i*s + b[0], j*s + b[1], k*s + b[2])
                    points.append(p)
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    return points


def _generate_icosahedron(n, spacing=1.0):
    """Generate icosahedral shell structure."""
    points = [np.zeros(3)]
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        (0, 1/phi, phi), (0, -1/phi, phi), (0, 1/phi, -phi), (0, -1/phi, -phi),
        (1/phi, phi, 0), (-1/phi, phi, 0), (1/phi, -phi, 0), (-1/phi, -phi, 0),
        (phi, 0, 1/phi), (phi, 0, -1/phi), (-phi, 0, 1/phi), (-phi, 0, -1/phi),
    ]
    ico_vertices = np.array(ico_vertices, dtype=float)
    ico_vertices /= np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    shell_radius = spacing
    for v in ico_vertices:
        if len(points) >= n:
            break
        points.append(v * shell_radius)
    
    if len(points) < n:
        oct_dirs = [
            (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
            (1,1,0)/np.sqrt(2), (-1,1,0)/np.sqrt(2), (1,-1,0)/np.sqrt(2), (-1,-1,0)/np.sqrt(2),
            (1,0,1)/np.sqrt(2), (-1,0,1)/np.sqrt(2), (1,0,-1)/np.sqrt(2), (-1,0,-1)/np.sqrt(2),
            (0,1,1)/np.sqrt(2), (0,-1,1)/np.sqrt(2), (0,1,-1)/np.sqrt(2), (0,-1,-1)/np.sqrt(2),
        ]
        shell2_radius = spacing * 1.7
        for d in oct_dirs:
            if len(points) >= n:
                break
            points.append(np.array(d) * shell2_radius)
    
    rng = np.random.default_rng(42)
    while len(points) < n:
        p = rng.normal(size=3) * spacing * 2.5
        if all(np.linalg.norm(p - np.array(q)) >= 0.5 * spacing for q in points):
            points.append(p)
        else:
            points.append(p + rng.normal(0, 0.3, 3))
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _generate_random(n, spacing=1.0, seed=None):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 20000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - np.array(q)) >= 0.6 * spacing for q in points):
            points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _generate_seeds(n, rho):
    """Generate diverse initial configurations."""
    seeds = []
    seeds.append(_generate_fcc(n))
    seeds.append(_generate_bcc(n))
    seeds.append(_generate_icosahedron(n))
    for s in range(6):
        seeds.append(_generate_random(n, seed=s))
    return seeds


def _basin_hopping_worker(seed_pos, potential, rho, n, T0=3.0, cooling=0.97, max_iter=200, seed_offset=0):
    """Run basin hopping from a given seed."""
    rng = np.random.default_rng(seed_offset)
    pos = seed_pos.copy()
    energy, pos = _local_minimize(pos, potential, rho)
    best_energy = energy
    best_pos = pos.copy()
    T = T0
    
    for it in range(max_iter):
        if _time_left() < 15.0:
            break
        
        # Perturb
        noise_scale = np.sqrt(2 * T / n)
        perturbed = pos + rng.normal(0, noise_scale, pos.shape)
        perturbed = _enforce_min_dist(perturbed)
        
        # Relax
        new_energy, new_pos = _local_minimize(perturbed, potential, rho)
        
        # Accept
        if new_energy < energy or rng.random() < np.exp(-(new_energy - energy) / T):
            energy = new_energy
            pos = new_pos
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        
        T *= cooling
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Generate diverse seeds
    seeds = _generate_seeds(size, rho)
    
    # Run parallel basin hopping on multiple seeds
    best_energy = np.inf
    best_positions = seeds[0]
    
    # Use concurrent futures for parallel search
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, seed in enumerate(seeds[:4]):  # Use 4 best seeds for parallel
            if _time_left() < 20.0:
                break
            futures.append(executor.submit(_basin_hopping_worker, seed, potential, rho, size, seed_offset=i*1000))
        
        for future in concurrent.futures.as_completed(futures):
            if _time_left() < 10.0:
                break
            try:
                energy, positions = future.result(timeout=5.0)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                pass
    
    # If time remains, do more sequential basin hopping on remaining seeds
    for seed in seeds[4:]:
        if _time_left() < 20.0:
            break
        try:
            energy, positions = _basin_hopping_worker(seed, potential, rho, size, T0=2.0, cooling=0.96, max_iter=150, seed_offset=5000)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # Final local minimization to ensure we're at a minimum
    if _time_left() > 5.0:
        final_energy, final_pos = _local_minimize(best_positions, potential, rho)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
    
    return best_positions
# EVOLVE-BLOCK-END
