"""Morse cluster global minima finder using parallel basin hopping with vectorized operations."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed

TOTAL_TIME_BUDGET = 290.0
INSTANCE_TIME_BUDGET = TOTAL_TIME_BUDGET / 6.0
_start_global = time.time()


def _morse_energy(pos, rho):
    """Compute Morse energy for all pairs (vectorized)."""
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    dist = dist[mask]
    x = rho * (1.0 - dist)
    exp_x = np.exp(x)
    energy = np.sum(exp_x * (exp_x - 2.0))
    return energy


def _morse_gradient(pos, rho):
    """Compute gradient of Morse energy (vectorized)."""
    n = pos.shape[0]
    if n < 2:
        return np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-15)
    mask = ~np.eye(n, dtype=bool)
    
    x = rho * (1.0 - dist)
    exp_x = np.exp(x)
    dE_dr = -2.0 * rho * exp_x * (exp_x - 1.0)
    
    # Vectorized gradient computation
    # grad_i = sum_{j != i} dE_dr[i,j] * (pos_i - pos_j) / r_ij
    # = sum_{j != i} dE_dr[i,j] / r_ij * (pos_i - pos_j)
    # = (sum_j dE_dr[i,j]/r_ij) * pos_i - sum_j (dE_dr[i,j]/r_ij) * pos_j
    
    # Set diagonal to 0 to avoid self-interaction
    dE_dr_diag = np.where(mask, dE_dr, 0.0)
    dist_diag = np.where(mask, dist, 1.0)  # avoid division by zero on diagonal
    
    coeff = dE_dr_diag / dist_diag  # shape (n, n)
    
    # grad_i = sum_j coeff[i,j] * (pos_i - pos_j)
    #        = pos_i * sum_j coeff[i,j] - sum_j coeff[i,j] * pos_j
    sum_coeff = np.sum(coeff, axis=1)  # shape (n,)
    weighted_pos = coeff @ pos  # shape (n, 3)
    
    grad = pos * sum_coeff[:, np.newaxis] - weighted_pos
    return grad


def _local_minimize(pos, rho, maxiter=100, time_limit=None):
    """Local minimization using L-BFGS-B."""
    n = pos.shape[0]
    
    def objective_and_grad(x):
        p = x.reshape(-1, 3)
        e = _morse_energy(p, rho)
        g = _morse_gradient(p, rho).flatten()
        return e, g
    
    try:
        res = minimize(
            objective_and_grad,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-8}
        )
        return res.fun, res.x.reshape(-1, 3)
    except Exception:
        return _morse_energy(pos, rho), pos


def _enforce_min_distance(pos, min_dist=0.5):
    """Push apart atoms that are too close."""
    n = pos.shape[0]
    for _ in range(100):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                r = np.linalg.norm(d)
                if r < min_dist and r > 1e-10:
                    push = (min_dist - r) / r * 0.5
                    pos[i] += d * push
                    pos[j] -= d * push
                    moved = True
                elif r <= 1e-10:
                    pos[i] += np.random.randn(3) * 0.01
                    moved = True
        if not moved:
            break
    return pos


def _generate_fcc_seed(n, spacing=1.0):
    """Generate FCC-like initial configuration."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    max_coord = int(np.ceil(n ** (1/3.0))) + 4
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * spacing
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = base + np.array([dx, dy, dz]) * spacing
                    if np.linalg.norm(p) <= max_coord * spacing:
                        points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    seed = points[idx].copy()
    seed -= seed.mean(axis=0)
    return seed


def _generate_hcp_seed(n, spacing=1.0):
    """Generate HCP-like initial configuration."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    a = spacing
    c = np.sqrt(8.0/3.0) * a
    max_coord = int(np.ceil(n ** (1/3.0))) + 4
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0)/2.0 * j
                z = c * k
                p = np.array([x, y, z])
                if np.linalg.norm(p) <= max_coord * a:
                    points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    seed = points[idx].copy()
    seed -= seed.mean(axis=0)
    return seed


def _generate_icosahedral_seed(n, spacing=1.0):
    """Generate icosahedral shell-like initial configuration."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Icosahedral vertices (12 points)
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi]
    ]
    
    for v in ico_vertices:
        p = np.array(v, dtype=float) * spacing
        points.append(p)
    
    # Add more shells if needed
    shell_radius = spacing * 2.0
    while len(points) < n:
        # Add points on a sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi_angle = np.random.uniform(0, np.pi)
        p = shell_radius * np.array([
            np.sin(phi_angle) * np.cos(theta),
            np.sin(phi_angle) * np.sin(theta),
            np.cos(phi_angle)
        ])
        points.append(p)
        shell_radius *= 1.3
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _generate_random_seed(n, spacing=1.0, rng=None):
    """Generate random compact seed."""
    if rng is None:
        rng = np.random.default_rng()
    radius = spacing * n ** (1.0/3.0) * 0.9
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't fit enough, just use random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _basin_hopping_worker(args):
    """Worker for parallel basin hopping."""
    pos, rho, n, time_limit, rng_seed = args
    rng = np.random.default_rng(rng_seed)
    t_start = time.time()
    
    current_energy, current_pos = _local_minimize(pos, rho)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    temp = 0.15 * max(1.0, abs(current_energy) / n)
    
    while time.time() - t_start < time_limit:
        if time.time() - t_start > time_limit:
            break
        
        kick_size = temp * 0.25
        kick = rng.normal(0, kick_size, (n, 3))
        trial_pos = current_pos + kick
        trial_pos = _enforce_min_distance(trial_pos, 0.5)
        
        trial_energy, trial_pos = _local_minimize(trial_pos, rho, maxiter=80)
        
        delta_e = trial_energy - current_energy
        if delta_e < 0 or (delta_e < temp and rng.random() < np.exp(-delta_e / temp)):
            current_energy = trial_energy
            current_pos = trial_pos.copy()
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = trial_pos.copy()
        
        temp *= 0.97
        if temp < 1e-5 * max(1.0, abs(best_energy) / n):
            break
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    t_start = time.time()
    time_remaining = lambda: INSTANCE_TIME_BUDGET - (time.time() - t_start)
    
    rng = np.random.default_rng(hash((n, rho)) % 2**32)
    
    # Generate diverse initial seeds
    seeds = []
    seeds.append(_generate_fcc_seed(n))
    seeds.append(_generate_hcp_seed(n))
    seeds.append(_generate_icosahedral_seed(n))
    for _ in range(4):
        seeds.append(_generate_random_seed(n, rng=rng))
    
    # Ensure all seeds satisfy minimum distance
    seeds = [_enforce_min_distance(s, 0.5) for s in seeds]
    
    n_seeds = len(seeds)
    per_seed_time = max(0.3, time_remaining() * 0.85 / n_seeds)
    
    # Run parallel basin hopping
    best_energy = np.inf
    best_pos = seeds[0].copy()
    
    workers = min(4, n_seeds)
    tasks = [(seeds[i], rho, n, per_seed_time, hash((n, rho, i)) % 2**32) for i in range(n_seeds)]
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_basin_hopping_worker, task): i for i, task in enumerate(tasks)}
        
        for future in as_completed(futures):
            if time.time() - t_start > INSTANCE_TIME_BUDGET - 5:
                break
            try:
                e, pos = future.result(timeout=INSTANCE_TIME_BUDGET)
                if e < best_energy:
                    best_energy = e
                    best_pos = pos.copy()
            except Exception:
                pass
    
    # Final polish
    if time_remaining() > 3.0:
        best_pos = _enforce_min_distance(best_pos, 0.5)
        final_energy, final_pos = _local_minimize(best_pos, rho, maxiter=150)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pos = final_pos
        best_pos = _enforce_min_distance(best_pos, 0.5)
    
    return best_pos
# EVOLVE-BLOCK-END
