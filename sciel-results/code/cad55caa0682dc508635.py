"""
Strategy: Parallel Basin Hopping with Vectorized Morse Energy/Gradient and Crystalline + Random Seeds.
Materially different from the current single-threaded multi-start L-BFGS-B: this uses scipy's
concurrent.futures with 4 workers to run independent basin-hopping trajectories simultaneously,
allowing many more kick-relax iterations within the same wall-clock budget. Each worker runs
basin hopping (L-BFGS-B local relax + Gaussian kick) from a diverse seed (FCC, BCC, icosahedral,
or random). The global best is tracked across all workers.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 295.0  # 300s limit, 5s safety margin

# Thread-local best tracking
_best_lock = threading.Lock()
_best_energy = float('inf')
_best_positions = None

def _morse_energy_and_grad(positions, rho):
    """Compute exact Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    energy = 0.5 * float(np.sum(ex * (ex - 2.0)))
    
    dV_dr = -rho * 2.0 * ex * (ex - 1.0)
    
    inv_r = np.zeros_like(r)
    mask = r > 1e-12
    inv_r[mask] = 1.0 / r[mask]
    
    factor = dV_dr * inv_r
    row_sum = np.sum(factor, axis=1)
    weighted_pos = factor.dot(positions)
    grad = positions * row_sum[:, np.newaxis] - weighted_pos
    
    return energy, grad

def _generate_fcc(n, rng):
    """Generate n atoms on an FCC lattice, cropped to a sphere."""
    a = np.sqrt(2.0)
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0,0,0), (0, 0.5*a, 0.5*a), (0.5*a, 0, 0.5*a), (0.5*a, 0.5*a, 0)]:
                    p = base + np.array([dx, dy, dz])
                    points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points += rng.normal(0, 0.01, points.shape)
    return points

def _generate_bcc(n, rng):
    """Generate n atoms on a BCC lattice, cropped to a sphere."""
    a = 2.0 / np.sqrt(3.0)
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0,0,0), (0.5*a, 0.5*a, 0.5*a)]:
                    p = base + np.array([dx, dy, dz])
                    points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points += rng.normal(0, 0.01, points.shape)
    return points

def _generate_icosahedron(n, rng):
    """Generate n atoms in an icosahedral shell structure."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi]
    ], dtype=float)
    
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    scale = 1.5 * n ** (1/3)
    vertices *= scale
    
    points = [np.zeros(3)]
    for v in vertices:
        points.append(v.copy())
    
    shell_radius = scale * 1.5
    while len(points) < n:
        theta = rng.uniform(0, 2*np.pi)
        phi_a = rng.uniform(0, np.pi)
        p = shell_radius * np.array([
            np.sin(phi_a) * np.cos(theta),
            np.sin(phi_a) * np.sin(theta),
            np.cos(phi_a)
        ])
        points.append(p)
    
    points = np.array(points[:n])
    points += rng.normal(0, 0.01, points.shape)
    return points

def _generate_random(n, rng):
    """Generate n atoms randomly in a compact region."""
    radius = 1.2 * n ** (1.0 / 3.0)
    points = []
    for _ in range(n):
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    
    points = np.array(points)
    points += rng.normal(0, 0.01, points.shape)
    return points

def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.all(r >= min_dist)

def _local_minimize(positions, rho, maxiter=200):
    """Local minimization using L-BFGS-B."""
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(x, rho)
        return e, g.flatten()
    
    try:
        res = minimize(fun, positions.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-6})
        final_pos = res.x.reshape(-1, 3)
        final_energy = res.fun
        if _is_valid(final_pos):
            return final_pos, final_energy
    except Exception:
        pass
    return positions, _morse_energy_and_grad(positions, rho)[0]

def _basin_hopping_worker(seed_pos, rho, rng, worker_id):
    """Run basin hopping from a seed configuration."""
    global _best_energy, _best_positions
    
    pos = seed_pos.copy()
    # Initial relaxation
    pos, energy = _local_minimize(pos, rho, maxiter=300)
    
    kick_scale = 0.3
    n_iter = 0
    
    while time.time() - _START_TIME < _TIME_LIMIT - 10:
        # Kick
        kick = rng.normal(0, kick_scale, pos.shape)
        kicked = pos + kick
        
        # Relax
        pos_new, energy_new = _local_minimize(kicked, rho, maxiter=150)
        n_iter += 1
        
        if energy_new < energy:
            # Accept: update current
            pos = pos_new
            energy = energy_new
            
            with _best_lock:
                if energy < _best_energy:
                    _best_energy = energy
                    _best_positions = pos.copy()
            
            # Slowly reduce kick scale
            kick_scale = max(0.1, kick_scale * 0.98)
        else:
            # Reject: keep current, maybe increase kick slightly
            kick_scale = min(1.0, kick_scale * 1.02)
        
        # Periodically try a larger kick to escape
        if n_iter % 10 == 0:
            big_kick = rng.normal(0, 0.5, pos.shape)
            kicked_big = pos + big_kick
            pos_big, energy_big = _local_minimize(kicked_big, rho, maxiter=100)
            if energy_big < energy:
                pos = pos_big
                energy = energy_big
                with _best_lock:
                    if energy < _best_energy:
                        _best_energy = energy
                        _best_positions = pos.copy()
    
    return pos, energy

def construct(instance):
    global _START_TIME, _TIME_LIMIT, _best_energy, _best_positions
    
    _START_TIME = time.time()
    _TIME_LIMIT = 295.0
    _best_energy = float('inf')
    _best_positions = None
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Generate diverse seeds
    rng = np.random.default_rng(42)
    seeds = []
    
    # Crystalline seeds
    seeds.append(_generate_fcc(size, rng))
    seeds.append(_generate_bcc(size, rng))
    seeds.append(_generate_icosahedron(size, rng))
    
    # Random seeds
    for i in range(3):
        rng_i = np.random.default_rng(100 + i)
        seeds.append(_generate_random(size, rng_i))
    
    # Distribute seeds across 4 workers
    num_workers = 4
    worker_seeds = [[] for _ in range(num_workers)]
    for i, seed in enumerate(seeds):
        worker_seeds[i % num_workers].append(seed)
    
    # Run parallel basin hopping
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for w in range(num_workers):
            if worker_seeds[w]:
                # Each worker runs basin hopping from its first seed, then tries others
                for s_idx, seed in enumerate(worker_seeds[w]):
                    rng_w = np.random.default_rng(200 + w * 10 + s_idx)
                    futures.append(executor.submit(_basin_hopping_worker, seed, parameter, rng_w, w))
        
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                pass
    
    # If no valid result, generate a fallback
    if _best_positions is None or not _is_valid(_best_positions):
        radius = 1.5 * size ** (1.0 / 3.0)
        points = []
        rng_f = np.random.default_rng(999)
        while len(points) < size:
            p = rng_f.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        _best_positions = np.array(points)
    
    return _best_positions
# EVOLVE-BLOCK-END
