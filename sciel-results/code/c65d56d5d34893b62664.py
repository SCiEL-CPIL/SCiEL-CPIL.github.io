"""
Morse cluster global minima: Parallel Basin Hopping with optimized local minimization.
Uses 4 threads to run independent Basin Hopping chains.
Prioritizes chem_clusters.relax for local minimization, falling back to L-BFGS.
Combines diverse crystalline initializations (FCC, BCC, HCP, Shell) with vectorized
energy/gradient and efficient local relaxation.
"""
import numpy as np
import time
import threading
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Deadline guard
TOTAL_DEADLINE = 285.0  # Safety margin before 300s limit (evaluator kills at 330s, but task says 300s)
_START_TIME = None
_lock = threading.Lock()
_best_energy_global = np.inf
_best_positions_global = None


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient vectorized."""
    n = positions.shape[0]
    if n < 2:
        return 0.0, np.zeros_like(positions)
        
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(d, 1.0)
    r = np.maximum(d, 1e-10)
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    # Gradient
    # V(r) = exp(x)(exp(x) - 2)
    # dV/dr = exp(x)(exp(x) - 1) * dx/dr = -rho * exp(x) * (exp(x) - 1)
    dvdr = -rho * ex * (ex - 1.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        w = dvdr / d
    np.fill_diagonal(w, 0.0)
    
    grad = np.sum(w[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _local_minimize(positions, rho):
    """
    Local minimization.
    Tries chem_clusters.relax first, then falls back to L-BFGS.
    """
    global _START_TIME
    
    # Check deadline
    if _START_TIME is None:
        return positions, _morse_energy_and_grad(positions, rho)[0]
        
    if time.time() - _START_TIME > TOTAL_DEADLINE - 5:
        return positions, _morse_energy_and_grad(positions, rho)[0]

    # Try the provided relax function
    try:
        relaxed = relax(positions, 'morse', rho)
        if relaxed is not None:
            relaxed = np.array(relaxed, dtype=float)
            if relaxed.shape == positions.shape:
                e = _morse_energy_and_grad(relaxed, rho)[0]
                if np.isfinite(e):
                    return relaxed, e
    except Exception:
        pass

    # Fallback: L-BFGS
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.flatten()
        
    x0 = positions.flatten()
    try:
        result = minimize(fun, x0, method='L-BFGS-B', jac=True, 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return positions, _morse_energy_and_grad(positions, rho)[0]


def _is_valid(positions, min_dist=0.5):
    """Check if all pair distances are at least min_dist."""
    try:
        d = pdist(positions)
        return np.min(d) >= min_dist
    except Exception:
        return False


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like compact cluster."""
    a = spacing
    max_coord = int(np.ceil(n ** (1.0/3.0)) * 2) + 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]].copy()


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like compact cluster."""
    a = spacing
    max_coord = int(np.ceil(n ** (1.0/3.0)) * 2) + 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]].copy()


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like compact cluster."""
    a = spacing
    max_coord = int(np.ceil(n ** (1.0/3.0)) * 2) + 4
    points = []
    # Layer A (triangular)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            x = i * a
            y = j * a * np.sqrt(3) / 2
            if i % 2 == 1:
                y += a * np.sqrt(3) / 4
            points.append(np.array([x, y, 0]))
    z_b = a * np.sqrt(2.0/3.0)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            x = i * a
            y = j * a * np.sqrt(3) / 2
            if i % 2 == 1:
                y += a * np.sqrt(3) / 4
            points.append(np.array([x, y, z_b]))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]].copy()


def _generate_shell(n, spacing=1.0):
    """Generate spherical shell-like cluster (icosahedral-like)."""
    points = []
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    radius = spacing * n ** (1.0/3.0) * 0.45
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        radius_xy = np.sqrt(1.0 - z * z)
        theta = 2.0 * np.pi * i / golden_ratio
        x = radius_xy * np.cos(theta)
        y = radius_xy * np.sin(theta)
        points.append([x * radius, y * radius, z * radius])
    return np.array(points, dtype=float)


def _generate_random_compact(n, spacing=1.0, seed=None):
    """Generate a random compact cluster with minimum spacing."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.9
    min_dist = 0.55 * spacing
    points = []
    attempts = 0
    max_attempts = n * 300
    while len(points) < n and attempts < max_attempts:
        while True:
            p = rng.normal(0, 1, 3)
            norm = np.linalg.norm(p)
            if norm <= 1:
                p = p / norm * radius * rng.uniform(0.2, 1.0)
                break
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
        attempts += 1
    while len(points) < n:
        p = np.array(points[len(points) % max(1, len(points))]) + rng.normal(0, 0.1, 3)
        points.append(p)
    return np.array(points)


def _bh_worker(start_pos, rho, n, thread_id, stop_event, local_best):
    """
    Runs a single Basin Hopping chain.
    Updates local_best dict if a better solution is found.
    """
    global _START_TIME
    
    current_pos = start_pos.copy()
    current_pos, current_energy = _local_minimize(current_pos, rho)
    
    if current_energy < local_best['energy']:
        local_best['energy'] = current_energy
        local_best['positions'] = current_pos.copy()

    rng = np.random.default_rng(thread_id * 10000 + n)
    bh_step = 0.3
    
    while not stop_event.is_set():
        if _START_TIME is not None and time.time() - _START_TIME > TOTAL_DEADLINE - 2:
            break
            
        noise = rng.normal(0, bh_step, current_pos.shape)
        perturbed = current_pos + noise
        
        new_pos, new_energy = _local_minimize(perturbed, rho)
        
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
            if current_energy < local_best['energy']:
                local_best['energy'] = current_energy
                local_best['positions'] = current_pos.copy()
        else:
            delta_e = new_energy - current_energy
            T = 1.0 
            if delta_e < 50:
                if rng.random() < np.exp(-delta_e / T):
                    current_pos = new_pos
                    current_energy = new_energy
                    if current_energy < local_best['energy']:
                        local_best['energy'] = current_energy
                        local_best['positions'] = current_pos.copy()


def construct(instance):
    global _START_TIME, _best_energy_global, _best_positions_global
    
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    starts = []
    
    try: starts.append(_generate_fcc(size, 1.0))
    except: pass
    try: starts.append(_generate_bcc(size, 1.0))
    except: pass
    try: starts.append(_generate_hcp(size, 1.0))
    except: pass
    try: starts.append(_generate_shell(size, 1.0))
    except: pass
    
    for seed in [size, size+1, size+2, size+3, size+4, size+5]:
        try: starts.append(_generate_random_compact(size, 1.0, seed=seed))
        except: pass
        
    if not starts:
        starts.append(_generate_random_compact(size, 1.0, seed=42))

    best_energy = np.inf
    best_positions = starts[0]
    
    for s in starts:
        if time.time() - _START_TIME > TOTAL_DEADLINE - 10:
            break
        try:
            pos, e = _local_minimize(s, rho)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            pass

    num_threads = 4
    np.random.shuffle(starts)
    
    threads = []
    stop_event = threading.Event()
    
    lock = threading.Lock()
    global_best = {'energy': best_energy, 'positions': best_positions.copy()}

    def worker_wrapper(start_pos, tid):
        local_best = {'energy': np.inf, 'positions': None}
        _bh_worker(start_pos, rho, size, tid, stop_event, local_best)
        with lock:
            if local_best['energy'] < global_best['energy']:
                global_best['energy'] = local_best['energy']
                global_best['positions'] = local_best['positions'].copy()

    for i in range(num_threads):
        start_pos = starts[i % len(starts)]
        t = threading.Thread(target=worker_wrapper, args=(start_pos, i))
        t.daemon = True
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if global_best['positions'] is not None:
        return global_best['positions']
    else:
        return best_positions
