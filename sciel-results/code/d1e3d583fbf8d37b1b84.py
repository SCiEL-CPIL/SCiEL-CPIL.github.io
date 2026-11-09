"""Dzugutov cluster global minima via multi-start basin hopping with polytetrahedral seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_DEADLINE = time.time() + 275.0  # safety margin before 300s kill
_STOP = threading.Event()

# Dzugutov potential parameters
_A, _B, _a, _b, _c, _d, _m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16.0

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask upper triangle
    iu = np.triu_indices(n, 1)
    r = dists[iu]
    # Avoid division by zero
    r = np.maximum(r, 1e-10)
    energy = 0.0
    # Region 1: r < a
    mask1 = r < _a
    if np.any(mask1):
        r1 = r[mask1]
        energy += np.sum(_A * (r1 ** (-_m) - _B) * np.exp(_c / (r1 - _a)))
    # Region 2: a <= r < b
    mask2 = (r >= _a) & (r < _b)
    if np.any(mask2):
        r2 = r[mask2]
        energy += np.sum(_B * np.exp(_d / (r2 - _b)))
    return float(energy)

def _make_icosahedron_shell(n_atoms, seed=0):
    """Generate atoms on icosahedral shells for polytetrahedral structure."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (12 atoms)
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0]) * 1.0
    
    points = []
    if n_atoms >= 12:
        points.extend(ico_verts.tolist())
        n_atoms -= 12
    
    # Second shell: 30 atoms (truncated icosahedron / fullerene-like)
    if n_atoms > 0:
        # Generate points on a slightly larger shell
        # Use icosahedral symmetry directions
        all_dirs = []
        # 12 icosahedron vertices
        all_dirs.extend(ico_verts.tolist())
        # 20 face centers
        phi2 = phi
        face_centers = np.array([
            [1, 1, 1]/np.sqrt(3), [1, 1, -1]/np.sqrt(3), [1, -1, 1]/np.sqrt(3), [1, -1, -1]/np.sqrt(3),
            [-1, 1, 1]/np.sqrt(3), [-1, 1, -1]/np.sqrt(3), [-1, -1, 1]/np.sqrt(3), [-1, -1, -1]/np.sqrt(3),
            [1, phi2, 0]/np.sqrt(1+phi2**2), [1, -phi2, 0]/np.sqrt(1+phi2**2),
            [-1, phi2, 0]/np.sqrt(1+phi2**2), [-1, -phi2, 0]/np.sqrt(1+phi2**2),
            [phi2, 0, 1]/np.sqrt(1+phi2**2), [phi2, 0, -1]/np.sqrt(1+phi2**2),
            [-phi2, 0, 1]/np.sqrt(1+phi2**2), [-phi2, 0, -1]/np.sqrt(1+phi2**2),
            [0, 1, phi2]/np.sqrt(1+phi2**2), [0, 1, -phi2]/np.sqrt(1+phi2**2),
            [0, -1, phi2]/np.sqrt(1+phi2**2), [0, -1, -phi2]/np.sqrt(1+phi2**2),
        ], dtype=float)
        all_dirs.extend(face_centers.tolist())
        # 30 edge midpoints
        # For now, use a sphere with some random perturbation
        while len(points) < n_atoms + 12:
            if len(all_dirs) > 0:
                idx = len(points) % len(all_dirs)
                d = all_dirs[idx]
                r = 1.9 + rng.normal(0, 0.05)
                p = np.array(d) * r
            else:
                p = rng.normal(0, 0.5, 3)
                p = p / (np.linalg.norm(p) + 1e-10) * (1.9 + rng.normal(0, 0.05))
            # Check minimum distance
            if all(np.linalg.norm(p - np.array(q)) >= 0.55 for q in points):
                points.append(p)
            else:
                all_dirs = all_dirs  # keep trying
                if len(all_dirs) < 200:
                    all_dirs.append(rng.normal(0, 1, 3) / (np.linalg.norm(rng.normal(0, 1, 3)) + 1e-10))
    
    points = points[:n_atoms + 12]
    return np.array(points[:n_atoms]) if len(points) >= n_atoms else None

def _random_cluster(n_atoms, seed=0):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    points = []
    radius = 1.1 * n_atoms ** (1.0/3.0) * 0.7
    attempts = 0
    while len(points) < n_atoms and attempts < n_atoms * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
        attempts += 1
    if len(points) < n_atoms:
        # Fill remaining with random positions
        while len(points) < n_atoms:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points[:n_atoms])

def _basin_hopping(positions, potential, parameter, max_iter=50, step_size=0.15, seed=0):
    """Simple basin hopping: random displacement + local relaxation."""
    rng = np.random.default_rng(seed)
    current = positions.copy()
    current_energy = _dzugutov_energy(current)
    best = current.copy()
    best_energy = current_energy
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        # Random displacement
        noise = rng.normal(0, step_size, current.shape)
        displaced = current + noise
        # Local relaxation
        try:
            energy, relaxed = relax(displaced, potential, parameter)
        except Exception:
            continue
        if energy < best_energy:
            best = relaxed.copy()
            best_energy = energy
        # Metropolis accept
        delta_e = energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.1):
            current = relaxed.copy()
            current_energy = energy
        # Decrease step size slowly
        step_size *= 0.98
    
    return best, best_energy

def _worker(task_queue, results, lock, potential, parameter, n_atoms, seed_base):
    """Worker thread for parallel basin hopping."""
    while not _STOP.is_set():
        try:
            task_id = task_queue.get_nowait()
        except Exception:
            break
        if time.time() > _DEADLINE - 10:
            break
        seed = seed_base + task_id * 1000
        # Generate initial structure
        if task_id % 3 == 0:
            init = _make_icosahedron_shell(n_atoms, seed=seed)
            if init is None:
                init = _random_cluster(n_atoms, seed=seed)
        elif task_id % 3 == 1:
            init = _random_cluster(n_atoms, seed=seed)
        else:
            # Hybrid: start from icosahedron, add random atoms
            ico = _make_icosahedron_shell(min(n_atoms, 20), seed=seed)
            if ico is not None and len(ico) < n_atoms:
                remaining = _random_cluster(n_atoms - len(ico), seed=seed + 1)
                # Scale remaining to fit
                remaining = remaining * 1.5
                init = np.vstack([ico, remaining])[:n_atoms]
            else:
                init = _random_cluster(n_atoms, seed=seed)
        
        # Basin hopping
        try:
            best_pos, best_energy = _basin_hopping(init, potential, parameter, 
                                                    max_iter=30, step_size=0.15, seed=seed)
        except Exception:
            task_queue.task_done()
            continue
        
        with lock:
            if results['best_energy'] is None or best_energy < results['best_energy']:
                results['best_energy'] = best_energy
                results['best_pos'] = best_pos.copy()
        task_queue.task_done()

def construct(instance):
    """Find low-energy Dzugutov cluster via multi-start basin hopping."""
    n_atoms = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    t_start = time.time()
    deadline = t_start + 270.0  # safety margin
    
    # First, get a quick baseline from the seed approach
    rng = np.random.default_rng(n_atoms)
    spacing = 1.1
    radius = spacing * n_atoms ** (1.0/3.0) * 0.75
    best_energy = None
    best_pos = None
    
    # Quick initial relaxations
    for _ in range(3):
        if time.time() > deadline - 30:
            break
        points = []
        while len(points) < n_atoms:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
        try:
            energy, positions = relax(np.array(points), potential, parameter)
            if best_energy is None or energy < best_energy:
                best_energy = energy
                best_pos = positions.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: random
        best_pos = np.random.default_rng(42).normal(0, 1, (n_atoms, 3))
        best_energy = _dzugutov_energy(best_pos)
    
    # Set up parallel basin hopping
    import queue
    task_queue = queue.Queue()
    n_tasks = 20
    for i in range(n_tasks):
        task_queue.put(i)
    
    results = {'best_energy': best_energy, 'best_pos': best_pos.copy()}
    lock = threading.Lock()
    _STOP.clear()
    
    threads = []
    for t in range(4):
        th = threading.Thread(target=_worker, args=(task_queue, results, lock, 
                                                     potential, parameter, n_atoms, t * 10000))
        th.daemon = True
        th.start()
        threads.append(th)
    
    # Wait for threads to finish or deadline
    while not task_queue.empty() and time.time() < deadline - 15:
        time.sleep(0.5)
    
    _STOP.set()
    for th in threads:
        th.join(timeout=5)
    
    # Final best from results
    if results['best_energy'] is not None and results['best_energy'] < best_energy:
        best_energy = results['best_energy']
        best_pos = results['best_pos'].copy()
    
    # One more round of relaxation on the best structure
    if time.time() < deadline - 10:
        try:
            energy, positions = relax(best_pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_pos = positions.copy()
        except Exception:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
