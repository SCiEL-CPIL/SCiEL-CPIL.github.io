"""Morse cluster global minima search with integrated Basin Hopping and L-BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import morse_energy_gradient

# Global deadline shared across all six construct() calls
_START_TIME = time.time()
_TOTAL_BUDGET = 310.0  # 20s safety margin before 330s kill

def _time_left():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)

def _morse_energy(pos, rho):
    """Compute total Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    e = np.exp(rho * (1.0 - d))
    return float(np.sum(e * (e - 2.0)))

def _morse_gradient(pos, rho):
    """Compute Morse gradient."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid division by zero
    dist[dist == 0] = 1e-6
    # Energy gradient: dE/dr * dr/dpos
    # E = e*(e-2), dE/de = 2e-2, de/dr = -rho*e
    # dE/dr = -rho*e*(2e-2)
    # dE/dpos_i = sum_{j!=i} dE/dr_ij * (pos_i - pos_j)/r_ij
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    e = np.exp(rho * (1.0 - d))
    de_dr = -rho * e * (2 * e - 2)
    # Vectorized gradient accumulation
    grad = np.zeros_like(pos)
    # diff[iu] is pos[i] - pos[j]
    # grad[i] += sum_j de_dr[i,j] * (pos[i]-pos[j])/d[i,j]
    # grad[j] -= sum_i de_dr[i,j] * (pos[i]-pos[j])/d[i,j]
    
    # Use scatter add logic via numpy advanced indexing or simple loop for clarity if n is small, 
    # but for n=80, loop is fine (3160 pairs).
    # Actually, let's use the provided function if it's fast, but the prompt says we can use it.
    # However, to be safe and fast, let's stick to the vectorized manual calc if possible, 
    # or just call the provided one if it's C-optimized. 
    # The prompt says: chem_clusters.morse_energy_gradient(positions, rho) may be called.
    # Let's try to use it first, but fallback to manual if it fails or is slow.
    try:
        return morse_energy_gradient(pos, rho)
    except Exception:
        # Fallback manual
        for i, j in zip(iu[0], iu[1]):
            rij = pos[i] - pos[j]
            factor = de_dr[i, j] / d[i, j]
            grad[i] += factor * rij
            grad[j] -= factor * rij
        return grad

def _build_fcc_cluster(n, spacing=1.0):
    """Build an FCC-like compact cluster."""
    points = []
    # Generate FCC lattice points
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * spacing * 0.7071
                    if np.linalg.norm(p) <= 4.0 * spacing:
                        points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _build_bcc_cluster(n, spacing=1.0):
    """Build a BCC-like compact cluster."""
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                p = np.array([i, j, k]) * spacing * 0.5
                if np.linalg.norm(p) <= 3.5 * spacing:
                    points.append(p)
                # BCC center
                if (i+j+k) % 2 == 0:
                    pc = np.array([i+0.5, j+0.5, k+0.5]) * spacing * 0.5
                    if np.linalg.norm(pc) <= 3.5 * spacing:
                        points.append(pc)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _build_icosahedral_shells(n, spacing=1.0):
    """Build cluster using icosahedral shells."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico12 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico12 /= np.linalg.norm(ico12[0])
    
    points = [np.zeros(3)]
    # Shell 1: 12 atoms at radius ~1.0 (equilibrium)
    for v in ico12:
        points.append(v * 1.0)
    
    # Shell 2: 30 atoms? Or just add more icosahedral points scaled
    # For simplicity, add more points on a larger shell
    shell_radius = 2.0
    rng = np.random.default_rng(42)
    while len(points) < n:
        # Add points on spherical shell
        p = rng.normal(0, 1, 3)
        p = p / np.linalg.norm(p) * shell_radius
        # Check min distance
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.6:
                valid = False
                break
        if valid:
            points.append(p)
        else:
            shell_radius += 0.2
            if shell_radius > 5.0:
                 points.append(p) # Force add to avoid infinite loop
    return np.array(points[:n])

def _build_random_compact(n, spacing=1.0, seed=42):
    """Build a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    max_attempts = 5000
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
            points.append(p)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])

def _ensure_min_distance(pos, min_dist=0.5, max_iter=100):
    """Nudge atoms apart if they are too close."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-10:
                        diff = np.array([1.0, 0, 0])
                    diff /= np.linalg.norm(diff)
                    push = (min_dist - d) * 0.5
                    pos[i] += diff * push
                    pos[j] -= diff * push
                    moved = True
        if not moved:
            break
    return pos

def _local_relax(pos, potential, param, time_limit=5.0):
    """Local relaxation using L-BFGS-B."""
    start = time.time()
    
    def objective(x):
        if time.time() - start > time_limit:
            raise TimeoutError
        p = x.reshape(-1, 3)
        return _morse_energy(p, param)
        
    def gradient(x):
        if time.time() - start > time_limit:
            raise TimeoutError
        p = x.reshape(-1, 3)
        return _morse_gradient(p, param).flatten()
        
    try:
        res = minimize(objective, pos.flatten(), jac=gradient, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success:
            return res.fun, res.x.reshape(-1, 3)
        else:
            # Return best found so far
            p = res.x.reshape(-1, 3)
            return _morse_energy(p, param), p
    except TimeoutError:
        return _morse_energy(pos, param), pos
    except Exception:
        return _morse_energy(pos, param), pos

def _basin_hopping(pos, potential, param, n_restarts=10, step_size=0.5, time_budget=30.0):
    """Basin Hopping algorithm."""
    best_energy, best_pos = _local_relax(pos, potential, param, time_limit=time_budget/10)
    current_energy = best_energy
    current_pos = best_pos
    start = time.time()
    rng = np.random.default_rng(123)
    
    for _ in range(n_restarts):
        if time.time() - start > time_budget:
            break
        
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        perturbed = _ensure_min_distance(perturbed)
        
        # Relax
        e, p = _local_relax(perturbed, potential, param, time_limit=time_budget/10)
        
        # Metropolis accept
        if e < current_energy:
            current_energy = e
            current_pos = p
            if e < best_energy:
                best_energy = e
                best_pos = p
        else:
            # Accept with probability exp(-(e - current_energy)/T)
            # Temperature T can be adapted, but let's keep it simple
            T = 0.1
            if rng.random() < np.exp(-(e - current_energy) / T):
                current_energy = e
                current_pos = p
                
    return best_energy, best_pos

def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    tl = _time_left()
    if tl < 5.0:
        # Emergency fallback
        pos = _build_random_compact(size, seed=size)
        return _ensure_min_distance(pos)
    
    # Generate diverse starting configurations
    start_configs = []
    try:
        start_configs.append(_build_fcc_cluster(size))
    except Exception:
        pass
    try:
        start_configs.append(_build_bcc_cluster(size))
    except Exception:
        pass
    try:
        start_configs.append(_build_icosahedral_shells(size))
    except Exception:
        pass
    try:
        start_configs.append(_build_random_compact(size, seed=size))
    except Exception:
        pass
    try:
        start_configs.append(_build_random_compact(size, seed=size+100))
    except Exception:
        pass
    
    # Allocate time budget
    # We have 5 configs. Give each a share of the time.
    # Reserve 10% for final polish.
    search_time = tl * 0.9
    time_per_config = search_time / len(start_configs)
    
    best_energy = float('inf')
    best_pos = start_configs[0]
    
    for i, config in enumerate(start_configs):
        tl = _time_left()
        if tl < 10.0:
            break
        
        # Run Basin Hopping on this config
        # Adjust step size based on rho? Higher rho -> steeper -> smaller step?
        # Let's use a default step size of 0.5
        e, p = _basin_hopping(config, potential, parameter, 
                              n_restarts=20, 
                              step_size=0.5, 
                              time_budget=time_per_config)
        
        if e < best_energy:
            best_energy = e
            best_pos = p
            
    # Final high-precision polish
    tl = _time_left()
    if tl > 5.0:
        # Do a very tight local relaxation on the best found
        e, p = _local_relax(best_pos, potential, parameter, time_limit=tl - 2.0)
        if e < best_energy:
            best_energy = e
            best_pos = p
            
    # Ensure constraints
    best_pos = _ensure_min_distance(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END
