# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

TOTAL_DEADLINE = 295.0
_START_TIME = time.time()

def _time_left():
    return max(0.0, TOTAL_DEADLINE - (time.time() - _START_TIME))

def _morse_energy_fast(positions, rho):
    """Vectorized Morse energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    iu = np.triu_indices(n, k=1)
    return np.sum(e[iu])

def _morse_gradient_fast(positions, rho):
    """Vectorized Morse gradient."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    dE_dr = 2.0 * rho * ex * (ex - 1.0)
    np.fill_diagonal(dE_dr, 0.0)
    safe_dists = np.where(dists > 1e-10, dists, 1e-10)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i+1, n):
            factor = dE_dr[i, j] / safe_dists[i, j]
            grad[i] += factor * diff[i, j]
            grad[j] -= factor * diff[i, j]
    return grad

def _enforce_min_distance(positions, min_dist=0.5):
    """Nudge atoms apart if they are too close."""
    n = len(positions)
    positions = positions.copy()
    for _ in range(20):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    if d < 1e-10:
                        diff = np.array([0.01, 0, 0])
                    else:
                        diff = (positions[i] - positions[j]) / d
                    shift = (min_dist - d + 0.01) / 2.0
                    positions[i] += diff * shift
                    positions[j] -= diff * shift
                    moved = True
        if not moved:
            break
    return positions

def _make_fcc_cluster(n, spacing=1.0):
    side = int(np.ceil(n ** (1/3))) + 2
    points = []
    a = spacing * np.sqrt(2)
    basis = np.array([[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]) * a
    for ix in range(-side, side+1):
        for iy in range(-side, side+1):
            for iz in range(-side, side+1):
                for b in basis:
                    pt = np.array([ix, iy, iz]) * a + b
                    if np.linalg.norm(pt) <= side * a * 0.8:
                        points.append(pt)
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _make_bcc_cluster(n, spacing=1.0):
    side = int(np.ceil(n ** (1/3))) + 2
    points = []
    a = spacing * (2.0 / np.sqrt(3.0))
    basis = np.array([[0,0,0], [0.5,0.5,0.5]]) * a
    for ix in range(-side, side+1):
        for iy in range(-side, side+1):
            for iz in range(-side, side+1):
                for b in basis:
                    pt = np.array([ix, iy, iz]) * a + b
                    if np.linalg.norm(pt) <= side * a * 0.8:
                        points.append(pt)
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _make_hcp_cluster(n, spacing=1.0):
    a = spacing
    c = a * np.sqrt(8/3)
    points = []
    radius = int(np.ceil(np.sqrt(n / 3))) + 2
    for layer in range(2 * radius + 1):
        z = (layer - radius) * c
        for q in range(-radius, radius + 1):
            for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1):
                x = a * (q + r/2)
                y = a * (np.sqrt(3)/2) * r
                if layer % 2 == 0:
                    points.append([x, y, z])
                else:
                    points.append([x + a/2, y + a*np.sqrt(3)/6, z])
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _make_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms
    return v

def _make_icosahedral_cluster(n, spacing=1.0):
    """Build an icosahedral cluster: 1 center + 12 shell + more shells."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    vertices = _make_icosahedron_vertices()
    R1 = spacing * (1 + np.sqrt(5)) / 4.0
    
    points = [np.array([0.0, 0.0, 0.0])]
    for v in vertices:
        points.append(v * R1)
    
    if n > 13:
        octa = np.array([
            [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
        ], dtype=float)
        R2 = R1 * np.sqrt(2) * 0.95
        for v in octa:
            if len(points) >= n:
                break
            points.append(v * R2)
    
    if len(points) < n:
        rng = np.random.default_rng(123)
        while len(points) < n:
            d = rng.normal(0, 1, 3)
            d = d / np.linalg.norm(d)
            r = R1 * (1.0 + 0.1 * rng.random())
            points.append(d * r)
    
    points = np.array(points[:n])
    return points

def _make_decahedral_cluster(n, spacing=1.0):
    """Build a decahedral cluster (twinned fcc)."""
    half = n // 2
    frag1 = _make_fcc_cluster(half, spacing)
    frag2 = _make_fcc_cluster(n - half, spacing)
    frag2 = frag2 * np.array([1, 1, -1])
    frag2 += np.array([0, 0, 0.5])
    combined = np.vstack([frag1, frag2])
    combined -= combined.mean(axis=0)
    return combined[:n]

def _make_random_compact(n, rng, spacing=1.0):
    radius = spacing * n ** (1/3) * 1.2
    points = []
    attempts = 0
    while len(points) < n and attempts < 5000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0:
                points.append(p)
            else:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.8 * spacing:
                    points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _make_spherical_shell(n, spacing=1.0):
    points = []
    if n == 1:
        return np.array([[0,0,0]])
    golden = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z*z)
        phi = 2 * np.pi * i / golden
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x, y, z])
    points = np.array(points)
    scale = 1.5 * (n ** (1/3)) / 3.0 
    points *= scale
    return points

def _lbfgs_refine(pos, rho, max_iter=300, time_budget=8.0):
    """Refine a configuration using L-BFGS-B with explicit gradient."""
    x0 = pos.flatten()
    
    def objective(x):
        positions = x.reshape(-1, 3)
        return _morse_energy_fast(positions, rho)
    
    def gradient(x):
        positions = x.reshape(-1, 3)
        return _morse_gradient_fast(positions, rho).flatten()
    
    try:
        start_t = time.time()
        result = minimize(
            objective, x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-7}
        )
        elapsed = time.time() - start_t
        if elapsed > time_budget:
            return np.inf, pos
        
        if result.success or result.fun < np.inf:
            refined = result.x.reshape(-1, 3)
            refined = _enforce_min_distance(refined, 0.5)
            final_e = _morse_energy_fast(refined, rho)
            return final_e, refined
    except Exception:
        pass
    return np.inf, pos

def _relax_config(pos, rho):
    """Relax using both L-BFGS-B and the provided relax function, take best."""
    best_e = np.inf
    best_pos = pos
    
    e1, p1 = _lbfgs_refine(pos, rho, max_iter=200, time_budget=6.0)
    if e1 < best_e:
        best_e = e1
        best_pos = p1
    
    try:
        energy, relaxed_pos = relax(pos.copy(), 'morse', rho)
        relaxed_pos = _enforce_min_distance(relaxed_pos, 0.5)
        final_energy = _morse_energy_fast(relaxed_pos, rho)
        if final_energy < best_e:
            best_e = final_energy
            best_pos = relaxed_pos
    except Exception:
        pass
    
    return best_e, best_pos

def _simulated_annealing(pos, rho, time_budget=10.0):
    """Simulated annealing to escape local minima."""
    current_pos = pos.copy()
    current_e = _morse_energy_fast(current_pos, rho)
    best_pos = current_pos.copy()
    best_e = current_e
    
    T = 1.0
    cooling = 0.95
    step_size = 0.1
    n = len(current_pos)
    
    start_t = time.time()
    rng = np.random.default_rng(42)
    
    while time.time() - start_t < time_budget and T > 0.001:
        # Random perturbation
        delta = rng.normal(0, step_size, size=current_pos.shape)
        new_pos = current_pos + delta
        new_e = _morse_energy_fast(new_pos, rho)
        
        # Accept if better or with Boltzmann probability
        if new_e < current_e or rng.random() < np.exp(-(new_e - current_e) / T):
            current_pos = new_pos
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best_pos = current_pos.copy()
        
        T *= cooling
        step_size *= 0.99
    
    return best_e, best_pos

def construct(instance):
    """
    Find low-energy Morse cluster configurations using diverse seeds and gradient-based refinement.
    """
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    best_energy = np.inf
    best_positions = None
    
    rng = np.random.default_rng(42)
    
    initial_configs = []
    
    # Physically-motivated seeds
    initial_configs.append(_make_icosahedral_cluster(n))
    initial_configs.append(_make_decahedral_cluster(n))
    initial_configs.append(_make_fcc_cluster(n))
    initial_configs.append(_make_bcc_cluster(n))
    initial_configs.append(_make_hcp_cluster(n))
    initial_configs.append(_make_spherical_shell(n))
    
    # Random compact
    initial_configs.append(_make_random_compact(n, rng))
    
    # Jittered versions of key seeds
    for base in initial_configs[:3]:
        jitter = rng.normal(0, 0.15, size=base.shape)
        initial_configs.append(base + jitter)
    
    # More random starts
    for i in range(2):
        initial_configs.append(_make_random_compact(n, rng))
    
    # Shuffle
    rng.shuffle(initial_configs)
    
    time_limit = _time_left()
    if time_limit < 5:
        for pos in initial_configs:
            e = _morse_energy_fast(pos, rho)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        return _enforce_min_distance(best_positions, 0.5)
    
    # Phase 1: Parallel gradient-based relaxation
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_config, pos, rho): pos for pos in initial_configs}
        
        for future in concurrent.futures.as_completed(futures):
            if _time_left() < 10.0:
                break
            try:
                energy, pos = future.result(timeout=3.0)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                continue
    
    if best_positions is None:
        best_positions = initial_configs[0]
        best_energy = _morse_energy_fast(best_positions, rho)
    
    # Phase 2: Simulated annealing on the best configuration found
    if _time_left() > 5.0:
        sa_time = min(15.0, _time_left() - 5.0)
        sa_e, sa_pos = _simulated_annealing(best_positions.copy(), rho, time_budget=sa_time)
        if sa_e < best_energy:
            best_energy = sa_e
            best_positions = sa_pos
    
    # Phase 3: Final refinement with more iterations
    if _time_left() > 3.0:
        final_e, final_pos = _lbfgs_refine(best_positions, rho, max_iter=500, time_budget=_time_left() - 3.0)
        if final_e < best_energy:
            best_energy = final_e
            best_positions = final_pos
    
    best_positions = _enforce_min_distance(best_positions, 0.5)
    return best_positions
# EVOLVE-BLOCK-END
