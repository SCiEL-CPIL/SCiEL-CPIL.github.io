"""Morse cluster global minima: Parallel basin hopping with diverse crystalline and icosahedral starts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # 330s kill, keep 10s safety margin


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient using vectorized operations."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r_sq = np.sum(diff * diff, axis=2)
    r = np.sqrt(r_sq + 1e-12)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_masked = r[mask]
    diff_masked = diff[mask]
    
    if len(r_masked) == 0:
        return 0.0, np.zeros_like(positions)
    
    r_safe = np.where(r_masked < 1e-12, 1e-12, r_masked)
    
    exp_term = np.exp(rho * (1.0 - r_safe))
    pair_e = exp_term * (exp_term - 2.0)
    energy = np.sum(pair_e)
    
    dE_dr = exp_term * (2.0 * exp_term - 2.0) * (-rho)
    factor = dE_dr / r_safe
    
    grad = np.zeros_like(positions)
    i_idx, j_idx = np.where(mask)
    np.add.at(grad, i_idx, factor[:, np.newaxis] * diff_masked)
    np.add.at(grad, j_idx, -factor[:, np.newaxis] * diff_masked)
    
    return energy, grad


def _relax_lbfgs(positions, rho, maxiter=100):
    """Relax using L-BFGS-B with Morse energy and gradient."""
    def objective_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20})
        return result.fun, result.x.reshape(-1, 3)
    except Exception:
        # Fallback to provided relax if L-BFGS fails
        try:
            energy, pos = relax(positions, 'morse', rho)
            return energy, pos
        except Exception:
            return np.inf, positions


def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like cluster."""
    lattice = []
    for i in range(-8, 9):
        for j in range(-8, 9):
            for k in range(-8, 9):
                lattice.append([i * spacing, j * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    lattice.sort(key=lambda p: np.linalg.norm(p))
    points = lattice[:n]
    return np.array(points, dtype=float)


def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like cluster."""
    lattice = []
    for i in range(-8, 9):
        for j in range(-8, 9):
            for k in range(-8, 9):
                lattice.append([i * spacing, j * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, j * spacing, (k + 0.5) * spacing])
                lattice.append([i * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    unique = []
    seen = set()
    for p in lattice:
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    unique.sort(key=lambda p: np.linalg.norm(p))
    points = unique[:n]
    return np.array(points, dtype=float)


def _generate_hcp(n, spacing=1.0):
    """Generate HCP-like cluster."""
    lattice = []
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    for i in range(-8, 9):
        for j in range(-8, 9):
            for k in range(-8, 9):
                # Layer 1: hexagonal
                x1 = a * (i + 0.5 * j)
                y1 = a * (np.sqrt(3.0) / 2.0 * j)
                z1 = c * k
                lattice.append([x1, y1, z1])
                # Layer 2: hexagonal shifted
                x2 = a * (i + 0.5 * j + 1.0/3.0)
                y2 = a * (np.sqrt(3.0) / 2.0 * j + 1.0 / np.sqrt(3.0))
                z2 = c * (k + 0.5)
                lattice.append([x2, y2, z2])
    
    unique = []
    seen = set()
    for p in lattice:
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    unique.sort(key=lambda p: np.linalg.norm(p))
    points = unique[:n]
    return np.array(points, dtype=float)


def _generate_icosahedral(n, spacing=1.0):
    """Generate icosahedral shell cluster."""
    points = []
    # Icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = [
        (0, 1, phi), (0, 1, -phi), (0, -1, phi), (0, -1, -phi),
        (1, phi, 0), (1, -phi, 0), (-1, phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
    ]
    verts = np.array(verts)
    verts /= np.linalg.norm(verts[0])
    
    # Generate points on icosahedron surface
    radius = spacing * n ** (1.0 / 3.0) * 0.5
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        radius_xy = np.sqrt(1.0 - z * z)
        theta = 2.0 * np.pi * i / golden_ratio
        x = radius_xy * np.cos(theta)
        y = radius_xy * np.sin(theta)
        points.append([x * radius, y * radius, z * radius])
    
    return np.array(points, dtype=float)


def _generate_random_compact(n, spacing=1.0, seed=42):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.6
    points = []
    for _ in range(n):
        while True:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                # Check distance to existing points
                if len(points) == 0:
                    break
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) > 0.6:
                    break
                # If too close, try again or accept if radius is large
                if rng.random() < 0.1:
                    break
        points.append(p)
    return np.array(points, dtype=float)


def _basin_hop(positions, rho, n_iter=50, step_size=0.3, rng=None, deadline=None):
    """
    Basin hopping: perturb, relax, accept/reject.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(positions)
    current_energy, current_pos = _relax_lbfgs(positions, rho)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    T = 1.0
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb a subset of atoms
        n_perturb = max(1, n // 5)
        indices = rng.choice(n, n_perturb, replace=False)
        
        new_pos = current_pos.copy()
        new_pos[indices] += rng.normal(0, step_size, (n_perturb, 3))
        
        new_energy, new_pos_relaxed = _relax_lbfgs(new_pos, rho)
        
        # Check validity
        if len(new_pos_relaxed) > 1:
            diff = new_pos_relaxed[:, np.newaxis, :] - new_pos_relaxed[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dist, np.inf)
            if np.min(dist) < 0.5 - 1e-6:
                continue
        
        # Metropolis criterion
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-9)):
            current_pos = new_pos_relaxed
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        
        T *= 0.98
        if T < 0.1:
            break
    
    return best_energy, best_pos


def _run_basin_hop(args):
    """Wrapper for parallel execution."""
    positions, rho, n_iter, step_size, seed, deadline = args
    rng = np.random.default_rng(seed)
    try:
        return _basin_hop(positions, rho, n_iter=n_iter, step_size=step_size, rng=rng, deadline=deadline)
    except Exception:
        return np.inf, positions


def construct(instance):
    """Parallel basin hopping from diverse starts."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    deadline = _START_TIME + _TIME_LIMIT - 10
    
    if time.time() > deadline:
        return np.zeros((size, 3))
    
    # Generate diverse starting points
    starts = []
    try:
        starts.append(_generate_bcc(size, 1.0))
    except Exception:
        pass
    try:
        starts.append(_generate_fcc(size, 1.0))
    except Exception:
        pass
    try:
        starts.append(_generate_hcp(size, 1.0))
    except Exception:
        pass
    try:
        starts.append(_generate_icosahedral(size, 1.0))
    except Exception:
        pass
    try:
        starts.append(_generate_random_compact(size, 1.0, seed=1))
    except Exception:
        pass
    try:
        starts.append(_generate_random_compact(size, 1.0, seed=2))
    except Exception:
        pass
    
    if not starts:
        return np.zeros((size, 3))
    
    # Prepare arguments for parallel basin hopping
    # Use 4 threads max
    n_threads = min(4, len(starts))
    tasks = []
    for i, start_pos in enumerate(starts[:n_threads]):
        tasks.append((start_pos, rho, 30, 0.3, i + 42, deadline))
    
    best_energy = np.inf
    best_pos = starts[0]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(_run_basin_hop, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                e, p = future.result()
                if e < best_energy:
                    best_energy = e
                    best_pos = p
            except Exception:
                pass
    
    # Final relaxation
    final_energy, final_pos = _relax_lbfgs(best_pos, rho, maxiter=100)
    
    return final_pos
# EVOLVE-BLOCK-END
