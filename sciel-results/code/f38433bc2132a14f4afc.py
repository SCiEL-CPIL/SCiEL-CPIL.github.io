"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _time_remaining(deadline, margin=5.0):
    return deadline - time.time() - margin


def _min_pair_dist(positions):
    """Compute minimum pairwise distance efficiently."""
    n = len(positions)
    min_d = np.inf
    for i in range(n):
        diffs = positions[i+1:] - positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        if len(dists) > 0:
            d = np.min(dists)
            if d < min_d:
                min_d = d
    return min_d


def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    if _min_pair_dist(positions) < min_dist:
        return False
    return True


def _generate_fcc(n, spacing=0.72, seed=0):
    """Generate n points from FCC lattice closest to origin."""
    rng = np.random.default_rng(seed)
    a = spacing
    basis = np.array([[0,0,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]], dtype=float) * a
    
    max_coord = int(np.ceil(n ** (1.0/3.0))) + 3
    candidates = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                center = np.array([ix, iy, iz]) * a
                for b in basis:
                    candidates.append(center + b)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    positions = candidates[order[:n]].copy()
    positions += rng.uniform(-0.01, 0.01, positions.shape)
    return positions


def _generate_hcp(n, spacing=0.72, seed=0):
    """Generate n points from HCP lattice closest to origin."""
    rng = np.random.default_rng(seed)
    a = spacing
    sqrt3 = np.sqrt(3.0)
    c = a * np.sqrt(8.0 / 3.0)
    
    max_layer = int(np.ceil(n ** (1.0/3.0))) + 3
    candidates = []
    
    for layer in range(-max_layer, max_layer + 1):
        z = layer * c
        for ix in range(-max_layer, max_layer + 1):
            for iy in range(-max_layer, max_layer + 1):
                x = a * (ix + 0.5 * (iy % 2))
                y = a * sqrt3 / 2.0 * iy
                if layer % 2 != 0:
                    x += a / 2.0
                    y += a * sqrt3 / 6.0
                candidates.append(np.array([x, y, z]))
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    positions = candidates[order[:n]].copy()
    positions += rng.uniform(-0.01, 0.01, positions.shape)
    return positions


def _generate_bcc(n, spacing=0.72, seed=0):
    """Generate n points from BCC lattice closest to origin."""
    rng = np.random.default_rng(seed)
    a = spacing
    basis = np.array([[0,0,0],[0.5,0.5,0.5]], dtype=float) * a
    
    max_coord = int(np.ceil(n ** (1.0/3.0))) + 3
    candidates = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                center = np.array([ix, iy, iz]) * a
                for b in basis:
                    candidates.append(center + b)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    positions = candidates[order[:n]].copy()
    positions += rng.uniform(-0.01, 0.01, positions.shape)
    return positions


def _generate_random_compact(n, spacing=0.72, seed=0):
    """Generate random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    min_required = 0.7 * spacing
    
    while len(points) < n and attempts < 20000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if not points or min(np.linalg.norm(p - q) for q in points) >= min_required:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _generate_icosahedral(n, spacing=0.72, seed=0):
    """Generate positions based on icosahedral shells."""
    rng = np.random.default_rng(seed)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    ico_verts = verts / norms * spacing
    
    positions = [np.array([0.0, 0.0, 0.0])]
    remaining = n - 1
    
    if remaining >= 12:
        positions.extend(ico_verts.tolist())
        remaining -= 12
    
    # Add more shells using Fibonacci spheres
    shell_radius = spacing * 1.5
    while remaining > 0:
        n_on_shell = min(remaining, 20 + int(len(positions) * 0.5))
        if n_on_shell <= 0:
            n_on_shell = remaining
        
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        y = np.linspace(1.0, -1.0, n_on_shell)
        radius_per_y = np.sqrt(1.0 - y**2)
        theta = golden_angle * np.arange(n_on_shell)
        x = np.cos(theta) * radius_per_y
        z = np.sin(theta) * radius_per_y
        pts = np.column_stack([x, y, z]) * shell_radius
        pts += rng.uniform(-0.03, 0.03, pts.shape)
        positions.extend(pts.tolist())
        remaining -= n_on_shell
        shell_radius += spacing * 0.5
    
    return np.array(positions[:n])


def _basin_hopping(positions, potential, kind, deadline, n_iterations=50, step_scale=0.15, seed=0):
    """Perform basin hopping on the given positions."""
    rng = np.random.default_rng(seed)
    current_energy, current_pos = sutton_chen_energy_gradient(positions, kind)
    best_energy, best_pos = current_energy, current_pos.copy()
    
    for it in range(n_iterations):
        if _time_remaining(deadline, 3.0) <= 0:
            break
        
        # Perturb
        perturbation = rng.normal(0, step_scale, current_pos.shape)
        trial_pos = current_pos + perturbation
        
        # Relax trial
        try:
            trial_energy, trial_pos = relax(trial_pos, potential, kind)
        except Exception:
            continue
        
        if not _is_valid(trial_pos):
            continue
        
        # Metropolis acceptance
        if trial_energy < best_energy:
            best_energy, best_pos = trial_energy, trial_pos.copy()
        
        delta = trial_energy - current_energy
        if delta < 0 or rng.random() < np.exp(-delta / 0.5):
            current_energy, current_pos = trial_energy, trial_pos
        
        # Adapt step size
        if trial_energy < current_energy:
            step_scale *= 1.05
        else:
            step_scale *= 0.95
        step_scale = max(0.05, min(0.5, step_scale))
    
    return best_energy, best_pos


def _high_precision_polish(positions, potential, kind, deadline):
    """Do a final high-precision relaxation with tighter tolerance."""
    if _time_remaining(deadline, 2.0) <= 0:
        return positions
    
    try:
        energy, grad = sutton_chen_energy_gradient(positions, kind)
        result = minimize(
            lambda x: sutton_chen_energy_gradient(x.reshape(-1, 3), kind)[0],
            positions.flatten(),
            jac=lambda x: sutton_chen_energy_gradient(x.reshape(-1, 3), kind)[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        polished = result.x.reshape(-1, 3)
        if _is_valid(polished):
            return polished
    except Exception:
        pass
    return positions


def construct(instance):
    """Multi-start with diverse structural templates, parallel basin hopping."""
    start_time = time.time()
    total_budget = 300.0
    deadline = start_time + total_budget
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    spacing = 0.72
    
    # Generate diverse starting structures
    generators = [
        (lambda s: _generate_fcc(n, spacing, seed=s), "fcc"),
        (lambda s: _generate_hcp(n, spacing, seed=s), "hcp"),
        (lambda s: _generate_bcc(n, spacing, seed=s), "bcc"),
        (lambda s: _generate_random_compact(n, spacing, seed=s), "rand"),
        (lambda s: _generate_icosahedral(n, spacing, seed=s), "ico"),
    ]
    
    seeds_per_generator = 4
    all_starts = []
    
    for gen_func, gen_name in generators:
        for seed in range(seeds_per_generator):
            if _time_remaining(deadline, 10.0) <= 0:
                break
            try:
                positions = gen_func(seed * 1000 + 42)
                if positions.shape == (n, 3) and _is_valid(positions):
                    all_starts.append(positions)
            except Exception:
                continue
    
    # Add more random starts
    for seed in range(50):
        if _time_remaining(deadline, 10.0) <= 0:
            break
        try:
            positions = _generate_random_compact(n, spacing, seed=seed + 1000)
            if positions.shape == (n, 3) and _is_valid(positions):
                all_starts.append(positions)
        except Exception:
            continue
    
    # Limit number of starts to keep within time budget
    max_starts = min(len(all_starts), 40)
    all_starts = all_starts[:max_starts]
    
    # Parallel basin hopping using 4 threads
    best_energy = np.inf
    best_positions = None
    lock = __import__('threading').Lock()
    
    def worker(start_pos, worker_id):
        nonlocal best_energy, best_positions
        try:
            # First, do a quick relaxation
            energy, pos = relax(start_pos, potential, kind)
            if not _is_valid(pos):
                return None, None
            
            # Then basin hopping
            bh_energy, bh_pos = _basin_hopping(pos, potential, kind, deadline, 
                                                n_iterations=40, step_scale=0.12, 
                                                seed=worker_id)
            
            if bh_energy < best_energy:
                with lock:
                    if bh_energy < best_energy:
                        best_energy = bh_energy
                        best_positions = bh_pos.copy()
            
            return bh_energy, bh_pos
        except Exception:
            return None, None
    
    # Use ThreadPoolExecutor with 4 workers
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, start_pos in enumerate(all_starts):
            if _time_remaining(deadline, 5.0) <= 0:
                break
            futures.append(executor.submit(worker, start_pos, i))
        
        for future in as_completed(futures):
            if _time_remaining(deadline, 5.0) <= 0:
                break
            try:
                future.result()
            except Exception:
                pass
    
    # Final polish with high-precision relaxation
    if best_positions is not None and _time_remaining(deadline, 10.0) > 0:
        best_positions = _high_precision_polish(best_positions, potential, kind, deadline)
    
    # Fallback
    if best_positions is None:
        best_positions = _generate_fcc(n, spacing, seed=0)
        try:
            _, best_positions = relax(best_positions, potential, kind)
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
