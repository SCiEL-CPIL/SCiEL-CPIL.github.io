"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _icosahedron_vertices():
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _fib_sphere(n_pts, radius, rng):
    """Generate n_pts points on a sphere of given radius using Fibonacci lattice."""
    if n_pts <= 0:
        return np.empty((0, 3))
    if n_pts == 1:
        return np.array([[0.0, 0.0, radius]])
    
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    y = np.linspace(1.0, -1.0, n_pts)
    radius_per_y = np.sqrt(1.0 - y**2)
    theta = golden_angle * np.arange(n_pts)
    
    x = np.cos(theta) * radius_per_y
    z = np.sin(theta) * radius_per_y
    
    perturbation = rng.uniform(-0.05, 0.05, (n_pts, 3))
    points = np.column_stack([x, y, z]) * radius + perturbation
    
    return points


def _generate_icosahedral(n, spacing=0.72, seed=0):
    """Generate positions based on icosahedral shells."""
    rng = np.random.default_rng(seed)
    ico_verts = _icosahedron_vertices() * spacing
    
    positions = [np.array([0.0, 0.0, 0.0])]
    remaining = n - 1
    
    if remaining >= 12:
        positions.extend(ico_verts.tolist())
        remaining -= 12
    
    if remaining >= 30:
        ico_idx = [
            (0,1),(0,3),(0,4),(0,8),(1,2),(1,4),(1,6),
            (2,3),(2,5),(2,8),(3,5),(3,9),(4,6),(4,9),
            (4,10),(5,7),(5,10),(5,11),(6,7),(6,10),
            (7,11),(8,9),(8,11),(9,10),(10,11),(0,2),(1,5),(2,7),(3,6),(7,9)
        ]
        for i, j in ico_idx:
            mid = (ico_verts[i] + ico_verts[j]) / 2.0
            mid = mid * (spacing * np.sqrt(2.0) / np.linalg.norm(mid))
            positions.append(mid)
        take = min(30, remaining)
        positions = positions[:len(positions) - remaining + take] if remaining < 30 else positions
        remaining -= take
    
    while remaining > 0:
        shell_radius = spacing * (2.0 + 0.3 * (len(positions) // 12))
        n_on_shell = min(remaining, 12 + len(positions) // 4)
        shell_pts = _fib_sphere(n_on_shell, shell_radius, rng)
        positions.extend(shell_pts.tolist())
        remaining -= n_on_shell
    
    return np.array(positions[:n])


def _generate_fcc(n, spacing=0.72, seed=0):
    """Generate positions based on FCC-like packing."""
    rng = np.random.default_rng(seed)
    a = spacing
    basis = np.array([
        [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]
    ]) * a
    
    max_coord = int(np.ceil(n ** (1.0/3.0))) + 3
    candidates = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                center = np.array([ix, iy, iz]) * a
                for b in basis:
                    p = center + b
                    candidates.append(p)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    positions = candidates[order[:n]]
    positions = positions + rng.uniform(-0.02, 0.02, positions.shape)
    return positions


def _generate_hcp(n, spacing=0.72, seed=0):
    """Generate positions based on HCP-like packing."""
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
                p = np.array([x, y, z])
                candidates.append(p)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    positions = candidates[order[:n]]
    positions = positions + rng.uniform(-0.02, 0.02, positions.shape)
    return positions


def _generate_bcc(n, spacing=0.72, seed=0):
    """Generate positions based on BCC-like packing."""
    rng = np.random.default_rng(seed)
    a = spacing
    basis = np.array([
        [0, 0, 0], [0.5, 0.5, 0.5]
    ]) * a
    
    max_coord = int(np.ceil(n ** (1.0/3.0))) + 3
    candidates = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                center = np.array([ix, iy, iz]) * a
                for b in basis:
                    p = center + b
                    candidates.append(p)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    positions = candidates[order[:n]]
    positions = positions + rng.uniform(-0.02, 0.02, positions.shape)
    return positions


def _generate_random_compact(n, spacing=0.72, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    while len(points) < n and attempts < 20000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min([np.linalg.norm(p - q) for q in points], default=np.inf)
            if min_dist >= 0.7 * spacing:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _check_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite and min distance)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True


def _worker(args):
    """Worker function for parallel basin hopping."""
    positions, potential, kind, n_perturb, perturb_scale, seed = args
    rng = np.random.default_rng(seed)
    
    best_e = np.inf
    best_pos = positions.copy()
    
    for step in range(n_perturb):
        # Perturb current best
        perturbed = best_pos + rng.normal(0, perturb_scale, best_pos.shape)
        
        if not _check_valid(perturbed):
            continue
        
        try:
            energy, relaxed = relax(perturbed, potential, kind)
            if energy < best_e:
                best_e = energy
                best_pos = relaxed
        except Exception:
            continue
    
    return best_e, best_pos


def construct(instance):
    """Parallel multi-start basin hopping with diverse structural templates."""
    start_time = time.time()
    total_budget = 300.0
    deadline = start_time + total_budget
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    spacing_map = {"sutton_chen": 0.72, "morse": 1.0, "dzugutov": 1.1}
    spacing = spacing_map.get(potential, 0.72)
    
    generators = [
        (lambda s: _generate_icosahedral(n, spacing, seed=s), "ico"),
        (lambda s: _generate_fcc(n, spacing, seed=s), "fcc"),
        (lambda s: _generate_hcp(n, spacing, seed=s), "hcp"),
        (lambda s: _generate_bcc(n, spacing, seed=s), "bcc"),
        (lambda s: _generate_random_compact(n, spacing, seed=s), "rand"),
    ]
    
    # Number of basin hopping iterations per start
    n_perturb = 8
    perturb_scale = 0.15
    
    # Build task list: multiple seeds per generator
    tasks = []
    for gen_func, gen_name in generators:
        for seed in range(4):
            try:
                positions = gen_func(seed * 1000 + 42)
                if positions.shape != (n, 3):
                    continue
                if not _check_valid(positions):
                    continue
                tasks.append((positions, potential, kind, n_perturb, perturb_scale, seed * 10000 + 123))
            except Exception:
                continue
    
    # Add some purely random starts
    for seed in range(8):
        try:
            positions = _generate_random_compact(n, spacing, seed=5000 + seed)
            if positions.shape == (n, 3) and _check_valid(positions):
                tasks.append((positions, potential, kind, n_perturb, perturb_scale, 90000 + seed))
        except Exception:
            continue
    
    best_energy = np.inf
    best_positions = None
    
    # Run in parallel with 4 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for i, task in enumerate(tasks):
            if time.time() > deadline - 15:
                break
            try:
                fut = executor.submit(_worker, task)
                futures[fut] = i
            except Exception:
                continue
        
        for fut in concurrent.futures.as_completed(futures):
            try:
                energy, pos = fut.result(timeout=30)
                if energy is not None and np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                continue
    
    # If we have time, do sequential refinement on best
    if best_positions is not None and time.time() < deadline - 20:
        try:
            energy, relaxed = relax(best_positions, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            pass
    
    # Fallback
    if best_positions is None:
        best_positions = np.zeros((n, 3))
        best_positions[:, 0] = np.linspace(-n * spacing / 2, n * spacing / 2, n)
    
    return best_positions
# EVOLVE-BLOCK-END
