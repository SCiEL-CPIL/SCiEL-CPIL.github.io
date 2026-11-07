"""Sutton-Chen transition-metal cluster global minima search.
Uses a Basin Hopping approach with diverse structural seeds (FCC, BCC, HCP, Icosahedral)
to escape local minima and find the global minimum."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START = time.time()
_DEADLINE = 295.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _make_fcc(n):
    """Generate a compact FCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1.0 / 3.0))) + 2
    a = 1.0
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if len(pts) < n:
                        pts.append(p)
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-mc * a, mc * a, 3)
            pts.append(p)
    return np.array(pts[:n])


def _make_bcc(n):
    """Generate a compact BCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1.0 / 3.0))) + 2
    a = 1.0
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                p = np.array([i, j, k]) * a
                if len(pts) < n:
                    pts.append(p)
                pc = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                if len(pts) < n:
                    pts.append(pc)
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-mc * a, mc * a, 3)
            pts.append(p)
    return np.array(pts[:n])


def _make_hcp(n):
    """Generate a compact HCP-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1.0 / 3.0))) + 2
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)  # ideal c/a ratio
    # Simple HCP generation in a box
    # Layer 1: (i, j, 0)
    # Layer 2: (i+0.5, j+0.5/sqrt(3), c/2)
    # Layer 3: (i, j, c)
    # This is a rough approximation for seeding
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                # Layer 1
                p1 = np.array([i, j, k * c]) * a
                if len(pts) < n:
                    pts.append(p1)
                # Layer 2
                p2 = np.array([i + 0.5, j + 0.5 / np.sqrt(3), k * c + c / 2.0]) * a
                if len(pts) < n:
                    pts.append(p2)
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            p = rng.uniform(-mc * a, mc * a, 3)
            pts.append(p)
    return np.array(pts[:n])


def _make_icosahedral(n):
    """Generate an icosahedral cluster (Mackay-like)."""
    pts = []
    pts.append(np.array([0.0, 0.0, 0.0]))
    
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ]
    ico = np.array(ico, dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Shell 1: 12 vertices
    for v in ico:
        if len(pts) < n:
            pts.append(v * 1.0)
    
    # Shell 2: 30 edge midpoints (approx)
    if len(pts) < n:
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(ico[i] - ico[j])
                if 0.9 < d < 1.1:
                    mid = (ico[i] + ico[j]) / 2.0
                    mid = mid / np.linalg.norm(mid) * 1.5
                    edges.append(mid)
        for e in edges:
            if len(pts) < n:
                pts.append(e)
                
    # Shell 3: 20 face centers (approx)
    if len(pts) < n:
        faces = []
        # Find triangles
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d1 = np.linalg.norm(ico[i] - ico[j])
                    d2 = np.linalg.norm(ico[j] - ico[k])
                    d3 = np.linalg.norm(ico[i] - ico[k])
                    if (0.9 < d1 < 1.1) and (0.9 < d2 < 1.1) and (0.9 < d3 < 1.1):
                        center = (ico[i] + ico[j] + ico[k]) / 3.0
                        center = center / np.linalg.norm(center) * 1.5
                        faces.append(center)
        for f in faces:
            if len(pts) < n:
                pts.append(f)

    if len(pts) < n:
        rng = np.random.default_rng(123)
        while len(pts) < n:
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            r = 2.2
            p = np.array([r * np.sin(phi) * np.cos(theta),
                         r * np.sin(phi) * np.sin(theta),
                         r * np.cos(phi)])
            pts.append(p)
    return np.array(pts[:n])


def _make_random_compact(n, seed, spacing=0.72):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _basin_hopping(start_pos, kind, max_iters, rng, min_dist=0.35):
    """
    Perform Basin Hopping from start_pos.
    Returns (best_energy, best_pos).
    """
    current_energy, current_pos = relax(start_pos, "sutton_chen", kind)
    if not np.isfinite(current_energy) or not _is_valid(current_pos, min_dist):
        return np.inf, None

    best_energy = current_energy
    best_pos = current_pos.copy()
    
    # Adaptive step size
    step_size = 0.3
    
    for i in range(max_iters):
        if _time_left() < 5.0:
            break
        
        # Perturb
        perturb = rng.normal(0, step_size, current_pos.shape)
        new_pos = current_pos + perturb
        
        if not _is_valid(new_pos, min_dist):
            # If invalid, try smaller step or skip
            step_size *= 0.9
            continue
        
        # Relax
        try:
            new_energy, new_pos_relaxed = relax(new_pos, "sutton_chen", kind)
        except Exception:
            new_energy = np.inf
            new_pos_relaxed = new_pos

        if not np.isfinite(new_energy) or not _is_valid(new_pos_relaxed, min_dist):
            step_size *= 0.9
            continue

        # Metropolis acceptance (T is effectively 0 for global min search, 
        # but we can accept worse moves to escape basins if we were doing annealing.
        # For pure BH, we usually just keep the best found so far, but 
        # standard BH accepts based on energy diff to explore.
        # Here we use a simple strategy: accept if better, or accept with prob exp(-dE/T) 
        # with a small T to allow some exploration.
        T = 0.1
        dE = new_energy - current_energy
        if dE < 0 or rng.random() < np.exp(-dE / T):
            current_energy = new_energy
            current_pos = new_pos_relaxed
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos_relaxed.copy()
            # Increase step size slightly if we are making progress
            step_size = min(0.5, step_size * 1.05)
        else:
            step_size *= 0.95

    return best_energy, best_pos


def construct(instance):
    size = instance["n"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting points
    seeds = []
    seeds.append(_make_fcc(size))
    seeds.append(_make_bcc(size))
    seeds.append(_make_hcp(size))
    seeds.append(_make_icosahedral(size))
    
    # Add a few random compact seeds
    for s in range(3):
        seeds.append(_make_random_compact(size, seed=100 + s))
    
    # Time budget per seed
    # Total time ~295s. 6 seeds. ~40s per seed.
    # Relaxation takes ~0.1-0.5s. BH iter takes ~0.2s.
    # So ~200 iters per seed is feasible if we are fast, but let's be conservative.
    # We will distribute time dynamically.
    
    rng = np.random.default_rng(42)
    
    for idx, seed in enumerate(seeds):
        if _time_left() < 10.0:
            break
        
        if not _is_valid(seed):
            continue
            
        # Estimate remaining time and allocate
        remaining_seeds = len(seeds) - idx
        time_per_seed = (_time_left() - 5.0) / remaining_seeds if remaining_seeds > 0 else _time_left()
        
        # Max iterations based on time
        # Assume 0.1s per iter (optimistic) to 0.5s (pessimistic)
        # Let's just run until time is up for this seed or a max count
        max_iters = int(time_per_seed * 10) # conservative estimate
        max_iters = max(10, min(500, max_iters))
        
        try:
            e, p = _basin_hopping(seed, kind, max_iters, rng)
            if np.isfinite(e) and _is_valid(p):
                if e < best_energy:
                    best_energy = e
                    best_positions = p.copy()
        except Exception:
            continue

    # If we still have time, refine the best found
    if best_positions is not None and _time_left() > 10.0:
        # Run BH again from the best found with more iterations
        # or just more perturbations
        max_iters = int(_time_left() * 5)
        max_iters = max(10, min(1000, max_iters))
        try:
            e, p = _basin_hopping(best_positions, kind, max_iters, rng)
            if np.isfinite(e) and _is_valid(p):
                if e < best_energy:
                    best_energy = e
                    best_positions = p.copy()
        except Exception:
            pass

    if best_positions is None:
        # Fallback
        best_positions = _make_fcc(size)
        try:
            _, best_positions = relax(best_positions, "sutton_chen", kind)
        except Exception:
            pass

    if not _is_valid(best_positions):
        best_positions = np.zeros((size, 3))
        for i in range(size):
            best_positions[i] = [i * 0.5, (i // 10) * 0.5, (i % 10) * 0.5]

    return best_positions
# EVOLVE-BLOCK-END
