# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_HARD_LIMIT = 290.0
_INSTANCE_ORDER = [38, 45, 50, 55, 57, 60]
_INSTANCE_INDEX = {n: i for i, n in enumerate(_INSTANCE_ORDER)}

# Dzugutov potential parameters
A_DZ = 5.82
B_DZ = 1.28
a_DZ = 1.87
b_DZ = 1.94
c_DZ = 1.10
d_DZ = 0.27
m_DZ = 16


def _get_icosahedron_vertices():
    """Return 12 icosahedron vertices (unit sphere)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    return verts


def _icosahedron_edges(verts):
    """Find edges of icosahedron from vertex list."""
    edges = []
    dists = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists.append((d, i, j))
    dists.sort()
    min_dist = dists[0][0]
    for d, i, j in dists:
        if d < min_dist * 1.01:
            edges.append((i, j))
    return edges


def _icosahedron_faces(verts, edges):
    """Find triangular faces of icosahedron."""
    connected = set()
    for i, j in edges:
        connected.add((i, j))
        connected.add((j, i))
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            if (i, j) in connected:
                for k in range(j + 1, 12):
                    if (j, k) in connected and (i, k) in connected:
                        faces.append((i, j, k))
    return faces


def _mackay_icosahedron_seed(n_atoms, seed_idx):
    """
    Build a Mackay-type icosahedral cluster seed.
    Shell 1: center (1 atom)
    Shell 2: 12 vertices (total 13)
    Shell 3: 30 edge midpoints (total 43)
    Shell 4: 20 face centers (total 63)
    For n <= 13: just the icosahedron
    For 14 <= n <= 43: center + 12 verts + first (n-13) edge mids
    For 44 <= n <= 63: center + 12 verts + 30 edge mids + first (n-43) face centers
    """
    verts = _get_icosahedron_vertices()
    edges = _icosahedron_edges(verts)
    faces = _icosahedron_faces(verts, edges)
    
    rng = np.random.default_rng(seed_idx * 7919 + n_atoms * 13)
    
    positions = [np.zeros(3)]  # center
    
    # Shell 2: 12 vertices at r=1
    r1 = 1.0
    for v in verts:
        if len(positions) >= n_atoms:
            break
        p = v * r1 + rng.normal(0, 0.02, 3)
        positions.append(p)
    
    remaining = n_atoms - len(positions)
    if remaining <= 0:
        return np.array(positions[:n_atoms])
    
    # Shell 3: 30 edge midpoints at r=sqrt(2)
    r2 = r1 * np.sqrt(2.0)
    for i, j in edges:
        if remaining <= 0:
            break
        mid = (verts[i] + verts[j]) / 2.0
        p = mid * r2 + rng.normal(0, 0.02, 3)
        positions.append(p)
        remaining -= 1
    
    if remaining <= 0:
        return np.array(positions[:n_atoms])
    
    # Shell 4: 20 face centers at r=sqrt(3)
    r3 = r1 * np.sqrt(3.0)
    for i, j, k in faces:
        if remaining <= 0:
            break
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        p = fc * r3 + rng.normal(0, 0.02, 3)
        positions.append(p)
        remaining -= 1
    
    # If still need more (n > 63), add random surface atoms
    if remaining > 0:
        r4 = r1 * 2.0
        for _ in range(remaining):
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi ** 2)
            p = r4 * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            p = p + rng.normal(0, 0.1, 3)
            positions.append(p)
    
    return np.array(positions[:n_atoms])


def _partial_icosahedron_seed(n_atoms, seed_idx):
    """
    Alternative: start from icosahedron core, add atoms at
    geometrically favorable positions with more randomness.
    """
    verts = _get_icosahedron_vertices()
    edges = _icosahedron_edges(verts)
    faces = _icosahedron_faces(verts, edges)
    
    rng = np.random.default_rng(seed_idx * 104729 + n_atoms)
    
    positions = [np.zeros(3)]
    r1 = 1.0
    for v in verts:
        if len(positions) >= n_atoms:
            break
        p = v * r1 + rng.normal(0, 0.05, 3)
        positions.append(p)
    
    remaining = n_atoms - len(positions)
    if remaining <= 0:
        return np.array(positions[:n_atoms])
    
    # Mix of edge mids and face centers with random perturbation
    r2 = r1 * np.sqrt(2.0)
    candidates = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        candidates.append(mid * r2)
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        candidates.append(fc * r1 * np.sqrt(3.0))
    
    # Shuffle candidates
    idxs = rng.permutation(len(candidates))
    for idx in idxs:
        if remaining <= 0:
            break
        p = candidates[idx] + rng.normal(0, 0.08, 3)
        # Check minimum distance
        valid = True
        for q in positions:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            positions.append(p)
            remaining -= 1
    
    if remaining > 0:
        r3 = r1 * 1.8
        for _ in range(remaining):
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi ** 2)
            p = r3 * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            p = p + rng.normal(0, 0.15, 3)
            positions.append(p)
    
    return np.array(positions[:n_atoms])


def _random_compact_seed(n_atoms, seed_idx):
    """Random compact cluster."""
    rng = np.random.default_rng(seed_idx * 31 + n_atoms)
    radius = 1.2 * n_atoms ** (1.0 / 3.0) * 0.7
    points = []
    max_attempts = 3000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n_atoms])


def _generate_seeds(n_atoms, n_seeds):
    """Generate diverse seeds for n_atoms atoms."""
    seeds = []
    
    # Mackay icosahedral seeds (primary)
    for s in range(min(4, n_seeds)):
        try:
            seeds.append(_mackay_icosahedron_seed(n_atoms, seed_idx=s))
        except Exception:
            pass
    
    # Partial icosahedron with more randomness
    for s in range(min(4, max(0, n_seeds - 4))):
        try:
            seeds.append(_partial_icosahedron_seed(n_atoms, seed_idx=s))
        except Exception:
            pass
    
    # Random compact for diversity
    for s in range(min(4, max(0, n_seeds - 8))):
        try:
            seeds.append(_random_compact_seed(n_atoms, seed_idx=s))
        except Exception:
            pass
    
    # Fallback
    if not seeds:
        rng = np.random.default_rng(n_atoms)
        radius = 1.2 * n_atoms ** (1.0 / 3.0) * 0.7
        points = []
        while len(points) < n_atoms:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        seeds.append(np.array(points))
    
    return seeds[:n_seeds]


def _basin_hopping(positions, potential, parameter, rng, max_iters, kick_start, kick_end, deadline):
    """
    Basin hopping with adaptive cooling.
    Returns (best_energy, best_positions).
    """
    try:
        best_energy, best_positions = relax(positions.copy(), potential, parameter)
    except Exception:
        return float('inf'), positions
    
    current_energy = best_energy
    current_positions = best_positions.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        t = it / max(1, max_iters - 1)
        # Exponential cooling for kick scale
        kick_scale = kick_start * (kick_end / kick_start) ** t
        
        kick = rng.normal(0, kick_scale, size=current_positions.shape)
        new_positions = current_positions + kick
        
        try:
            new_energy, new_relaxed = relax(new_positions, potential, parameter)
        except Exception:
            continue
        
        if new_energy < current_energy:
            current_energy = new_energy
            current_positions = new_relaxed
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_relaxed.copy()
        else:
            delta_e = new_energy - current_energy
            # Metropolis acceptance with cooling
            temp = 0.08 * (1.0 - t) + 0.01
            if delta_e < 0.5 and rng.random() < np.exp(-delta_e / temp):
                current_energy = new_energy
                current_positions = new_relaxed
    
    return best_energy, best_positions


def construct(instance):
    """
    Two-stage search:
    Stage 1: Fast multi-start local minimization from diverse seeds.
    Stage 2: Basin hopping from top-K relaxed configurations.
    """
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    elapsed = time.time() - _START_TIME
    remaining_total = max(0.0, _HARD_LIMIT - elapsed)
    
    idx = _INSTANCE_INDEX.get(size, 0)
    n_rem = len(_INSTANCE_ORDER) - idx
    
    if n_rem > 0:
        call_budget = remaining_total / n_rem
    else:
        call_budget = remaining_total
    
    call_budget = max(call_budget, 1.5)
    call_deadline = time.time() + call_budget
    
    # Stage 1: Generate seeds and do fast relaxations
    n_seeds = min(16, max(8, int(call_budget * 2)))
    seeds = _generate_seeds(size, n_seeds)
    
    # Relax all seeds in parallel
    relaxed_results = []  # list of (energy, positions)
    
    def do_relax(seed):
        if time.time() > call_deadline:
            return (float('inf'), None)
        try:
            e, pos = relax(seed.copy(), potential, parameter)
            return (e, pos)
        except Exception:
            return (float('inf'), None)
    
    n_threads = min(4, len(seeds))
    
    if n_threads == 1:
        for s in seeds:
            e, pos = do_relax(s)
            if pos is not None:
                relaxed_results.append((e, pos))
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = {executor.submit(do_relax, s): i for i, s in enumerate(seeds)}
            for future in as_completed(futures):
                try:
                    timeout = max(0.1, call_deadline - time.time())
                    e, pos = future.result(timeout=timeout)
                    if pos is not None:
                        relaxed_results.append((e, pos))
                except Exception:
                    continue
    
    if not relaxed_results:
        # Fallback: return first seed
        return seeds[0]
    
    # Sort by energy
    relaxed_results.sort(key=lambda x: x[0])
    
    best_energy = relaxed_results[0][0]
    best_positions = relaxed_results[0][1].copy()
    
    # Check remaining time for Stage 2
    time_used = time.time() - (call_deadline - call_budget)
    remaining_for_bh = call_deadline - time.time()
    
    if remaining_for_bh < 3.0:
        return best_positions
    
    # Stage 2: Basin hopping from top-K relaxed configs
    K = min(4, len(relaxed_results))
    
    # Allocate time for BH: use up to 60% of remaining
    bh_deadline = time.time() + min(remaining_for_bh * 0.6, 15.0)
    
    # Determine max iterations based on time
    # Estimate: each relax ~0.05-0.2s, so ~5-20 iters per chain
    max_iters = min(30, max(5, int(remaining_for_bh * 0.5)))
    
    kick_start = 0.30
    kick_end = 0.05
    
    def run_bh_chain(start_pos, chain_id):
        if time.time() > bh_deadline:
            return (float('inf'), None)
        rng = np.random.default_rng(chain_id * 1000 + size)
        e, pos = _basin_hopping(
            start_pos, potential, parameter, rng,
            max_iters, kick_start, kick_end, bh_deadline
        )
        return (e, pos)
    
    n_bh_threads = min(4, K)
    
    if n_bh_threads == 1:
        e, pos = run_bh_chain(relaxed_results[0][1], 0)
        if pos is not None and e < best_energy:
            best_energy = e
            best_positions = pos
    else:
        with ThreadPoolExecutor(max_workers=n_bh_threads) as executor:
            futures = {
                executor.submit(run_bh_chain, relaxed_results[i][1], i): i
                for i in range(K)
            }
            for future in as_completed(futures):
                try:
                    timeout = max(0.1, bh_deadline - time.time())
                    e, pos = future.result(timeout=timeout)
                    if pos is not None and e < best_energy:
                        best_energy = e
                        best_positions = pos
                except Exception:
                    continue
    
    return best_positions
# EVOLVE-BLOCK-END
