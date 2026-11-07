"""Sutton-Chen transition-metal cluster global minima: hierarchical growth from small cores."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Time management
_START_TIME = time.time()
_TOTAL_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 20.0

def _time_left():
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

# --- Helpers ---

def _validate(positions):
    """Check if positions are valid."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    i_idx, j_idx = np.triu_indices(n, k=1)
    d = np.linalg.norm(positions[i_idx] - positions[j_idx])
    if np.any(d < 0.35):
        return False
    return True

def _relax_one(start, kind):
    """Relax a single starting configuration. Returns (energy, positions) or (inf, None)."""
    try:
        if not _validate(start):
            return (np.inf, None)
        energy, positions = relax(start, 'sutton_chen', kind)
        if not np.all(np.isfinite(positions)):
            return (np.inf, None)
        if not _validate(positions):
            return (np.inf, None)
        return (energy, positions)
    except Exception:
        return (np.inf, None)

def _get_small_core(n_target, spacing=0.72):
    """Get a small icosahedral core with the largest Mackay shell <= n_target."""
    # Mackay icosahedral shell sizes: 1, 12, 42, 92, 147, 220, ...
    shells = [1, 12, 42, 92, 147, 220]
    core_n = 1
    for s in shells:
        if s <= n_target:
            core_n = s
        else:
            break
    return _icosahedron(core_n, spacing)

def _icosahedron(n, spacing=0.72):
    """Generate an icosahedral cluster with exactly n atoms (Mackay-like)."""
    if n <= 1:
        return np.zeros((n, 3))
    
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * spacing
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    positions = [np.array([0.0, 0.0, 0.0])]
    
    # Add vertices
    for v in verts:
        if len(positions) >= n:
            break
        positions.append(v)
    
    # If we need more, subdivide faces
    if len(positions) < n:
        # Try increasing subdivision levels
        for subdivisions in [2, 3, 4, 5, 6, 8, 10]:
            if len(positions) >= n:
                break
            for face in faces:
                v1, v2, v3 = verts[face[0]], verts[face[1]], verts[face[2]]
                for i in range(subdivisions + 1):
                    for j in range(subdivisions + 1 - i):
                        k = subdivisions - i - j
                        if i + j + k == subdivisions:
                            p = (i * v1 + j * v2 + k * v3) / subdivisions
                            if len(positions) > 0:
                                dists = np.linalg.norm(np.array(positions) - p, axis=1)
                                if np.min(dists) > 0.3 * spacing:
                                    if len(positions) < n:
                                        positions.append(p)
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break

    while len(positions) < n:
        p = np.random.uniform(-spacing * 2.5, spacing * 2.5, 3)
        positions.append(p)
    
    return np.array(positions[:n])

def _generate_surface_candidates(core, n_add, spacing=0.72, rng=None, n_candidates=200):
    """Generate candidate positions on the surface of the core for adding atoms."""
    if rng is None:
        rng = np.random.default_rng()
    
    n_core = len(core)
    if n_core == 0:
        return np.array([rng.uniform(-spacing, spacing, 3)])
    
    # Compute centroid and radius
    centroid = np.mean(core, axis=0)
    core_centered = core - centroid
    radius = np.mean(np.linalg.norm(core_centered, axis=1))
    
    candidates = []
    
    # 1. Points on the convex hull surface (approximated by sampling directions)
    n_dirs = max(50, n_add * 10)
    # Generate random directions
    dirs = rng.normal(0, 1, (n_dirs, 3))
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    
    for d in dirs:
        # Find the point on the surface in direction d
        # Project all core atoms onto d and find the max projection
        projections = core_centered @ d
        max_proj = np.max(projections)
        # The surface point is at centroid + (max_proj + spacing*0.5) * d
        # But we want it at a reasonable distance from existing atoms
        surface_point = centroid + (max_proj + spacing * 0.5) * d
        candidates.append(surface_point)
    
    # 2. Random points in a shell around the core
    n_shell = max(50, n_add * 5)
    for _ in range(n_shell):
        d = rng.normal(0, 1, 3)
        d = d / np.linalg.norm(d)
        r = radius + rng.uniform(0.3, 1.5) * spacing
        candidates.append(centroid + r * d)
    
    candidates = np.array(candidates)
    
    # Filter out candidates that are too close to existing atoms
    valid_candidates = []
    for c in candidates:
        dists = np.linalg.norm(core - c, axis=1)
        if np.min(dists) >= 0.4 * spacing:
            valid_candidates.append(c)
    
    if len(valid_candidates) == 0:
        # Fallback: just use random points
        valid_candidates = [centroid + rng.normal(0, spacing, 3) for _ in range(n_add)]
    
    return np.array(valid_candidates[:min(len(valid_candidates), n_candidates)])

def _grow_from_core(core, n_target, kind, spacing=0.72, rng=None, time_budget=60.0):
    """Grow a core to n_target atoms by adding one at a time with relaxation."""
    if rng is None:
        rng = np.random.default_rng()
    
    positions = core.copy()
    n_current = len(positions)
    
    if n_current >= n_target:
        return positions[:n_target]
    
    # Relax the core first
    try:
        energy, positions = relax(positions, 'sutton_chen', kind)
        if not np.all(np.isfinite(positions)):
            return core[:n_target]
    except Exception:
        pass
    
    n_to_add = n_target - n_current
    relax_every = max(1, n_to_add // 5)  # Relax every few additions
    
    for i in range(n_to_add):
        if _time_left() < 10.0:
            break
        
        # Generate candidate positions
        n_cands = min(50, n_to_add - i + 10)
        candidates = _generate_surface_candidates(positions, n_cands, spacing, rng)
        
        if len(candidates) == 0:
            # Fallback
            centroid = np.mean(positions, axis=0)
            candidates = np.array([centroid + rng.normal(0, spacing, 3)])
        
        # Try each candidate, keep the one with lowest energy
        best_e = np.inf
        best_pos = None
        
        for cand in candidates[:min(10, len(candidates))]:  # Limit to 10 per step for speed
            new_positions = np.vstack([positions, cand.reshape(1, 3)])
            if not _validate(new_positions):
                continue
            try:
                e, _ = sutton_chen_energy_gradient(new_positions, kind)
                if e < best_e:
                    best_e = e
                    best_pos = new_positions.copy()
            except Exception:
                continue
        
        if best_pos is None:
            # Fallback: just add a random point
            centroid = np.mean(positions, axis=0)
            new_pos = centroid + rng.normal(0, spacing, 3)
            best_pos = np.vstack([positions, new_pos.reshape(1, 3)])
        
        positions = best_pos
        
        # Periodic relaxation
        if (i + 1) % relax_every == 0 or i == n_to_add - 1:
            if _time_left() > 5.0:
                try:
                    energy, positions = relax(positions, 'sutton_chen', kind)
                    if not np.all(np.isfinite(positions)):
                        break
                except Exception:
                    break
    
    return positions[:n_target]

def _random_compact(n, spacing=0.72, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    max_attempts = n * 1000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.4 * spacing:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    
    return np.array(points)

def construct(instance):
    """Find near-global minimum for Sutton-Chen transition-metal clusters using hierarchical growth."""
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345 + 67890)
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Hierarchical growth from icosahedral core
    core_n = min(n, 42)  # Use 42-atom core as max for smaller clusters
    core = _get_small_core(core_n, 0.72)
    
    try:
        grown = _grow_from_core(core, n, kind, 0.72, rng)
        if len(grown) == n and _validate(grown):
            e, pos = relax(grown, 'sutton_chen', kind)
            if np.all(np.isfinite(pos)) and _validate(pos) and e < best_energy:
                best_energy = e
                best_positions = pos.copy()
    except Exception:
        pass
    
    # Strategy 2: Multiple random restarts in parallel
    if _time_left() > 20.0:
        n_restarts = min(8, max(2, int(_time_left() / 30)))
        starts = [_random_compact(n, 0.72, rng) for _ in range(n_restarts)]
        
        # Also add icosahedral and deformed starts
        icosa = _icosahedron(n, 0.72)
        if _validate(icosa):
            starts.append(icosa)
        
        if best_positions is not None:
            for scale in [0.05, 0.1]:
                deformed = best_positions + rng.normal(0, scale, best_positions.shape)
                if _validate(deformed):
                    starts.append(deformed)
        
        max_workers = min(4, len(starts))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_relax_one, start, kind): i for i, start in enumerate(starts)}
            for future in as_completed(futures):
                if _time_left() < 5.0:
                    break
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
    
    # Strategy 3: More random restarts if time permits
    attempts = 0
    max_attempts = 20
    while _time_left() > 15.0 and attempts < max_attempts:
        batch_size = min(4, max_attempts - attempts)
        batch = []
        for _ in range(batch_size):
            start = _random_compact(n, 0.72, rng)
            if _validate(start):
                batch.append(start)
        
        if not batch:
            attempts += batch_size
            continue
        
        with ThreadPoolExecutor(max_workers=min(4, len(batch))) as executor:
            futures = {executor.submit(_relax_one, start, kind): i for i, start in enumerate(batch)}
            for future in as_completed(futures):
                if _time_left() < 5.0:
                    break
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
        
        attempts += batch_size
    
    # Final polish
    if _time_left() > 8.0 and best_positions is not None and np.isfinite(best_energy):
        try:
            energy, positions = relax(best_positions, 'sutton_chen', kind)
            if np.all(np.isfinite(positions)) and _validate(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
        except Exception:
            pass
    
    # Fallback if nothing worked
    if best_positions is None or not _validate(best_positions):
        best_positions = _random_compact(n, 0.72, rng)
        for i in range(len(best_positions)):
            for j in range(i + 1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.35:
                    direction = best_positions[i] - best_positions[j]
                    if np.linalg.norm(direction) > 1e-10:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.array([1.0, 0, 0])
                    best_positions[j] = best_positions[i] + direction * 0.36
    
    return best_positions
# EVOLVE-BLOCK-END
