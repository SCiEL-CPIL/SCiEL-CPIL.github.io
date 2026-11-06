"""Dzugutov cluster global minima: Mackay icosahedral seeds + parallel basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_DEADLINE = 295.0  # seconds, safety margin before 330s kill
_THREAD_LOCK = threading.Lock()


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _energy(pos):
    """Compute Dzugutov energy vectorized."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    triu_idx = np.triu_indices(n, k=1)
    r = dist[triu_idx]
    
    E = 0.0
    if np.min(r) < 0.5:
        return 1e10
    
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        E += np.sum(A * (r1 ** (-m) - B) * np.exp(c / (r1 - a)))
    
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2 = r[mask2]
        E += np.sum(B * np.exp(d / (r2 - b)))
    
    return float(E)


def _icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts


def _mckay_icosahedron(n):
    """
    Generate a Mackay icosahedron structure with exactly n atoms.
    If n is not a Mackay number (13, 55, 147...), it truncates the next larger
    Mackay icosahedron by removing outer atoms randomly to maintain symmetry as much as possible,
    or simply takes the first n atoms from the sorted list of the full Mackay structure.
    """
    # Mackay numbers: 1, 13, 55, 147, 301, ...
    # Formula: N = 1 + 12 * sum_{k=1}^{m} k(k+1)
    # m=0: 1
    # m=1: 1 + 12*2 = 25? No.
    # Mackay icosahedron shells:
    # Center: 1
    # Shell 1 (k=1): 12 atoms. Total 13.
    # Shell 2 (k=2): 42 atoms. Total 55.
    # Shell 3 (k=3): 92 atoms. Total 147.
    
    # We will generate the full Mackay icosahedron up to the shell that covers n.
    # Then we will select n atoms from it.
    
    ico_verts = _icosahedron_vertices()
    
    # Shell 0: Center
    positions = [np.zeros(3)]
    
    # Shell 1: 12 vertices
    r1 = 1.0
    for v in ico_verts:
        positions.append(r1 * v)
    
    if n <= 13:
        return np.array(positions[:n])
    
    # Shell 2: 42 atoms
    # Directions: 12 face centers + 30 edge midpoints
    # We need to generate these directions properly.
    
    # Compute edges and faces of the icosahedron defined by ico_verts
    dists_ico = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists_ico[i, j] = np.linalg.norm(ico_verts[i] - ico_verts[j])
    
    edge_set = set()
    for i in range(12):
        neighbors = np.argsort(dists_ico[i])
        for j in neighbors[1:6]:
            if i < j:
                edge_set.add((i, j))
            else:
                edge_set.add((j, i))
    
    # Edge midpoints (30 directions)
    edge_mid_dirs = []
    for (i, j) in edge_set:
        mid = (ico_verts[i] + ico_verts[j])
        mid = mid / np.linalg.norm(mid)
        edge_mid_dirs.append(mid)
    
    # Face centers (20 directions)
    face_dirs = []
    for i in range(12):
        for j in range(i + 1, 12):
            if (i, j) in edge_set:
                for k in range(j + 1, 12):
                    if (i, k) in edge_set and (j, k) in edge_set:
                        fc = (ico_verts[i] + ico_verts[j] + ico_verts[k])
                        fc = fc / np.linalg.norm(fc)
                        face_dirs.append(fc)
    
    # The 42-atom shell consists of 12 "face center" type positions and 30 "edge midpoint" type positions?
    # Actually, the 42-atom shell in Mackay icosahedron is formed by:
    # 12 atoms at the centers of the 20 faces? No, that's 20.
    # The 42 atoms are: 12 at the vertices of a dodecahedron (which are the face centers of the icosahedron? No, dodecahedron vertices correspond to icosahedron face centers).
    # Wait, the dual of icosahedron is dodecahedron. The 20 faces of icosahedron correspond to 20 vertices of dodecahedron.
    # The 12 vertices of icosahedron correspond to 12 faces of dodecahedron.
    
    # Standard Mackay icosahedron shell 2 (42 atoms):
    # 12 atoms at the centers of the 12 faces of the dodecahedron? No.
    # Let's look at the coordinates.
    # Shell 2 radius is $r_2 = r_1 \sqrt{5+2\sqrt{5}}/2 \approx 1.902 r_1$.
    # The directions are:
    # 1. 12 directions corresponding to the vertices of the dodecahedron (which are the face centers of the icosahedron).
    # 2. 30 directions corresponding to the edge midpoints of the icosahedron.
    
    # So we have 20 face centers of the icosahedron. We need 12.
    # Actually, the 12 positions in shell 2 are the vertices of a dodecahedron.
    # The vertices of a dodecahedron are the face centers of an icosahedron.
    # But there are 20 face centers.
    
    # Correction: The 42-atom shell is composed of:
    # - 12 atoms at the vertices of a dodecahedron (which are 12 of the 20 face centers of the icosahedron? No, all 20 face centers form a dodecahedron? No, the dodecahedron has 20 vertices. The icosahedron has 20 faces. So the 20 face centers of the icosahedron are the 20 vertices of the dual dodecahedron.)
    
    # Let's check the count.
    # Shell 1: 12 atoms (icosahedron vertices).
    # Shell 2: 42 atoms.
    # If we take all 20 face centers (dodecahedron vertices) and all 30 edge midpoints, that's 50.
    # So it's not all of them.
    
    # Let's use a known set of 42 directions.
    # A common construction for the 42-atom shell:
    # It consists of 12 atoms at the centers of the 12 pentagonal faces of the dodecahedron?
    
    # Alternative: Just generate 42 points on the sphere with icosahedral symmetry.
    # We can use the 30 edge midpoints and 12 face centers (choosing 12 out of 20?).
    
    # For the purpose of this seed, precision is less critical than having a good compact start.
    # Let's use the 30 edge midpoints and 12 of the face centers (e.g., those with positive z, or just first 12).
    
    r2 = r1 * 1.902113032590307 # sqrt(5+2*sqrt(5))/2
    
    shell2_dirs = np.array(edge_mid_dirs) # 30 dirs
    # Add 12 face centers
    # We have 20 face_dirs. Let's take the first 12.
    shell2_dirs = np.vstack([shell2_dirs, np.array(face_dirs[:12])])
    
    for d in shell2_dirs:
        positions.append(r2 * d)
    
    if n <= 55:
        return np.array(positions[:n])
    
    # Shell 3: 92 atoms
    # This is more complex. For n up to 60, we might just need to truncate the 147-atom icosahedron.
    # But generating 92 directions is hard.
    # Instead, for n > 55, we can generate a random compact cluster or a larger shell approximation.
    # However, the task asks for n in 38-60.
    # So we only need to handle n up to 60.
    # The 55-atom Mackay icosahedron is a very good seed for n=55.
    # For n=57, 60, we can take the 55-atom structure and add 2-5 atoms at the surface.
    # Or we can just truncate the 147-atom structure if we had it.
    
    # Let's just return the 55-atom structure truncated/padded for n > 55?
    # No, we need exactly n atoms.
    # If n > 55, we will use the 55-atom structure as a seed and let basin hopping add/adjust?
    # No, the number of atoms is fixed.
    
    # For n > 55, let's use a random compact seed or a simple expansion.
    # Since n is max 60, we can just take the 55-atom structure and add 5 atoms at the outer shell positions (randomly chosen from available spots).
    
    if n > 55:
        # Add atoms at r3
        r3 = r1 * 3.0 # Approximate
        # Just scatter them on the surface
        for _ in range(n - 55):
            theta = np.arccos(2 * np.random.random() - 1)
            phi = 2 * np.pi * np.random.random()
            d = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            positions.append(r3 * d)
    
    return np.array(positions[:n])


def _scatter(n, radius=3.0):
    """Generate random compact cluster."""
    rng = np.random.default_rng()
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
    
    while len(points) < n:
        p = np.array([rng.uniform(-0.3, 0.3) for _ in range(3)])
        points.append(p)
    
    return np.array(points[:n])


def _fcc_cluster(n):
    """Generate FCC-like cluster."""
    points = []
    a = 1.0
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                for offset in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]]) * a
                    if np.linalg.norm(p) <= 2.5:
                        points.append(p)
    
    rng = np.random.default_rng(42)
    rng.shuffle(points)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-1, 1, 3)
            points.append(p)
    
    return np.array(points[:n])


def _seeds(n):
    """Generate diverse seed structures."""
    seeds = []
    
    # 1. Icosahedral Mackay seed (High quality)
    try:
        ico = _mckay_icosahedron(n)
        if len(ico) == n and np.all(np.isfinite(ico)):
            seeds.append(ico)
    except:
        pass
    
    # 2. FCC seed
    try:
        fcc = _fcc_cluster(n)
        seeds.append(fcc)
    except:
        pass
    
    # 3. Random compact seeds
    for trial in range(3):
        try:
            sc = _scatter(n, radius=2.5 + 0.5 * trial)
            seeds.append(sc)
        except:
            pass
    
    return seeds


def _basin_hopping(pos, n, max_iter=50, time_limit=30):
    """Basin hopping: perturb, relax, keep if lower energy."""
    start_time = time.time()
    current_energy = _energy(pos)
    best_energy = current_energy
    best_pos = pos.copy()
    rng = np.random.default_rng(hash(tuple(pos.flatten()[:3])) % (2**32))
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if _remaining() < 5:
            break
        
        # Adaptive step size
        step_size = 0.4 + 0.2 * (1 - it / max_iter)
        perturbed = pos + rng.normal(0, step_size, size=pos.shape)
        
        # Relax
        try:
            e, relaxed = relax(perturbed, 'dzugutov', None)
        except:
            continue
        
        if not np.all(np.isfinite(e)) or e >= 1e9:
            continue
        
        if e < current_energy:
            current_energy = e
            pos = relaxed
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
    
    return best_energy, best_pos


def _single_search(n, seed_idx, time_limit=40):
    """Run a single search from a given seed."""
    start_time = time.time()
    seeds = _seeds(n)
    
    if seed_idx >= len(seeds):
        seed_idx = seed_idx % len(seeds)
    
    pos = seeds[seed_idx].copy()
    
    # First, relax the seed
    try:
        e, pos = relax(pos, 'dzugutov', None)
    except:
        return 1e10, None
    
    if not np.all(np.isfinite(e)) or e >= 1e9:
        return 1e10, None
    
    best_energy = e
    best_pos = pos.copy()
    
    # Basin hopping
    try:
        bh_energy, bh_pos = _basin_hopping(pos, n, max_iter=40, time_limit=time_limit)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    except:
        pass
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster structure."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    
    # Generate seeds
    seeds = _seeds(n)
    
    best_energy = 1e10
    best_pos = None
    
    # Run parallel searches
    n_seeds = len(seeds)
    n_threads = min(4, n_seeds)
    
    # Time budget per search
    total_time_budget = min(280.0, _remaining())
    per_search_time = total_time_budget / n_threads
    
    results = []
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {}
        for i in range(n_seeds):
            if _remaining() < 10:
                break
            future = executor.submit(_single_search, n, i, per_search_time)
            futures[future] = i
        
        for future in as_completed(futures):
            if _remaining() < 5:
                break
            try:
                energy, pos = future.result(timeout=10)
                if pos is not None and energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except:
                pass
    
    # If no valid result, use a fallback
    if best_pos is None or not np.all(np.isfinite(best_pos)):
        if seeds:
            try:
                e, pos = relax(seeds[0], 'dzugutov', None)
                if np.all(np.isfinite(pos)) and e < 1e9:
                    best_pos = pos
                    best_energy = e
            except:
                pass
        
        if best_pos is None:
            best_pos = _scatter(n, radius=2.0)
            try:
                _, best_pos = relax(best_pos, 'dzugutov', None)
            except:
                pass
    
    # Final check: ensure minimum distance
    if best_pos is not None:
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        
        if min_dist < 0.5:
            for _ in range(100):
                if min_dist >= 0.5:
                    break
                i, j = np.unravel_index(np.argmin(dist), dist.shape)
                if i > j:
                    i, j = j, i
                dir_vec = best_pos[i] - best_pos[j]
                norm = np.linalg.norm(dir_vec)
                if norm < 1e-10:
                    dir_vec = np.array([1.0, 0, 0])
                else:
                    dir_vec = dir_vec / norm
                best_pos[i] += 0.01 * dir_vec
                best_pos[j] -= 0.01 * dir_vec
                diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff ** 2, axis=2))
                np.fill_diagonal(dist, np.inf)
                min_dist = np.min(dist)
            
            try:
                _, best_pos = relax(best_pos, 'dzugutov', None)
            except:
                pass
    
    return best_pos
# EVOLVE-BLOCK-END
