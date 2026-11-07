# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
import scipy.spatial
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Constants
KIND_PARAMS = {
    '12-6': (12, 6, 144.41),
    '9-6': (9, 6, 39.432),
    '10-8': (10, 8, 34.408),
}

def _energy(positions, n_exp, m_exp, c):
    """Compute total energy for a configuration."""
    if positions is None or len(positions) == 0:
        return 0.0
    n = len(positions)
    # Compute all pair distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    # Mask out self-interactions (diagonal)
    np.fill_diagonal(dists, np.inf)
    if np.any(dists < 1e-9): # Prevent division by zero if atoms overlap
        return np.inf
    
    r_n = np.sum(dists ** (-n_exp), axis=1)
    r_m = np.sum(dists ** (-m_exp), axis=1)
    
    # E = sum_i [ 1/2 * sum_{j!=i} r^-n - c * sqrt(sum_{j!=i} r^-m) ]
    # Note: The prompt says "sum_i [ 1/2 sum_{j!=i} ... ]"
    # This effectively counts each pair twice in the first term (once for i, once for j), 
    # but the 1/2 factor makes it 1/2 * 2 * PairSum = PairSum.
    # Wait, standard LJ is sum_pairs. 
    # Let's check the formula in the prompt:
    # E = sum_i [1/2 sum_{j!=i} r_ij^-n_exp - c sqrt(sum_{j!=i} r_ij^-m_exp)]
    # Term 1: sum_i 1/2 sum_{j!=i} r^-n = sum_{i<j} r^-n. Correct.
    # Term 2: -c * sum_i sqrt(sum_{j!=i} r^-m).
    # This is NOT standard LJ. It's the Sutton-Chen form.
    
    e = 0.0
    for i in range(n):
        e += 0.5 * r_n[i]
        if r_m[i] > 0:
            e -= c * np.sqrt(r_m[i])
    return e

def _validate(positions, n, min_dist=0.35):
    """Check validity: finite, correct shape, min distance."""
    if positions is None:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    if len(positions) != n:
        return False
    if positions.shape[1] != 3:
        return False
    # Check min distance efficiently using cKDTree
    if n > 1:
        tree = scipy.spatial.cKDTree(positions)
        dists, _ = tree.query(positions, k=2)
        min_d = np.min(dists[:, 1])
        if min_d < min_dist:
            return False
    return True

def _random_compact(n, rng, scale=1.0):
    """Generate a random compact cluster in a sphere."""
    # Estimate radius based on density ~ 1 atom / 1.0^3
    # Volume V = 4/3 pi R^3 = n * 1.0
    r_est = (3.0 * n / (4.0 * np.pi)) ** (1.0 / 3.0) * scale
    pts = []
    while len(pts) < n:
        # Sample from normal distribution and clip to sphere? 
        # Or rejection sample.
        p = rng.normal(0, 1, size=3)
        if np.linalg.norm(p) <= 1.0:
            pts.append(p)
    pts = np.array(pts) * r_est
    # Add small noise to break symmetry
    pts += rng.normal(0, 0.1, size=pts.shape)
    return pts

def _icosahedral_core(n, rng):
    """
    Generate an icosahedral-like core.
    For n=12, perfect icosahedron.
    For n>12, add shells.
    """
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices of icosahedron
    # Scaled such that nearest neighbor distance is ~ 1.0
    # Distance between (0,1,phi) and (0,-1,phi) is 2.
    # So scale by 0.5.
    scale = 0.5
    base_vertices = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ]) * scale
    
    if n <= 12:
        # For n < 12, random is often better than a broken icosahedron
        if n == 12:
            return base_vertices
        return _random_compact(n, rng)
    
    # n > 12
    # Start with the 12 vertices
    pts = [base_vertices]
    count = 12
    
    # Add center if n >= 13? 
    # The 13-atom icosahedron (Mackay) has a center.
    if n >= 13:
        pts.append(np.array([[0.0, 0.0, 0.0]]))
        count = 13
    
    if n <= 13:
        return np.vstack(pts)
    
    # Add more atoms
    # Strategy: Add atoms in shells of increasing radius
    # Mackay icosahedron: 1, 13, 55, 147...
    # Shell 1: 12 atoms (radius R1)
    # Shell 2: 42 atoms (radius R2 = 2*R1)
    # Shell 3: 120 atoms (radius R3 = 3*R1)
    
    R1 = np.linalg.norm(base_vertices[0])
    
    remaining = n - count
    shell_idx = 1 # We are filling shell 2 (index 1, after center+shell1)
    # Actually, let's just fill shells
    shells = [12, 42, 120, 240] # Approximate numbers for Mackay
    # Coordinates for shell 2 (42 atoms):
    # These are the vertices of a truncated icosahedron? No.
    # They are face centers and edge centers?
    # A simpler heuristic: Place atoms at the vertices of larger icosahedra?
    # No, let's just place atoms on a sphere of radius R1 * (shell_idx + 1)
    # with some angular distribution.
    
    # For robustness, if we don't have specific coordinates for large n,
    # we fall back to adding random points in the volume.
    # But let's try to generate a "liquid-like" shell.
    
    # Generate random points on spheres of increasing radius
    while count < n:
        shell_r = R1 * (shell_idx + 1)
        # Number of atoms to place in this shell
        # Estimate based on surface area
        area = 4 * np.pi * shell_r**2
        # Density ~ 1 atom / 1.0 area? 
        n_shell = int(np.pi * shell_r**2 / 2.0) # Heuristic
        if n_shell < 1:
            n_shell = 1
        # Place n_shell random points on the sphere
        pts_in_shell = []
        for _ in range(n_shell):
            # Random direction
            vec = rng.normal(0, 1, size=3)
            vec /= np.linalg.norm(vec)
            pos = vec * shell_r
            pts_in_shell.append(pos)
        pts_in_shell = np.array(pts_in_shell)
        # Add noise
        pts_in_shell += rng.normal(0, 0.1, size=pts_in_shell.shape)
        pts.append(pts_in_shell)
        count += len(pts_in_shell)
        shell_idx += 1
        
    all_pts = np.vstack(pts)
    # If we have too many, truncate
    if len(all_pts) > n:
        # Keep the first n? Or random n?
        # Random is better to break symmetry
        indices = rng.choice(len(all_pts), size=n, replace=False)
        all_pts = all_pts[indices]
        
    return all_pts

def _bcc_lattice(n, rng):
    """Generate BCC lattice points within a sphere."""
    # BCC density is 2 atoms per unit cell volume a^3.
    # Nearest neighbor distance = a * sqrt(3) / 2.
    # We want nn ~ 1.0, so a = 2 / sqrt(3) ~ 1.15.
    a = 2.0 / np.sqrt(3.0)
    # Volume per atom = a^3 / 2
    vol_per_atom = a**3 / 2.0
    R = (3.0 * n * vol_per_atom / (4.0 * np.pi)) ** (1.0 / 3.0)
    
    max_idx = int(np.ceil(R / a)) + 2
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i, j, k]) * a
                if np.linalg.norm(p1) <= R:
                    points.append(p1)
                p2 = (np.array([i + 0.5, j + 0.5, k + 0.5])) * a
                if np.linalg.norm(p2) <= R:
                    points.append(p2)
    
    if len(points) < n:
        return _random_compact(n, rng)
        
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    pos = points[sorted_idx[:n]]
    pos += rng.normal(0, 0.05, size=pos.shape)
    return pos

def _fcc_lattice(n, rng):
    """Generate FCC lattice points within a sphere."""
    # FCC density is 4 atoms per unit cell volume a^3.
    # Nearest neighbor distance = a * sqrt(2) / 2 = a / sqrt(2).
    # We want nn ~ 1.0, so a = sqrt(2).
    a = np.sqrt(2.0)
    vol_per_atom = a**3 / 4.0
    R = (3.0 * n * vol_per_atom / (4.0 * np.pi)) ** (1.0 / 3.0)
    
    max_idx = int(np.ceil(R / a)) + 2
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i, j, k]) * a
                if np.linalg.norm(p1) <= R:
                    points.append(p1)
                p2 = (np.array([i + 0.5, j + 0.5, k])) * a
                if np.linalg.norm(p2) <= R:
                    points.append(p2)
                p3 = (np.array([i + 0.5, j, k + 0.5])) * a
                if np.linalg.norm(p3) <= R:
                    points.append(p3)
                p4 = (np.array([i, j + 0.5, k + 0.5])) * a
                if np.linalg.norm(p4) <= R:
                    points.append(p4)
                
    if len(points) < n:
        return _random_compact(n, rng)
        
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    pos = points[sorted_idx[:n]]
    pos += rng.normal(0, 0.05, size=pos.shape)
    return pos

def _basin_hopping_worker(args):
    """
    Worker for Basin Hopping.
    args: (initial_pos, kind, n, seed, time_budget)
    """
    init_pos, kind, n, seed, time_budget = args
    rng = np.random.default_rng(seed)
    
    current_pos = init_pos.copy()
    
    # Initial relaxation
    try:
        current_e, current_pos = relax(current_pos, 'sutton_chen', kind)
        if not np.isfinite(current_e) or not _validate(current_pos, n):
            return (np.inf, None)
    except Exception:
        return (np.inf, None)
    
    best_e = current_e
    best_pos = current_pos.copy()
    start_time = time.time()
    
    # Basin Hopping loop
    while time.time() - start_time < time_budget:
        # Perturb
        # Adaptive step size? Start with 0.5, decrease if stuck?
        step_size = 0.5
        new_pos = current_pos + rng.normal(0, step_size, size=current_pos.shape)
        
        # Relax
        try:
            new_e, new_relaxed_pos = relax(new_pos, 'sutton_chen', kind)
            if not np.isfinite(new_e) or not _validate(new_relaxed_pos, n):
                continue
        except Exception:
            continue
        
        # Update best
        if new_e < best_e:
            best_e = new_e
            best_pos = new_relaxed_pos.copy()
        
        # Acceptance criterion
        # Metropolis: accept if new_e < current_e, else with prob exp(-(new_e-current_e)/T)
        # We don't have a temperature. 
        # A common choice is to accept if new_e < current_e + 1.0 (energy scale)
        # Or just greedy: accept if new_e < current_e.
        # Greedy gets stuck easily. 
        # Let's try a "hill climbing" with occasional jumps.
        # If new_e < current_e: accept
        # Else: accept with prob 0.05 (random escape)
        if new_e < current_e:
            current_pos = new_relaxed_pos
            current_e = new_e
        else:
            if rng.random() < 0.05:
                current_pos = new_relaxed_pos
                current_e = new_e
            else:
                # Stay put
                pass

    return (best_e, best_pos)

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    
    if kind not in KIND_PARAMS:
        # Fallback if kind is unknown
        return np.random.default_rng(0).normal(size=(n, 3))

    start_time = time.time()
    deadline = start_time + 290.0  # 40s safety margin for 330s limit
    
    # Generate diverse starting configurations
    rng_base = np.random.default_rng(42 ^ (n * 101))
    
    # 1. Icosahedral
    pos1 = _icosahedral_core(n, rng_base)
    
    # 2. Random Compact
    pos2 = _random_compact(n, rng_base)
    
    # 3. BCC
    pos3 = _bcc_lattice(n, rng_base)
    
    # 4. FCC
    pos4 = _fcc_lattice(n, rng_base)
    
    # 5. Another Random (different scale)
    pos5 = _random_compact(n, rng_base, scale=1.2)
    
    # 6. Another Icosahedral (noised)
    pos6 = _icosahedral_core(n, rng_base)
    pos6 += rng_base.normal(0, 0.2, size=pos6.shape)
    
    # 7. BCC noised
    pos7 = _bcc_lattice(n, rng_base)
    pos7 += rng_base.normal(0, 0.2, size=pos7.shape)
    
    # 8. FCC noised
    pos8 = _fcc_lattice(n, rng_base)
    pos8 += rng_base.normal(0, 0.2, size=pos8.shape)
    
    candidates = [pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8]
    
    best_e = np.inf
    best_p = None
    
    # Determine time budget per worker
    # We have 8 workers.
    # Let's give each 30 seconds? 8*30 = 240s.
    # We have ~290s available.
    # So 35s per worker is safe.
    time_per_worker = 35.0
    
    tasks = []
    for i, pos in enumerate(candidates):
        seed = 42 + i * 17
        tasks.append((pos, kind, n, seed, time_per_worker))
    
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for t in tasks:
            if time.time() > deadline - 10.0:
                break
            futures.append(executor.submit(_basin_hopping_worker, t))
            
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 5.0:
                break
            try:
                e, p = future.result()
                if e < best_e:
                    best_e = e
                    best_p = p
            except Exception:
                pass

    # If we have time left, refine the best solution
    if best_p is not None and time.time() < deadline - 20.0:
        remaining_time = deadline - 20.0 - time.time()
        # Run a short BH on the best solution with larger steps
        rng_ref = np.random.default_rng(999)
        perturbed_best = best_p + rng_ref.normal(0, 1.0, size=best_p.shape)
        
        # Run 1 worker for remaining time
        task_ref = (perturbed_best, kind, n, 1000, remaining_time)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_basin_hopping_worker, task_ref)
            try:
                e, p = future.result()
                if e < best_e:
                    best_e = e
                    best_p = p
            except Exception:
                pass

    if best_p is None:
        # Fallback: return the best of the initial relaxations? 
        # Or just a random valid one.
        # Let's relax the first candidate one more time to ensure validity
        try:
            e, p = relax(candidates[0], 'sutton_chen', kind)
            if _validate(p, n):
                best_p = p
            else:
                best_p = _random_compact(n, np.random.default_rng(0))
        except Exception:
            best_p = _random_compact(n, np.random.default_rng(0))
            
    return np.asarray(best_p, dtype=float)

# EVOLVE-BLOCK-END
