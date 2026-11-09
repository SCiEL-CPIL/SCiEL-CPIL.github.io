"""Improved Dzugutov cluster search with PT motif seeding and adaptive basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

def _generate_pt_shells(n, seed=42):
    """
    Generate polytetrahedral (PT) cluster using icosahedral shell construction.
    PT motifs are icosahedral shells: 13 (1+12), 55 (1+12+42), etc.
    For arbitrary n, we build the largest complete shell <= n, then add
    remaining atoms in a greedy manner near the shell surface.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (unit)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0])
    
    # Shell radii for icosahedral packing (approximate)
    # Shell 0: center (1 atom)
    # Shell 1: 12 atoms at radius r1
    # Shell 2: 42 atoms (30 edge midpoints + 12 face centers... actually it's more complex)
    # For Mackay: 13 + 30 + 55 = 98, but that's not quite right either.
    # Let's use a simpler approach: build concentric icosahedral shells.
    
    points = [np.zeros(3)]  # Center
    
    # Shell 1: 12 icosahedron vertices
    r1 = 1.0
    shell1 = verts * r1
    points.extend(shell1.tolist())  # Total: 13
    
    if n <= 13:
        return np.array(points[:n])
    
    # Shell 2: 30 atoms - edge midpoints of icosahedron
    # Find all edges (pairs with minimum distance)
    d_min = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < d_min - 1e-6:
                d_min = d
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(np.linalg.norm(verts[i] - verts[j]) - d_min) < 1e-5:
                edges.append((i, j))
    
    # Edge midpoints (30 edges)
    r2 = np.sqrt(r1**2 + (d_min/2)**2)  # Approximate radius for edge midpoints
    shell2 = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        # Normalize to radius r2
        mid_dir = mid / np.linalg.norm(mid)
        shell2.append(mid_dir * r2)
    points.extend(shell2.tolist())  # Total: 13 + 30 = 43
    
    if n <= 43:
        return np.array(points[:n])
    
    # Shell 3: Add more atoms. Use face centers or additional shell.
    # For simplicity, add atoms in a spherical shell at radius r3
    r3 = r2 * 1.3
    rng = np.random.default_rng(seed)
    
    remaining = n - len(points)
    current_points = np.array(points)
    min_dist = 0.7
    
    # Generate candidates in a shell
    candidates = []
    for _ in range(remaining * 10):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        p = v * r3 * rng.uniform(0.9, 1.1)
        candidates.append(p)
    
    candidates = np.array(candidates)
    
    # Greedily select candidates that maintain min distance
    selected = []
    for p in candidates:
        if len(selected) >= remaining:
            break
        dists = np.linalg.norm(current_points - p, axis=1)
        if np.min(dists) >= min_dist:
            # Also check against already selected in this batch
            if selected:
                sel_dists = np.linalg.norm(np.array(selected) - p, axis=1)
                if np.min(sel_dists) >= min_dist:
                    selected.append(p)
            else:
                selected.append(p)
    
    # If we didn't get enough, relax the min distance
    if len(selected) < remaining:
        min_dist2 = 0.55
        for p in candidates:
            if len(selected) >= remaining:
                break
            if p in selected:
                continue
            dists = np.linalg.norm(current_points - p, axis=1)
            if np.min(dists) >= min_dist2:
                if selected:
                    sel_dists = np.linalg.norm(np.array(selected) - p, axis=1)
                    if np.min(sel_dists) >= min_dist2:
                        selected.append(p)
                else:
                    selected.append(p)
    
    # Fallback: just add random points
    while len(selected) < remaining:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        p = v * r3
        selected.append(p)
    
    points.extend(selected)
    return np.array(points[:n])

def _generate_fcc_cluster(n, seed=123):
    """Generate a truncated FCC cluster."""
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ])
    
    N = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            for k in range(-N, N+1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    candidates.append(p)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)
    candidates = candidates[indices[:n]]
    
    centroid = np.mean(candidates, axis=0)
    candidates -= centroid
    
    return candidates

def _generate_random_packing(n, seed):
    """Generate a random compact packing."""
    rng = np.random.default_rng(seed)
    points = []
    radius = 1.2 * n ** (1.0 / 3.0)
    min_dist = 0.75
    attempts = 0
    max_attempts = n * 100
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def _basin_hopping(pos, potential, parameter, rng, max_steps, deadline):
    """
    Adaptive basin hopping with multiple perturbation scales.
    Returns (best_energy, best_position).
    """
    current_energy, current_pos = relax(pos, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    # Multiple perturbation scales for diversity
    base_sigma = 0.25
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Adaptive sigma: start large, decay slowly
        sigma = base_sigma * (1.0 + 0.5 * np.cos(step * 0.3))
        sigma = max(sigma, 0.1)
        
        # Multi-scale perturbation
        noise = rng.normal(0, sigma, size=current_pos.shape)
        perturbed = current_pos + noise
        
        try:
            new_energy, new_pos = relax(perturbed, potential, parameter)
        except Exception:
            continue
        
        # Accept if better, or with some probability for exploration
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        elif rng.random() < 0.1:  # 10% chance to accept worse for exploration
            current_energy = new_energy
            current_pos = new_pos
    
    return best_energy, best_pos

def construct(instance):
    """
    Multi-start search with PT motif seeding and adaptive basin hopping.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. PT motif (icosahedral shells) - primary seed
    try:
        p_pt = _generate_pt_shells(size, seed=42)
        candidates.append(('pt', p_pt))
    except Exception:
        pass
    
    # 2. PT motif with different seed
    try:
        p_pt2 = _generate_pt_shells(size, seed=123)
        candidates.append(('pt2', p_pt2))
    except Exception:
        pass
    
    # 3. FCC cluster
    try:
        p_fcc = _generate_fcc_cluster(size)
        candidates.append(('fcc', p_fcc))
    except Exception:
        pass
    
    # 4. Random packings
    for seed in [1, 2, 3, 4]:
        try:
            p_rand = _generate_random_packing(size, seed)
            candidates.append((f'rand{seed}', p_rand))
        except Exception:
            pass
    
    # Relax all candidates and track best
    best_energy = np.inf
    best_pos = None
    best_name = None
    
    for name, pos in candidates:
        if time.time() > deadline - 30:
            break
        try:
            energy, relaxed_pos = relax(pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos
                best_name = name
        except Exception:
            continue
    
    if best_pos is None:
        # Fallback
        if candidates:
            return candidates[0][1]
        return np.array([np.array([i, 0, 0]) for i in range(size)])
    
    # Basin hopping refinement on the best structure
    rng = np.random.default_rng(size + 1000)
    
    # Estimate time per relax and set max steps
    elapsed = time.time() - start_time
    remaining_time = deadline - time.time()
    # Assume ~2s per relax+gradient step
    max_bh_steps = max(5, int(remaining_time / 3.0))
    
    bh_energy, bh_pos = _basin_hopping(best_pos, potential, parameter, rng, max_bh_steps, deadline)
    
    if bh_energy < best_energy:
        best_energy = bh_energy
        best_pos = bh_pos
    
    # If time remains, try basin hopping on second-best initial structure
    if time.time() < deadline - 20:
        # Find second best
        second_best_energy = np.inf
        second_best_pos = None
        for name, pos in candidates:
            if name == best_name:
                continue
            try:
                energy, relaxed_pos = relax(pos, potential, parameter)
                if energy < second_best_energy and energy > best_energy:
                    second_best_energy = energy
                    second_best_pos = relaxed_pos
            except Exception:
                continue
        
        if second_best_pos is not None:
            remaining = deadline - time.time()
            max_steps2 = max(3, int(remaining / 4.0))
            bh2_energy, bh2_pos = _basin_hopping(second_best_pos, potential, parameter, rng, max_steps2, deadline)
            if bh2_energy < best_energy:
                best_energy = bh2_energy
                best_pos = bh2_pos
    
    return best_pos
# EVOLVE-BLOCK-END
