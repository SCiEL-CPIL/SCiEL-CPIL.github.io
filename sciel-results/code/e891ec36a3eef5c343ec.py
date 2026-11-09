# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_HARD_LIMIT = 290.0  # Safety margin: 300s limit, keep 10s buffer
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


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = np.linalg.norm(positions[i] - positions[j])
            if rij < 1e-10:
                continue
            if rij < a_DZ:
                v = A_DZ * (rij ** (-m_DZ) - B_DZ) * np.exp(c_DZ / (rij - a_DZ))
            elif rij < b_DZ:
                v = B_DZ * np.exp(d_DZ / (rij - b_DZ))
            else:
                v = 0.0
            total += v
    return total


def _generate_mackay_icosahedron(n_atoms):
    """
    Generate an icosahedral cluster structure for n_atoms atoms.
    Uses concentric icosahedral shells.
    """
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    positions = []
    positions.append(np.zeros(3))
    
    r1 = 1.0
    positions.extend([v * r1 for v in ico_verts])
    
    remaining = n_atoms - len(positions)
    
    if remaining > 0:
        # Generate second shell points (edges and faces)
        ico_edges = []
        dists = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                dists.append((d, i, j))
        dists.sort()
        min_dist = dists[0][0]
        for d, i, j in dists:
            if d < min_dist * 1.01:
                ico_edges.append((i, j))
        
        edge_mids = []
        for i, j in ico_edges:
            mid = (ico_verts[i] + ico_verts[j]) / 2
            edge_mids.append(mid)
        edge_mids = np.array(edge_mids)
        
        connected = set()
        for i, j in ico_edges:
            connected.add((i, j))
            connected.add((j, i))
        
        face_centers = []
        for i in range(12):
            for j in range(i+1, 12):
                if (i, j) in connected:
                    for k in range(j+1, 12):
                        if (j, k) in connected and (i, k) in connected:
                            center = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
                            face_centers.append(center)
        face_centers = np.array(face_centers)
        
        r2 = r1 * np.sqrt(2)
        for mid in edge_mids:
            if remaining <= 0:
                break
            positions.append(mid * r2)
            remaining -= 1
        
        if remaining > 0:
            for fc in face_centers:
                if remaining <= 0:
                    break
                positions.append(fc * r2)
                remaining -= 1
    
    if remaining > 0:
        r3 = r1 * np.sqrt(3)
        rng = np.random.default_rng(42)
        for _ in range(remaining):
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            p = r3 * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            positions.append(p)
    
    return np.array(positions[:n_atoms])


def _generate_fcc_cluster(n_atoms):
    """Generate an FCC-like cluster."""
    positions = []
    a = 1.0
    max_coord = int(np.ceil(n_atoms ** 0.5)) + 2
    candidates = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    candidates.append(np.array([i, j, k]) * a / np.sqrt(2))
                candidates.append(np.array([i, j, k + 0.5]) * a / np.sqrt(2))
                candidates.append(np.array([i, j + 0.5, k]) * a / np.sqrt(2))
                candidates.append(np.array([i + 0.5, j, k]) * a / np.sqrt(2))
    
    candidates = np.array(candidates)
    centers = np.zeros(3)
    dists = np.linalg.norm(candidates - centers, axis=1)
    order = np.argsort(dists)
    
    selected = []
    for idx in order:
        p = candidates[idx]
        if all(np.linalg.norm(p - q) >= 0.5 for q in selected):
            selected.append(p)
        if len(selected) == n_atoms:
            break
    
    rng = np.random.default_rng(n_atoms)
    while len(selected) < n_atoms:
        p = rng.uniform(-3, 3, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in selected):
            selected.append(p)
    
    return np.array(selected)


def _generate_random_compact(n_atoms, seed):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    spacing = 1.1
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.75
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        attempts += 1
    
    if len(points) < n_atoms:
        rng2 = np.random.default_rng(seed + 1)
        radius2 = spacing * n_atoms ** (1.0 / 3.0) * 0.6
        points2 = []
        attempts = 0
        while len(points2) < n_atoms and attempts < max_attempts:
            p = rng2.uniform(-radius2, radius2, 3)
            if np.linalg.norm(p) <= radius2:
                if all(np.linalg.norm(p - q) >= 0.5 for q in points2):
                    points2.append(p)
            attempts += 1
        if len(points2) >= n_atoms:
            points = points2
    
    while len(points) < n_atoms:
        p = np.array([rng.uniform(-radius, radius) for _ in range(3)])
        points.append(p)
    
    return np.array(points[:n_atoms])


def _generate_langevin_seed(n_atoms, seed):
    """
    Generate a seed via physics-driven self-assembly:
    sequentially add atoms at random positions near the growing cluster,
    then perform short relaxation steps.
    """
    rng = np.random.default_rng(seed)
    
    # Start with a small nucleus (center + a few atoms)
    positions = [np.zeros(3)]
    
    # Add first shell atoms around the center
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    # Add up to 12 atoms from icosahedral vertices with small random perturbation
    n_added = 0
    for v in ico_verts:
        if len(positions) >= n_atoms:
            break
        p = v * 1.0 + rng.normal(0, 0.05, 3)
        # Ensure minimum distance from existing atoms
        valid = True
        for q in positions:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            positions.append(p)
            n_added += 1
    
    # Now sequentially add remaining atoms via self-assembly
    while len(positions) < n_atoms:
        # Compute current cluster centroid and radius
        pos_arr = np.array(positions)
        centroid = pos_arr.mean(axis=0)
        if len(pos_arr) > 1:
            radius = np.max(np.linalg.norm(pos_arr - centroid, axis=1))
        else:
            radius = 1.0
        
        # Try to add an atom at a random position on the surface
        for attempt in range(50):
            # Random direction
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            direction = np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            
            # Place at radius + small offset
            r_new = radius + rng.uniform(-0.2, 0.3)
            p_new = centroid + direction * r_new
            
            # Check minimum distance
            valid = True
            for q in positions:
                if np.linalg.norm(p_new - q) < 0.5:
                    valid = False
                    break
            if valid:
                positions.append(p_new)
                break
        
        # If we couldn't add an atom after 50 attempts, place it anyway with random position
        if len(positions) < n_atoms and attempt >= 49:
            p_new = centroid + rng.normal(0, 0.5, 3)
            positions.append(p_new)
    
    positions = np.array(positions[:n_atoms])
    
    return positions


def _polish_with_gradient(positions, max_steps=200, tol=1e-10):
    """
    Final high-precision polish using gradient-based local optimization.
    Uses L-BFGS-B from scipy.optimize with tight tolerance.
    """
    from scipy.optimize import minimize
    
    x0 = positions.ravel()
    
    def obj_and_grad(x):
        p = x.reshape(-1, 3)
        e = _dzugutov_energy(p)
        g = dzugutov_energy_gradient(p).ravel()
        return e, g
    
    result = minimize(
        obj_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_steps, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
    )
    
    best_pos = result.x.reshape(-1, 3)
    best_e = result.fun
    
    return best_e, best_pos


def _basin_hopping(positions, potential, parameter, rng, max_iters, kick_scale_start, kick_scale_end, deadline):
    """
    Basin hopping: random kick + local relaxation.
    Returns (best_energy, best_positions).
    """
    best_energy, best_positions = relax(positions.copy(), potential, parameter)
    current_energy = best_energy
    current_positions = best_positions.copy()
    
    # Annealing schedule: start with larger kicks, decrease over iterations
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        t = it / max(1, max_iters - 1)
        # Exponential decay of kick scale for better exploration/exploitation balance
        kick_scale = kick_scale_start * np.exp(-2.0 * t) + kick_scale_end * (1 - np.exp(-2.0 * t))
        
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
            if delta_e < 0.15:
                # Metropolis acceptance with adaptive temperature
                temp = 0.05 * (1.0 + 0.5 * (1 - t))
                accept_prob = np.exp(-delta_e / temp)
                if rng.random() < accept_prob:
                    current_energy = new_energy
                    current_positions = new_relaxed
    
    return best_energy, best_positions


def _search_chain(seed_positions, potential, parameter, n_atoms, deadline, chain_id):
    """
    Run a basin hopping search chain starting from seed_positions.
    Returns (best_energy, best_positions).
    """
    rng = np.random.default_rng(chain_id * 1000 + n_atoms)
    
    # Increased iterations with time-aware budget
    elapsed = time.time()
    time_available = max(0.1, deadline - elapsed)
    
    # Estimate per-iteration cost (~0.3s for relax + kick)
    est_iter_cost = 0.3
    max_iters = max(10, int(time_available / est_iter_cost))
    max_iters = min(max_iters, 40)  # Cap to avoid too many
    
    kick_scale_start = 0.35
    kick_scale_end = 0.08
    
    best_energy, best_positions = _basin_hopping(
        seed_positions, potential, parameter, rng,
        max_iters, kick_scale_start, kick_scale_end, deadline
    )
    
    # Final polish pass with gradient-based optimizer if time permits
    if time.time() < deadline - 2.0 and best_positions is not None:
        try:
            polished_energy, polished_positions = _polish_with_gradient(best_positions, max_steps=100, tol=1e-12)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_positions = polished_positions
        except Exception:
            pass
    
    return best_energy, best_positions


def _generate_core_seeds(n_atoms):
    """
    Generate seeds based on known low-energy cores (Mackay icosahedron core)
    with geometric insertion of additional atoms.
    """
    seeds = []
    
    # 13-atom Mackay icosahedron as core
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    core = np.vstack([np.zeros(3), ico_verts * 1.0])  # 13 atoms
    
    if n_atoms <= 13:
        # Just use the icosahedron directly
        seeds.append(core[:n_atoms])
        return seeds
    
    # For larger clusters, insert additional atoms at optimal positions
    for seed_idx in range(3):
        rng = np.random.default_rng(seed_idx * 100 + n_atoms)
        positions = core.copy()
        
        for _ in range(n_atoms - len(core)):
            # Find position that minimizes energy contribution
            best_pos = None
            best_e = float('inf')
            
            # Try positions near existing atoms (polytetrahedral preference)
            for attempt in range(30):
                # Pick a random existing atom and place new atom nearby
                ref_idx = rng.integers(0, len(positions))
                ref_pos = positions[ref_idx]
                
                # Random direction
                theta = rng.uniform(0, 2 * np.pi)
                cos_phi = rng.uniform(-1, 1)
                sin_phi = np.sqrt(1 - cos_phi**2)
                direction = np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
                
                # Place at typical bond distance
                r_new = rng.uniform(0.9, 1.2)
                candidate = ref_pos + direction * r_new
                
                # Check minimum distance
                if all(np.linalg.norm(candidate - q) >= 0.5 for q in positions):
                    # Quick energy estimate
                    e = 0.0
                    for q in positions:
                        d = np.linalg.norm(candidate - q)
                        if d < a_DZ:
                            e += A_DZ * (d ** (-m_DZ) - B_DZ) * np.exp(c_DZ / (d - a_DZ))
                        elif d < b_DZ:
                            e += B_DZ * np.exp(d_DZ / (d - b_DZ))
                    
                    if e < best_e:
                        best_e = e
                        best_pos = candidate
            
            if best_pos is not None:
                positions = np.vstack([positions, best_pos])
            else:
                # Fallback: random position
                p = rng.uniform(-2, 2, 3)
                if all(np.linalg.norm(p - q) >= 0.5 for q in positions):
                    positions = np.vstack([positions, p])
        
        seeds.append(positions[:n_atoms])
    
    return seeds


def construct(instance):
    """
    Find low-energy Dzugutov cluster structures using basin hopping
    with diverse polytetrahedral seeds and gradient-based polishing.
    """
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Calculate time budget
    elapsed = time.time() - _START_TIME
    remaining_total = max(0, _HARD_LIMIT - elapsed)
    
    # Determine how many instances are left, including this one
    idx = _INSTANCE_INDEX.get(size, 0)
    n_rem = len(_INSTANCE_ORDER) - idx
    
    # Allocate time for this call: share remaining time among remaining instances
    if n_rem > 0:
        call_budget = remaining_total / n_rem
    else:
        call_budget = remaining_total
    
    # Ensure we don't allocate negative or zero time
    call_budget = max(call_budget, 1.0)
    
    call_deadline = time.time() + call_budget
    
    # Generate diverse seed structures
    seeds = []
    
    # Core-based seeds (highest priority for polytetrahedral metals)
    try:
        core_seeds = _generate_core_seeds(size)
        seeds.extend(core_seeds)
    except Exception:
        pass
    
    try:
        ico = _generate_mackay_icosahedron(size)
        seeds.append(ico)
    except Exception:
        pass
    
    try:
        fcc = _generate_fcc_cluster(size)
        seeds.append(fcc)
    except Exception:
        pass
    
    # Add Langevin-style self-assembly seed
    try:
        lang = _generate_langevin_seed(size, 99)
        seeds.append(lang)
    except Exception:
        pass
    
    # Add random seeds based on available time
    n_random_seeds = 3
    for seed_val in range(n_random_seeds):
        try:
            rand = _generate_random_compact(size, seed_val + 1)
            seeds.append(rand)
        except Exception:
            pass
    
    if not seeds:
        rng = np.random.default_rng(size)
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        seeds.append(np.array(points))
    
    best_energy = float('inf')
    best_positions = None
    
    # If time is very tight, just relax the best seed
    if call_budget < 5.0:
        try:
            _, best_positions = relax(seeds[0].copy(), potential, parameter)
        except Exception:
            best_positions = seeds[0]
        return best_positions
    
    # Use all 4 threads with up to 4 seeds
    n_seeds_to_use = min(4, len(seeds))
    
    def run_chain(i):
        if time.time() > call_deadline:
            return (float('inf'), None)
        return _search_chain(seeds[i], potential, parameter, size, call_deadline, i)
    
    with ThreadPoolExecutor(max_workers=min(4, n_seeds_to_use)) as executor:
        futures = {executor.submit(run_chain, i): i for i in range(n_seeds_to_use)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                timeout = max(0.1, call_deadline - time.time())
                energy, positions = future.result(timeout=timeout)
                if positions is not None and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                continue
    
    # Final global polish if time permits
    if best_positions is not None and time.time() < call_deadline - 3.0:
        try:
            polished_energy, polished_positions = _polish_with_gradient(
                best_positions, max_steps=150, tol=1e-13
            )
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_positions = polished_positions
        except Exception:
            pass
    
    if best_positions is None:
        try:
            _, best_positions = relax(seeds[0].copy(), potential, parameter)
        except Exception:
            best_positions = seeds[0]
    
    return best_positions
# EVOLVE-BLOCK-END
