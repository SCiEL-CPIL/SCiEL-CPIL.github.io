"""
Morse cluster global minima via Parallel Annealed Basin Hopping (PABH).
Strategy:
1. Generate diverse structural seeds (FCC, BCC, Icosahedral, Shells).
2. Run multiple parallel basin hopping chains (ensemble).
3. Use a 'cross' operator: occasionally mutate a chain from another chain's state to promote diversity.
4. Time-slice execution to ensure all instances are handled within the budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
_DEADLINE = _START_TIME + 285.0  # 15s safety margin before 300s limit

def _time_remaining():
    return _DEADLINE - time.time()

def _morse_energy(coords, rho):
    """Compute Morse energy; return inf if any pair < 0.5 apart."""
    n = len(coords)
    if n < 2:
        return 0.0
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        return np.inf
    iu = np.triu_indices(n, 1)
    r = dists[iu]
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    return float(energy)

def _ensure_valid(coords, rho, rng, n):
    """Ensure coordinates are valid (min distance >= 0.5)."""
    if len(coords) != n:
        return _build_random_compact(n, rng)
    # Check distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        # Relax to fix
        try:
            energy, relaxed = relax(coords, 'morse', rho)
            if np.isfinite(energy):
                # Check again
                diff = relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=-1))
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) >= 0.5:
                    return relaxed
        except:
            pass
        # Fallback: random
        return _build_random_compact(n, rng)
    return coords

def _build_fcc_cluster(n, spacing=1.0):
    """Build an FCC-like compact cluster."""
    points = []
    # Generate FCC lattice points
    a = spacing * 0.9
    # Center at origin
    # Loop over a cube large enough to contain n points
    max_idx = int(np.ceil((n / 4) ** (1/3))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC basis
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k]) * a
                p3 = np.array([i + 0.5, j, k + 0.5]) * a
                p4 = np.array([i, j + 0.5, k + 0.5]) * a
                for p in [p1, p2, p3, p4]:
                    if len(points) < n:
                        points.append(p)
    if len(points) < n:
        # Fill with random
        while len(points) < n:
            p = np.array(points[0]) + np.random.uniform(-0.5, 0.5, 3)
            points.append(p)
    points = np.array(points[:n])
    # Center
    points -= np.mean(points, axis=0)
    return points

def _build_bcc_cluster(n, spacing=1.0):
    """Build a BCC-like compact cluster."""
    points = []
    a = spacing * 1.0
    max_idx = int(np.ceil((n / 2) ** (1/3))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                for p in [p1, p2]:
                    if len(points) < n:
                        points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = np.array(points[0]) + np.random.uniform(-0.5, 0.5, 3)
            points.append(p)
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _build_icosahedral_shells(n, spacing=1.0):
    """Build icosahedral shells."""
    points = []
    # First shell: 12 vertices of icosahedron
    phi = (1 + np.sqrt(5)) / 2
    r = spacing * 1.0
    # Vertices
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    for v in verts:
        v = np.array(v)
        v = v / np.linalg.norm(v) * r
        points.append(v)
    
    # Second shell: 30 edges (midpoints)
    if n > 12:
        edges = [
            (0,1), (0,2), (0,4), (0,8), (1,3), (1,4), (1,9), (2,3), (2,5), (2,8),
            (3,5), (3,9), (4,6), (4,7), (4,9), (5,7), (5,10), (5,11), (6,7), (6,8),
            (6,10), (7,11), (8,10), (8,12), (9,11), (9,13), (10,12), (10,14), (11,13), (11,14)
        ]
        # Wait, indices in verts are 0-11. Let's just use midpoints of all pairs in first shell that are close
        # Simpler: add 30 points at radius r*1.5
        for i in range(n - 12):
            angle1 = np.random.uniform(0, 2*np.pi)
            angle2 = np.random.uniform(0, np.pi)
            r2 = r * 1.5
            x = r2 * np.sin(angle2) * np.cos(angle1)
            y = r2 * np.sin(angle2) * np.sin(angle1)
            z = r2 * np.cos(angle2)
            points.append([x, y, z])
            
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _build_spherical_shells(n, spacing=1.0):
    """Build spherical shells."""
    points = []
    shells = [1, 12, 30, 55, 91] # Magic numbers
    radii = [0, 1.0, 1.5, 2.0, 2.5]
    
    # Center
    points.append([0,0,0])
    
    # Shell 1: 12
    if n > 1:
        phi = (1 + np.sqrt(5)) / 2
        verts = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ]
        for v in verts[:min(12, n-1)]:
            v = np.array(v)
            v = v / np.linalg.norm(v) * radii[1]
            points.append(v)
            
    # Remaining
    while len(points) < n:
        # Add to outer shell
        shell_idx = min(len(radii)-1, int((len(points)**0.5)**0.5))
        r = radii[shell_idx]
        angle1 = np.random.uniform(0, 2*np.pi)
        angle2 = np.random.uniform(0, np.pi)
        x = r * np.sin(angle2) * np.cos(angle1)
        y = r * np.sin(angle2) * np.sin(angle1)
        z = r * np.cos(angle2)
        points.append([x, y, z])
        
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _build_random_compact(n, rng, spacing=1.0):
    """Generate a compact random cluster."""
    points = []
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    min_dist = 0.55 * spacing
    max_attempts = 100000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            attempts += 1
            continue
        ok = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                ok = False
                break
        if ok:
            points.append(p)
        attempts += 1
    while len(points) < n:
        base = points[rng.integers(len(points))] if points else np.zeros(3)
        p = base + rng.uniform(-0.1, 0.1, 3)
        points.append(p)
    return np.array(points)

def _relax_and_check(coords, rho):
    """Relax coordinates and check validity."""
    try:
        energy, relaxed = relax(coords, 'morse', rho)
        if np.isfinite(energy):
            # Verify min distance
            diff = relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5:
                return energy, relaxed
    except:
        pass
    return np.inf, coords

def _basin_hopping_ensemble(n, rho, rng, time_limit, initial_seeds):
    """
    Run parallel annealed basin hopping.
    initial_seeds: list of initial coordinate arrays.
    """
    start_time = time.time()
    n_chains = len(initial_seeds)
    
    # Initialize chains
    chains = []
    best_energies = []
    best_coords = []
    
    for i, seed in enumerate(initial_seeds):
        seed = _ensure_valid(seed, rho, rng, n)
        energy, relaxed = _relax_and_check(seed, rho)
        if not np.isfinite(energy):
            energy = _morse_energy(relaxed, rho)
            if not np.isfinite(energy):
                # Reset to random
                relaxed = _build_random_compact(n, rng)
                energy, relaxed = _relax_and_check(relaxed, rho)
        chains.append(relaxed)
        best_energies.append(energy)
        best_coords.append(relaxed.copy())
        
    # Global best
    global_best_idx = np.argmin(best_energies)
    global_best_energy = best_energies[global_best_idx]
    global_best_coords = best_coords[global_best_idx].copy()
    
    base_step = 0.15
    step_scales = [1.0] * n_chains
    
    iteration = 0
    while _time_remaining() > 5.0:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # Choose a chain to update (round robin or random)
        chain_idx = rng.integers(n_chains)
        
        # Decide if we 'cross' (mutate from another chain)
        # Cross probability increases with stagnation
        cross_prob = 0.1 + 0.1 * (iteration // 100)
        if rng.random() < cross_prob:
            # Pick another chain
            other_idx = rng.integers(n_chains)
            if other_idx != chain_idx:
                # Mix states
                alpha = rng.uniform(0.3, 0.7)
                new_coords = alpha * chains[chain_idx] + (1 - alpha) * chains[other_idx]
            else:
                new_coords = chains[chain_idx]
        else:
            # Standard perturbation
            step = base_step * step_scales[chain_idx]
            new_coords = chains[chain_idx] + rng.normal(0, step, (n, 3))
            
        # Relax
        energy, relaxed = _relax_and_check(new_coords, rho)
        
        if np.isfinite(energy):
            # Metropolis accept/reject for the chain
            delta_e = energy - best_energies[chain_idx]
            temp = 0.1 + 0.05 * step_scales[chain_idx]
            if delta_e < 0 or rng.random() < np.exp(-delta_e / temp):
                chains[chain_idx] = relaxed
                best_energies[chain_idx] = energy
                best_coords[chain_idx] = relaxed.copy()
                step_scales[chain_idx] *= 0.98
            else:
                step_scales[chain_idx] = min(2.0, step_scales[chain_idx] * 1.05)
                
            # Update global best
            if energy < global_best_energy:
                global_best_energy = energy
                global_best_coords = relaxed.copy()
                global_best_idx = chain_idx
        else:
            # Invalid, reset step
            step_scales[chain_idx] = min(2.0, step_scales[chain_idx] * 1.2)
            
        iteration += 1
        
        # Periodic large kick
        if iteration % 50 == 0:
            for i in range(n_chains):
                step_scales[i] = max(0.3, step_scales[i] * 1.1)
                
    return global_best_energy, global_best_coords

def construct(instance):
    """Find global minimum of Morse cluster via PABH."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Allocate time based on cluster size
    if n <= 40:
        time_limit = 40.0
    elif n <= 60:
        time_limit = 50.0
    else:
        time_limit = 60.0
        
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse seeds
    seeds = []
    seeds.append(_build_fcc_cluster(n))
    seeds.append(_build_bcc_cluster(n))
    seeds.append(_build_icosahedral_shells(n))
    seeds.append(_build_spherical_shells(n))
    seeds.append(_build_random_compact(n, rng))
    
    # Filter valid seeds
    valid_seeds = []
    for seed in seeds:
        if len(seed) == n:
            valid_seeds.append(seed)
            
    if not valid_seeds:
        valid_seeds = [_build_random_compact(n, rng)]
        
    # Run PABH
    best_energy, best_coords = _basin_hopping_ensemble(n, rho, rng, time_limit, valid_seeds)
    
    # Final check
    if best_coords is None or not np.isfinite(best_energy):
        # Fallback
        coords = _build_random_compact(n, rng)
        best_energy, best_coords = _relax_and_check(coords, rho)
        
    # Ensure final validity
    best_coords = _ensure_valid(best_coords, rho, rng, n)
    
    return best_coords
# EVOLVE-BLOCK-END
