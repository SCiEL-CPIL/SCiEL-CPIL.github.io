"""
Morse cluster global minima via Deterministic Multi-Start Basin Hopping.
Strategy:
1. Generate a rich library of structurally diverse seeds:
   - Optimized Spherical Shells (Magic numbers)
   - FCC and BCC lattice fragments
   - Greedy dense packing
   - Random compact clusters
2. Pre-relax all seeds to local minima.
3. Perform a single, global best-tracking basin hopping loop.
   - Each step perturbs the current global best.
   - If the perturbation leads to a lower energy (after relaxation), it becomes the new best.
   - If rejected, we keep the best but might occasionally jump to a random seed or a previous local min to escape local traps (simple restart logic).
4. This avoids the overhead and potential dilution of parallel chains, focusing all compute on exploring the neighborhood of the best-known structure while maintaining diversity via occasional restarts from the seed pool.
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

def _build_fcc_cluster(n, spacing=1.0):
    """Build an FCC-like compact cluster."""
    points = []
    a = spacing * 0.9
    max_idx = int(np.ceil((n / 4) ** (1/3))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k]) * a
                p3 = np.array([i + 0.5, j, k + 0.5]) * a
                p4 = np.array([i, j + 0.5, k + 0.5]) * a
                for p in [p1, p2, p3, p4]:
                    if len(points) < n:
                        points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = np.array(points[0]) + np.random.uniform(-0.5, 0.5, 3)
            points.append(p)
    points = np.array(points[:n])
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
    """Build icosahedral shells with optimized magic numbers."""
    points = []
    # Magic numbers for icosahedral shells: 1, 12, 42, 100, 192...
    # But we need exactly n. We'll build shells and trim/pad.
    
    # Shell 0: Center
    points.append([0, 0, 0])
    
    # Shell 1: 12 vertices of icosahedron
    if n > 1:
        phi = (1 + np.sqrt(5)) / 2
        r1 = spacing * 1.0
        verts = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ]
        for v in verts:
            v = np.array(v)
            v = v / np.linalg.norm(v) * r1
            points.append(v)
            if len(points) >= n:
                break
        
    # Shell 2: 30 edge midpoints (approx) or face centers? 
    # Standard icosahedron has 30 edges. Midpoints of edges form a dodecahedron shell.
    if n > 13:
        r2 = spacing * 1.5
        # Generate 30 points roughly on a dodecahedron
        # Simplified: place points on a sphere of radius r2
        # We need 30 points.
        num_to_add = min(30, n - len(points))
        for i in range(num_to_add):
            # Golden spiral distribution for even coverage
            i_angle = 2 * np.pi * i / 30
            i_phi = np.arccos(1 - 2 * (i + 0.5) / 30)
            x = r2 * np.sin(i_phi) * np.cos(i_angle)
            y = r2 * np.sin(i_phi) * np.sin(i_angle)
            z = r2 * np.cos(i_phi)
            points.append([x, y, z])
            
    # Shell 3: 60 points?
    if n > 43:
        r3 = spacing * 2.0
        num_to_add = min(60, n - len(points))
        for i in range(num_to_add):
            i_angle = 2 * np.pi * i / 60
            i_phi = np.arccos(1 - 2 * (i + 0.5) / 60)
            x = r3 * np.sin(i_phi) * np.cos(i_angle)
            y = r3 * np.sin(i_phi) * np.sin(i_angle)
            z = r3 * np.cos(i_phi)
            points.append([x, y, z])
            
    # Pad if needed
    while len(points) < n:
        # Add to outermost shell
        r = spacing * (1.0 + 0.5 * (len(points) // 20))
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
    max_attempts = 50000
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

def _generate_seeds(n, rho, rng):
    """Generate a diverse set of initial seeds."""
    seeds = []
    seeds.append(_build_fcc_cluster(n))
    seeds.append(_build_bcc_cluster(n))
    seeds.append(_build_icosahedral_shells(n))
    seeds.append(_build_random_compact(n, rng))
    # Add a few more random ones
    for _ in range(2):
        seeds.append(_build_random_compact(n, rng))
        
    # Relax all seeds
    relaxed_seeds = []
    energies = []
    for seed in seeds:
        if len(seed) != n:
            continue
        e, c = _relax_and_check(seed, rho)
        if np.isfinite(e):
            relaxed_seeds.append(c)
            energies.append(e)
            
    if not relaxed_seeds:
        # Fallback
        c = _build_random_compact(n, rng)
        e, c = _relax_and_check(c, rho)
        if np.isfinite(e):
            relaxed_seeds.append(c)
            energies.append(e)
        else:
            return [c], [np.inf]
            
    return relaxed_seeds, energies

def construct(instance):
    """Find global minimum of Morse cluster via Deterministic Multi-Start BH."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate and relax seeds
    seeds, seed_energies = _generate_seeds(n, rho, rng)
    
    # Find initial best
    best_idx = np.argmin(seed_energies)
    best_energy = seed_energies[best_idx]
    best_coords = seeds[best_idx].copy()
    
    # Store all local minima found so far for potential restarts
    local_mins = [(seeds[i], seed_energies[i]) for i in range(len(seeds))]
    
    # Basin Hopping Loop
    # We will use a simple strategy:
    # 1. Perturb current best.
    # 2. Relax.
    # 3. If better, update best.
    # 4. If rejected, occasionally restart from a random local min or seed to avoid getting stuck.
    
    base_step = 0.2
    step = base_step
    iterations = 0
    stagnation_count = 0
    max_stagnation = 20
    
    while _time_remaining() > 5.0:
        # Perturb
        perturbed = best_coords + rng.normal(0, step, (n, 3))
        
        # Relax
        energy, relaxed = _relax_and_check(perturbed, rho)
        
        if np.isfinite(energy):
            if energy < best_energy:
                best_energy = energy
                best_coords = relaxed.copy()
                stagnation_count = 0
                step = max(0.05, step * 0.95)  # Shrink step on success
            else:
                stagnation_count += 1
                step = min(0.5, step * 1.05)  # Grow step on failure
                
                # If stagnating, try a restart from a different local min
                if stagnation_count > max_stagnation:
                    # Pick a random local min or seed
                    if local_mins:
                        restart_idx = rng.integers(len(local_mins))
                        restart_coords, restart_energy = local_mins[restart_idx]
                        # Use this as the new base for perturbation
                        best_coords = restart_coords.copy()
                        best_energy = restart_energy
                        # Also add the current best to local mins if it's good
                        if best_energy < local_mins[0][1]:
                            local_mins.insert(0, (best_coords.copy(), best_energy))
                        # Truncate list to keep it small
                        if len(local_mins) > 10:
                            local_mins = local_mins[:10]
                        
                    stagnation_count = 0
                    step = base_step
        else:
            # Invalid, increase step
            step = min(0.5, step * 1.2)
            stagnation_count += 1
            
        iterations += 1
        
        # Periodic check: if we have time, maybe add a new random seed to the pool?
        if iterations % 100 == 0 and _time_remaining() > 10.0:
            new_seed = _build_random_compact(n, rng)
            e, c = _relax_and_check(new_seed, rho)
            if np.isfinite(e) and e < best_energy:
                best_energy = e
                best_coords = c.copy()
            elif np.isfinite(e):
                local_mins.append((c, e))
                if len(local_mins) > 10:
                    local_mins.pop()

    # Final check
    if best_coords is None or not np.isfinite(best_energy):
        coords = _build_random_compact(n, rng)
        best_energy, best_coords = _relax_and_check(coords, rho)
        
    return best_coords
# EVOLVE-BLOCK-END
