"""Morse cluster global minima via diverse crystalline starts + multi-scale basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(positions, rho):
    """Compute total Morse energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    dist = np.maximum(dist, 1e-10)
    x = rho * (1.0 - dist)
    exp_x = np.exp(x)
    pair_energy = exp_x * (exp_x - 2.0)
    return 0.5 * np.sum(pair_energy)


def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like cluster of n atoms."""
    points = []
    # FCC lattice vectors
    a = spacing * 1.0  # equilibrium distance
    # Generate points in FCC structure
    max_r = a * (n ** (1.0/3.0)) * 1.5
    nx = int(2 * max_r / a) + 2
    # FCC basis
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]) * a
    # Generate candidate points
    candidates = []
    for i in range(-nx//2, nx//2 + 1):
        for j in range(-nx//2, nx//2 + 1):
            for k in range(-nx//2, nx//2 + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    if np.linalg.norm(p) <= max_r:
                        candidates.append(p)
    candidates = np.array(candidates)
    # Sort by distance from center and take first n
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    return candidates[order[:n]].copy()


def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like cluster of n atoms."""
    a = spacing * 1.0
    max_r = a * (n ** (1.0/3.0)) * 1.5
    nx = int(2 * max_r / a) + 2
    # BCC basis
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ]) * a
    candidates = []
    for i in range(-nx//2, nx//2 + 1):
        for j in range(-nx//2, nx//2 + 1):
            for k in range(-nx//2, nx//2 + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    if np.linalg.norm(p) <= max_r:
                        candidates.append(p)
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    return candidates[order[:n]].copy()


def _generate_hcp(n, spacing=1.0):
    """Generate HCP-like cluster of n atoms."""
    a = spacing * 1.0
    # HCP: close-packed layers with AB stacking
    max_r = a * (n ** (1.0/3.0)) * 1.5
    nx = int(2 * max_r / a) + 2
    # Layer 1: triangular lattice
    # Layer 2: shifted
    candidates = []
    a1 = a
    a2 = a * np.sqrt(3) / 2
    c = a * np.sqrt(2/3)  # interlayer spacing for HCP
    for layer in range(int(2 * max_r / c) + 1):
        z = (layer - int(max_r / c)) * c
        offset = 0 if layer % 2 == 0 else 0.5
        for i in range(-nx, nx):
            for j in range(-nx, nx):
                x = (i + offset) * a1
                y = j * a2
                if abs(j) % 2 == 1:
                    x += a1 / 2
                p = np.array([x, y, z])
                if np.linalg.norm(p) <= max_r:
                    candidates.append(p)
    if len(candidates) < n:
        # Fallback to FCC
        return _generate_fcc(n, spacing)
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    return candidates[order[:n]].copy()


def _generate_shell(n, spacing=1.0):
    """Generate spherical shell-like cluster."""
    points = []
    # Inner core
    core_size = max(1, int(n ** 0.5))
    rng = np.random.default_rng(42)
    # Core: small FCC-like
    if core_size > 0:
        core = _generate_fcc(core_size, spacing)
        points.extend(core.tolist())
    # Shell layers
    remaining = n - len(points)
    r_shell = spacing * (n ** (1.0/3.0)) * 0.6
    while len(points) < n and remaining > 0:
        p = rng.uniform(-r_shell, r_shell, 3)
        if np.linalg.norm(p) >= r_shell * 0.8 and np.linalg.norm(p) <= r_shell * 1.2:
            # Check minimum distance
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5 * spacing:
                    ok = False
                    break
            if ok:
                points.append(p)
                remaining -= 1
    # If not enough, add random points
    while len(points) < n:
        p = rng.uniform(-r_shell * 1.5, r_shell * 1.5, 3)
        ok = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5 * spacing:
                ok = False
                break
        if ok:
            points.append(p)
    return np.array(points[:n])


def _generate_random(n, spacing=1.0, seed=0):
    """Generate random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 1000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < 0.6 * spacing:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    # Fill remaining if needed
    while len(points) < n:
        p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
        points.append(p)
    return np.array(points[:n])


def _local_relax(positions, potential, parameter, max_iter=500):
    """Local relaxation using gradient-based method."""
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return energy, relaxed
    except Exception:
        return _morse_energy(positions, parameter), positions


def _basin_hop(positions, rho, n, start_time, deadline, rng):
    """Multi-scale basin hopping: perturb, relax, accept/reject."""
    current_energy = _morse_energy(positions, rho)
    best_energy = current_energy
    best_positions = positions.copy()
    
    # Temperature schedule
    T = 0.5
    T_min = 0.05
    perturbation_scale = 0.5
    
    while time.time() < deadline:
        # Perturb
        noise = rng.normal(0, perturbation_scale, size=(n, 3))
        perturbed = positions + noise
        
        # Check minimum distance
        if n > 1:
            diff = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dist, np.inf)
            if np.min(dist) < 0.5:
                # Rescale to fix
                continue
        
        # Relax
        try:
            new_energy, new_positions = relax(perturbed, 'morse', rho)
        except Exception:
            continue
        
        # Metropolis acceptance
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            positions = new_positions
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_positions.copy()
        
        # Anneal
        T = max(T * 0.98, T_min)
        perturbation_scale = max(0.2, perturbation_scale * 0.99)
    
    return best_energy, best_positions


def construct(instance):
    """Find low-energy Morse cluster configurations using diverse starts and basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    rng = np.random.default_rng(size)
    
    # Generate diverse starting configurations
    starts = []
    
    # FCC
    starts.append(_generate_fcc(size, 1.0))
    # BCC
    starts.append(_generate_bcc(size, 1.0))
    # HCP
    starts.append(_generate_hcp(size, 1.0))
    # Shell
    starts.append(_generate_shell(size, 1.0))
    # Random (different seeds)
    for seed in [0, 1, 2]:
        starts.append(_generate_random(size, 1.0, seed=seed + 100))
    
    # Relax each start locally
    best_energy = np.inf
    best_positions = None
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 30:
            break
        try:
            energy, relaxed = relax(start, potential, parameter)
        except Exception:
            energy = _morse_energy(start, parameter)
            relaxed = start
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed.copy()
    
    if best_positions is None:
        best_positions = starts[0]
        best_energy = _morse_energy(best_positions, parameter)
    
    # Basin hopping from the best start
    if time.time() < deadline - 10:
        bh_energy, bh_positions = _basin_hop(
            best_positions, parameter, size, time.time(), deadline, rng
        )
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_positions
    
    # Final check: ensure minimum distance constraint
    if size > 1:
        diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 0.5:
            # Nudge atoms apart
            for _ in range(100):
                min_idx = np.unravel_index(np.argmin(dist), dist.shape)
                i, j = min_idx
                if i == j:
                    break
                direction = best_positions[i] - best_positions[j]
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    direction = rng.normal(0, 1, 3)
                    norm = np.linalg.norm(direction)
                direction /= norm
                shift = (0.5 - norm) / 2 + 0.01
                best_positions[i] += shift * direction
                best_positions[j] -= shift * direction
                # Recompute distances
                diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff * diff, axis=2))
                np.fill_diagonal(dist, np.inf)
                if np.min(dist) >= 0.5:
                    break
    
    return best_positions
# EVOLVE-BLOCK-END
