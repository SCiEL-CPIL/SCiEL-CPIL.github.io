"""Morse cluster global minima: diverse crystalline starts + multi-scale basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _morse_energy(positions, rho):
    """Compute total Morse energy for a configuration."""
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

def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are at least min_dist."""
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return np.min(dist) >= min_dist

def _generate_fcc(n):
    """Generate an FCC-like cluster of n atoms."""
    a = 1.0  # lattice constant (equilibrium distance)
    # FCC basis: (0,0,0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]])
    atoms = []
    # Generate FCC lattice points within a sphere
    max_coord = int(np.ceil((n / 4) ** (1/3)) + 2)
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([ix, iy, iz]) + b
                    atoms.append(p)
    atoms = np.array(atoms)
    # Sort by distance from center
    center = np.mean(atoms, axis=0)
    dists = np.linalg.norm(atoms - center, axis=1)
    order = np.argsort(dists)
    atoms = atoms[order]
    # Take first n
    atoms = atoms[:n]
    # Center
    atoms = atoms - np.mean(atoms, axis=0)
    return atoms

def _generate_bcc(n):
    """Generate a BCC-like cluster of n atoms."""
    a = 1.0
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    atoms = []
    max_coord = int(np.ceil((n / 2) ** (1/3)) + 2)
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([ix, iy, iz]) + b
                    atoms.append(p)
    atoms = np.array(atoms)
    center = np.mean(atoms, axis=0)
    dists = np.linalg.norm(atoms - center, axis=1)
    order = np.argsort(dists)
    atoms = atoms[order]
    atoms = atoms[:n]
    atoms = atoms - np.mean(atoms, axis=0)
    return atoms

def _generate_hcp(n):
    """Generate an HCP-like cluster of n atoms."""
    a = 1.0
    c = np.sqrt(8.0/3.0) * a
    # HCP basis in hexagonal coordinates converted to Cartesian
    basis = np.array([
        [0, 0, 0],
        [a, 0, 0],
        [a/2, a*np.sqrt(3)/2, 0],
        [a/2, a*np.sqrt(3)/6, c/2],
        [a, a*np.sqrt(3)/3, c/2],
        [a/2, 2*a*np.sqrt(3)/3, c/2]
    ])
    atoms = []
    max_coord = int(np.ceil((n / 6) ** (1/3)) + 2)
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                for b in basis:
                    # Hexagonal to Cartesian
                    x = ix * a + iy * a/2
                    y = iy * a * np.sqrt(3)/2
                    z = iz * c
                    p = np.array([x, y, z]) + b
                    atoms.append(p)
    atoms = np.array(atoms)
    center = np.mean(atoms, axis=0)
    dists = np.linalg.norm(atoms - center, axis=1)
    order = np.argsort(dists)
    atoms = atoms[order]
    atoms = atoms[:n]
    atoms = atoms - np.mean(atoms, axis=0)
    return atoms

def _generate_shell(n):
    """Generate a shell-like cluster (atoms on a sphere surface)."""
    # Use Fibonacci sphere for uniform distribution
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    i = np.arange(n)
    theta = np.arccos(1 - 2*(i + 0.5)/n)
    psi = 2 * np.pi * i / phi
    r = n ** (1/3) * 0.6
    x = r * np.sin(theta) * np.cos(psi)
    y = r * np.sin(theta) * np.sin(psi)
    z = r * np.cos(theta)
    return np.column_stack([x, y, z])

def _generate_random(n, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = n ** (1/3) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 for q in points):
            points.append(p)
    return np.array(points)

def _basin_hopping(positions, rho, n_steps=50, step_size_start=0.5, step_size_end=0.05, 
                   accept_prob_start=1.0, accept_prob_end=0.1, max_time=None, start_time=None):
    """Multi-scale basin hopping: perturb, relax, accept based on Metropolis criterion."""
    if max_time is not None and start_time is None:
        start_time = time.time()
    
    best_pos = positions.copy()
    best_energy = _morse_energy(positions, rho)
    current_energy = best_energy
    current_pos = positions.copy()
    
    n = len(positions)
    
    for step in range(n_steps):
        if max_time is not None and (time.time() - start_time) > max_time:
            break
        
        # Annealing schedule
        t = step / max(n_steps - 1, 1)
        step_size = step_size_start * (step_size_end / step_size_start) ** t
        accept_prob = accept_prob_start * (accept_prob_end / accept_prob_start) ** t
        
        # Perturb
        rng = np.random.default_rng(step + int(np.sum(positions[:, 0]) * 1000))
        perturbation = rng.normal(0, step_size, size=current_pos.shape)
        new_pos = current_pos + perturbation
        
        # Check validity
        if not _is_valid(new_pos, min_dist=0.5):
            # Try to fix by pushing apart
            new_pos = _fix_min_dist(new_pos)
            if not _is_valid(new_pos, min_dist=0.5):
                continue
        
        # Relax
        energy, relaxed_pos = relax(new_pos, 'morse', rho)
        
        if not _is_valid(relaxed_pos, min_dist=0.5):
            continue
        
        # Accept/reject
        if energy < current_energy:
            accept = True
        else:
            delta_e = energy - current_energy
            # Simple Metropolis with temperature proportional to step_size
            temp = step_size * 0.5
            if temp > 1e-10:
                accept = np.random.random() < np.exp(-delta_e / temp)
            else:
                accept = False
        
        if accept:
            current_pos = relaxed_pos
            current_energy = energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = relaxed_pos.copy()
    
    return best_pos, best_energy

def _fix_min_dist(positions, min_dist=0.5):
    """Try to fix violations of minimum distance by pushing atoms apart."""
    positions = positions.copy()
    n = len(positions)
    for _ in range(10):
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        violated = dist < min_dist
        if not np.any(violated):
            break
        
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] < min_dist:
                    direction = positions[i] - positions[j]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction /= norm
                    else:
                        direction = np.random.randn(3)
                        direction /= np.linalg.norm(direction)
                    push = (min_dist - dist[i, j] + 0.01) / 2
                    positions[i] += direction * push
                    positions[j] -= direction * push
    
    return positions

def construct(instance):
    """Find global minima of Morse clusters using diverse starts and basin hopping."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    potential = instance.get("potential", "morse")
    
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, keep 20s margin
    
    # Generate diverse initial configurations
    generators = [
        ('fcc', _generate_fcc),
        ('bcc', _generate_bcc),
        ('hcp', _generate_hcp),
        ('shell', _generate_shell),
        ('random1', lambda n: _generate_random(n, seed=1)),
        ('random2', lambda n: _generate_random(n, seed=2)),
        ('random3', lambda n: _generate_random(n, seed=3)),
    ]
    
    initial_configs = []
    for name, gen in generators:
        if time.time() > deadline - 30:
            break
        try:
            pos = gen(n)
            # Center
            pos = pos - np.mean(pos, axis=0)
            if _is_valid(pos, min_dist=0.5):
                initial_configs.append((name, pos))
            else:
                pos_fixed = _fix_min_dist(pos)
                if _is_valid(pos_fixed, min_dist=0.5):
                    initial_configs.append((name, pos_fixed))
        except Exception:
            continue
    
    # If no valid configs, use random
    if not initial_configs:
        pos = _generate_random(n, seed=42)
        pos = _fix_min_dist(pos)
        if _is_valid(pos, min_dist=0.5):
            initial_configs.append(('fallback', pos))
        else:
            # Last resort: grid
            pos = np.zeros((n, 3))
            for i in range(n):
                pos[i] = [i % 10, (i // 10) % 10, i // 100]
            pos = _fix_min_dist(pos)
            initial_configs.append(('grid', pos))
    
    # Relax each initial config
    best_energy = np.inf
    best_pos = None
    relaxed_configs = []
    
    for name, pos in initial_configs:
        if time.time() > deadline - 60:
            break
        try:
            energy, relaxed = relax(pos, 'morse', rho)
            if _is_valid(relaxed, min_dist=0.5):
                relaxed_configs.append((name, relaxed, energy))
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed.copy()
        except Exception:
            continue
    
    # Basin hopping on each relaxed config
    n_bh_configs = min(len(relaxed_configs), 6)  # Limit to 6 configs for time
    bh_configs = sorted(relaxed_configs, key=lambda x: x[2])[:n_bh_configs]
    
    # Distribute remaining time among basin hopping runs
    remaining_time = deadline - time.time()
    time_per_bh = max(remaining_time / max(n_bh_configs, 1), 5)
    
    for name, pos, energy in bh_configs:
        if time.time() > deadline - 10:
            break
        
        try:
            bh_pos, bh_energy = _basin_hopping(
                pos, rho, 
                n_steps=100,
                step_size_start=0.8,
                step_size_end=0.03,
                max_time=time_per_bh,
                start_time=time.time()
            )
            
            if _is_valid(bh_pos, min_dist=0.5) and bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except Exception:
            continue
    
    # Final relaxation of best
    if best_pos is not None:
        try:
            final_energy, final_pos = relax(best_pos, 'morse', rho)
            if _is_valid(final_pos, min_dist=0.5) and final_energy < best_energy:
                best_pos = final_pos
        except Exception:
            pass
    
    # Safety: ensure valid output
    if best_pos is None or not _is_valid(best_pos, min_dist=0.5):
        # Fallback to first valid relaxed config
        for name, pos, energy in relaxed_configs:
            if _is_valid(pos, min_dist=0.5):
                best_pos = pos
                break
    
    if best_pos is None:
        # Absolute fallback
        best_pos = np.zeros((n, 3))
        for i in range(n):
            best_pos[i] = [i % 10, (i // 10) % 10, i // 100]
        best_pos = _fix_min_dist(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END
