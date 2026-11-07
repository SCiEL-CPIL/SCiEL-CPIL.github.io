"""Sutton-Chen transition-metal cluster global minima - DE + template + polish strategy."""
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_positions(n, spacing=1.0):
    """Generate positions for a truncated FCC cluster with n atoms."""
    # FCC lattice vectors
    a = spacing
    # Generate FCC points in a cube and select closest to origin until we have n
    max_coord = int(np.ceil((n * 2) ** (1.0/3.0))) + 2
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC lattice: (0,0,0) type and face centers
                # Standard FCC: points at (i,j,k) where i+j+k is even, scaled by a/2
                # Actually let's use the standard construction
                # FCC points: all integer triples (i,j,k) with i+j+k even, times a/2
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a / 2.0
                    points.append(p)
    # Sort by distance from origin
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return points[order[:n]]


def _hcp_positions(n, spacing=1.0):
    """Generate positions for a truncated HCP cluster with n atoms."""
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)  # ideal HCP c/a ratio
    points = []
    max_layer = int(np.ceil((n / 3) ** 0.5)) + 2
    for layer in range(-max_layer, max_layer + 1):
        z = layer * c
        for i in range(-max_layer, max_layer + 1):
            for j in range(-max_layer, max_layer + 1):
                # AB stacking: even layers offset
                if layer % 2 == 0:
                    x = (i + 0.5 * (j % 2)) * a
                    y = j * a * np.sqrt(3.0) / 2.0
                else:
                    x = i * a
                    y = (j + 0.5 * (i % 2)) * a * np.sqrt(3.0) / 2.0
                p = np.array([x, y, z])
                points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return points[order[:n]]


def _spherical_shells(n, spacing=1.0):
    """Generate positions on concentric spherical shells."""
    points = []
    # Try different shell configurations
    shell_counts = [1, 12, 42, 92]  # icosahedral shell sizes
    for sc in shell_counts:
        if len(points) + sc > n:
            # Add partial shell
            remaining = n - len(points)
            if remaining > 0:
                # Distribute remaining atoms on a sphere
                radius = spacing * (len(points) + remaining) ** (1.0/3.0) * 0.7
                for _ in range(remaining):
                    theta = np.random.uniform(0, 2 * np.pi)
                    phi = np.arccos(np.random.uniform(-1, 1))
                    p = radius * np.array([np.sin(phi) * np.cos(theta),
                                           np.sin(phi) * np.sin(theta),
                                           np.cos(phi)])
                    points.append(p)
            break
        if sc == 1:
            points.append(np.zeros(3))
        else:
            radius = spacing * (len(points) + sc) ** (1.0/3.0) * 0.7
            # Fibonacci sphere for even distribution
            golden = (1 + np.sqrt(5)) / 2
            for i in range(sc):
                theta = 2 * np.pi * i / golden
                phi = np.arccos(1 - 2 * (i + 0.5) / sc)
                p = radius * np.array([np.sin(phi) * np.cos(theta),
                                       np.sin(phi) * np.sin(theta),
                                       np.cos(phi)])
                points.append(p)
    return np.array(points)


def _random_compact(n, spacing=1.0, seed=None):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we didn't get enough, fill randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _de_step(positions, kind, n, bounds, pop_size, F, CR, iters, rng):
    """One round of differential evolution on the cluster coordinates."""
    n_atoms = len(positions)
    dim = n_atoms * 3
    # Flatten positions
    x = positions.flatten()
    
    # Create population
    pop = np.zeros((pop_size, dim))
    pop[0] = x
    for i in range(1, pop_size):
        # Perturb from best
        scale = 0.1 + 0.3 * (i / pop_size)
        pop[i] = x + rng.normal(0, scale, dim)
        # Clip to bounds
        pop[i] = np.clip(pop[i], bounds[0], bounds[1])
    
    # Evaluate initial population
    energies = np.zeros(pop_size)
    for i in range(pop_size):
        pos = pop[i].reshape(-1, 3)
        try:
            e, _ = sutton_chen_energy_gradient(pos, kind)
            if np.isfinite(e):
                energies[i] = e
            else:
                energies[i] = 1e10
        except:
            energies[i] = 1e10
    
    best_idx = np.argmin(energies)
    
    for it in range(iters):
        new_pop = pop.copy()
        for i in range(pop_size):
            # Pick three distinct random indices
            a, b, c = rng.choice(pop_size, 3, replace=False)
            # DE mutation
            mutant = pop[a] + F * (pop[b] - pop[c])
            # Crossover
            trial = np.zeros(dim)
            for j in range(dim):
                if rng.random() < CR or j == rng.integers(dim):
                    trial[j] = mutant[j]
                else:
                    trial[j] = pop[i][j]
            trial = np.clip(trial, bounds[0], bounds[1])
            
            # Evaluate trial
            pos = trial.reshape(-1, 3)
            try:
                e, _ = sutton_chen_energy_gradient(pos, kind)
                if np.isfinite(e):
                    trial_e = e
                else:
                    trial_e = 1e10
            except:
                trial_e = 1e10
            
            if trial_e <= energies[i]:
                new_pop[i] = trial
                energies[i] = trial_e
        
        pop = new_pop
        best_idx = np.argmin(energies)
    
    return pop[best_idx].reshape(-1, 3), energies[best_idx]


def construct(instance):
    """DE + template + polish strategy for Sutton-Chen clusters."""
    start_time = time.time()
    deadline = start_time + 290  # 290s deadline for safety
    
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    # Get parameters
    n_exp = instance.get("n_exp", 12)
    m_exp = instance.get("m_exp", 6)
    c = instance.get("c", 144.41)
    
    # Spacing based on potential
    spacing = 0.72  # typical for Sutton-Chen
    
    rng = np.random.default_rng(42)
    
    # Generate diverse templates
    templates = []
    
    # FCC template
    try:
        fcc = _fcc_positions(size, spacing)
        if len(fcc) >= size:
            templates.append(fcc[:size])
    except:
        pass
    
    # HCP template
    try:
        hcp = _hcp_positions(size, spacing)
        if len(hcp) >= size:
            templates.append(hcp[:size])
    except:
        pass
    
    # Spherical shells
    try:
        shells = _spherical_shells(size, spacing)
        if len(shells) >= size:
            templates.append(shells[:size])
    except:
        pass
    
    # Random compact templates
    for seed in [1, 2, 3, 4, 5]:
        try:
            rand = _random_compact(size, spacing, seed)
            templates.append(rand)
        except:
            pass
    
    # Relax each template with L-BFGS-B
    best_energy = np.inf
    best_positions = None
    
    for i, template in enumerate(templates):
        if time.time() > deadline - 30:
            break
        try:
            energy, positions = relax(template, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
    
    # If no valid result, use random
    if best_positions is None:
        best_positions = _random_compact(size, spacing, 42)
        best_energy, _ = sutton_chen_energy_gradient(best_positions, kind)
    
    # Now apply DE to improve
    if time.time() < deadline - 60:
        # Determine bounds
        radius = spacing * size ** (1.0/3.0) * 1.2
        bounds = (-radius, radius)
        
        # DE parameters
        pop_size = min(20, max(8, size // 2))
        F = 0.8
        CR = 0.7
        de_iters = 5
        
        try:
            de_pos, de_energy = _de_step(best_positions, kind, size, bounds, 
                                          pop_size, F, CR, de_iters, rng)
            if np.isfinite(de_energy) and de_energy < best_energy:
                best_energy = de_energy
                best_positions = de_pos
        except:
            pass
    
    # Final polish with tighter L-BFGS-B
    if time.time() < deadline - 10:
        try:
            energy, positions = relax(best_positions, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
    
    # Ensure validity: check minimum distance
    if len(best_positions) > 1:
        from scipy.spatial.distance import pdist
        min_dist = pdist(best_positions, 'euclidean').min()
        if min_dist < 0.35:
            # Nudge apart
            for i in range(len(best_positions)):
                for j in range(i+1, len(best_positions)):
                    d = best_positions[i] - best_positions[j]
                    dist = np.linalg.norm(d)
                    if dist < 0.35:
                        if dist < 1e-10:
                            d = rng.normal(0, 0.1, 3)
                            dist = np.linalg.norm(d)
                        d = d / dist
                        best_positions[i] += 0.18 * d
                        best_positions[j] -= 0.18 * d
    
    return best_positions
