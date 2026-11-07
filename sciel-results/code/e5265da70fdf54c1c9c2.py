"""Sutton-Chen transition-metal cluster global minima - DE + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_positions(n, spacing=1.0):
    """Generate FCC-like positions for n atoms."""
    # FCC lattice points within a sphere
    a = spacing
    positions = []
    # Generate FCC lattice points
    max_idx = int(np.ceil(n ** 0.5)) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC: atoms at (0,0,0) and face centers
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if len(positions) < n:
                        positions.append(p)
    positions = np.array(positions[:n])
    # Center
    positions -= positions.mean(axis=0)
    return positions


def _hcp_positions(n, spacing=1.0):
    """Generate HCP-like positions for n atoms."""
    positions = []
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    # Generate HCP lattice
    max_idx = int(np.ceil(n ** 0.5)) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # A-layer
                p_a = np.array([i * a, j * a * np.sqrt(3) / 2, k * c])
                # B-layer
                p_b = np.array([i * a + a / 2, j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6, k * c + c / 2])
                if len(positions) < n:
                    positions.append(p_a)
                if len(positions) < n:
                    positions.append(p_b)
    positions = np.array(positions[:n])
    positions -= positions.mean(axis=0)
    return positions


def _icosahedral_positions(n, spacing=1.0):
    """Generate icosahedral shell positions for n atoms."""
    positions = []
    # Icosahedron vertices (12)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi]
    ], dtype=float)
    ico /= np.linalg.norm(ico[0])
    
    # Center
    positions.append(np.zeros(3))
    # First shell (12)
    for v in ico:
        if len(positions) < n:
            positions.append(v * spacing)
    # Second shell (42) - approximate
    if len(positions) < n:
        # Add more points on a sphere
        rng = np.random.default_rng(42)
        while len(positions) < n:
            p = rng.normal(size=3)
            p /= np.linalg.norm(p)
            p *= spacing * 2.0
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= spacing * 0.8 for q in positions):
                positions.append(p)
    positions = np.array(positions[:n])
    positions -= positions.mean(axis=0)
    return positions


def _random_positions(n, spacing=1.0, seed=None):
    """Generate random compact positions."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    positions = []
    attempts = 0
    while len(positions) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= spacing * 0.7 for q in positions):
                positions.append(p)
        attempts += 1
    # If we couldn't place enough, add random points
    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        positions.append(p)
    return np.array(positions[:n])


def _basin_hopping(positions, potential, kind, max_iter=50, seed=None):
    """Simple basin hopping: perturb, relax, keep if lower energy."""
    rng = np.random.default_rng(seed)
    best_pos = positions.copy()
    best_energy, _ = sutton_chen_energy_gradient(best_pos, kind)
    current_pos = positions.copy()
    current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
    
    for i in range(max_iter):
        # Perturb
        step_size = 0.3 * (1.0 - 0.5 * i / max_iter)
        perturbed = current_pos + rng.normal(0, step_size, size=current_pos.shape)
        # Relax
        energy, relaxed = relax(perturbed, potential, kind)
        if energy < current_energy:
            current_pos = relaxed
            current_energy = energy
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    return best_energy, best_pos


def construct(instance):
    """DE-based global optimization with diverse initialization and basin hopping."""
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, keep 20s margin
    
    # Diverse initial population
    templates = [
        _fcc_positions(n),
        _hcp_positions(n),
        _icosahedral_positions(n),
        _random_positions(n, seed=1),
        _random_positions(n, seed=2),
        _random_positions(n, seed=3),
        _random_positions(n, seed=4),
        _random_positions(n, seed=5),
    ]
    
    # Relax each template
    population = []
    for tmpl in templates:
        if time.time() > deadline - 60:
            break
        energy, relaxed = relax(tmpl, potential, kind)
        population.append((energy, relaxed))
    
    # Sort by energy
    population.sort(key=lambda x: x[0])
    
    best_energy, best_pos = population[0]
    
    # Differential Evolution
    pop_size = min(16, len(population))
    if pop_size < 4:
        # Pad population if needed
        while len(population) < 4:
            e, p = population[-1]
            perturbed = p + np.random.default_rng(len(population)).normal(0, 0.3, size=p.shape)
            e2, p2 = relax(perturbed, potential, kind)
            population.append((e2, p2))
        population.sort(key=lambda x: x[0])
    
    # DE parameters
    F = 0.8  # scale factor
    CR = 0.9  # crossover rate
    max_de_iter = 30
    
    for gen in range(max_de_iter):
        if time.time() > deadline - 30:
            break
        
        new_population = [p for p in population[:pop_size]]
        
        for i in range(pop_size):
            if time.time() > deadline - 30:
                break
            
            # Select three distinct individuals
            a, b, c = np.random.choice(pop_size, 3, replace=False)
            while a == b or b == c or a == c:
                a, b, c = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            mutant = population[a][1] + F * (population[b][1] - population[c][1])
            
            # Crossover
            trial = np.zeros_like(mutant)
            j_rand = np.random.randint(3)
            for j in range(3):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = population[i][1][j]
            
            # Relax
            energy, relaxed = relax(trial, potential, kind)
            
            # Selection
            if energy < population[i][0]:
                new_population[i] = (energy, relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed.copy()
        
        population = new_population
        population.sort(key=lambda x: x[0])
    
    # Final basin hopping on best few
    if time.time() < deadline - 20:
        for i, (e, p) in enumerate(population[:3]):
            if time.time() > deadline - 10:
                break
            e_bh, p_bh = _basin_hopping(p, potential, kind, max_iter=20, seed=i)
            if e_bh < best_energy:
                best_energy = e_bh
                best_pos = p_bh.copy()
    
    return best_pos
# EVOLVE-BLOCK-END
