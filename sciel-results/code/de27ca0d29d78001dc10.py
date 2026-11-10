"""Dzugutov cluster global minima search using basin hopping with polytetrahedral seeding."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
a_POT = 1.87
b_POT = 1.94
c_POT = 1.10
d_POT = 0.27
m_POT = 16


def dzugutov_energy(positions):
    """Compute Dzugutov potential energy for a cluster."""
    n = len(positions)
    if n < 2:
        return 0.0
    r = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(r, np.inf)
    
    # Only consider pairs i < j
    triu_indices = np.triu_indices(n, k=1)
    r_ij = r[triu_indices]
    
    energy = 0.0
    for rij in r_ij:
        if rij < a_POT:
            energy += A_POT * (rij ** (-m_POT) - B_POT) * np.exp(c_POT / (rij - a_POT))
        elif rij < b_POT:
            energy += B_POT * np.exp(d_POT / (rij - b_POT))
        else:
            energy += 0.0
    return energy


def generate_icosahedron_seeds(n_atoms, seed=None):
    """Generate multiple icosahedral/polytetrahedral seed structures."""
    rng = np.random.default_rng(seed)
    seeds = []
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0], [-1, -phi, 0], [1, -phi, 0], [-1, phi, 0]
    ])
    # Remove duplicates
    ico_vertices = np.unique(ico_vertices.round(6), axis=0)
    # Scale to appropriate distance (~1.1 for Dzugutov)
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices[0]) * 1.1
    
    # Create multiple seed variants
    for variant in range(6):
        if variant == 0:
            # Pure icosahedron-based growth
            points = list(ico_vertices)
        elif variant == 1:
            # Icosahedron + random perturbations
            points = list(ico_vertices + rng.normal(0, 0.05, ico_vertices.shape))
        elif variant == 2:
            # Two icosahedra offset
            ico2 = ico_vertices + np.array([1.5, 0, 0])
            points = list(ico_vertices[:6]) + list(ico2[:6])
        elif variant == 3:
            # Fcc-like packing
            points = []
            for i in range(int(np.sqrt(n_atoms))):
                for j in range(int(np.sqrt(n_atoms))):
                    if len(points) >= n_atoms:
                        break
                    x, y = i * 1.1, j * 1.1
                    z = 0 if (i + j) % 2 == 0 else 0.5
                    points.append([x - n_atoms*0.3, y - n_atoms*0.3, z])
            points = np.array(points)
        elif variant == 4:
            # Random compact with minimum distance
            points = []
            radius = 1.1 * n_atoms ** (1.0/3.0) * 0.8
            attempts = 0
            while len(points) < n_atoms and attempts < 10000:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius:
                    if all(np.linalg.norm(p - q) >= 0.85 for q in points):
                        points.append(p)
                attempts += 1
            points = np.array(points) if len(points) >= n_atoms else None
        elif variant == 5:
            # HCP-like layered structure
            points = []
            layer_size = int(np.sqrt(n_atoms))
            for layer in range(int(n_atoms / layer_size) + 1):
                offset = 0.5 if layer % 2 else 0
                for i in range(layer_size):
                    if len(points) >= n_atoms:
                        break
                    x, y = i * 1.05 + offset, layer * 0.9
                    points.append([x, y, 0])
            points = np.array(points)
        
        if points is not None and len(points) >= n_atoms:
            points = np.array(points[:n_atoms])
            # Center the cluster
            points -= points.mean(axis=0)
            seeds.append(points)
    
    return [s for s in seeds if s is not None and len(s) == n_atoms]


def basin_hopping_step(positions, rng, step_size=0.1):
    """Perform one basin hopping step: random displacement + local relaxation."""
    n = len(positions)
    # Random displacement with anisotropic scaling
    displacement = rng.normal(0, step_size, positions.shape)
    new_positions = positions + displacement
    
    # Ensure minimum separation
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(new_positions[i] - new_positions[j])
            if d < 0.5:
                # Push apart
                if d < 1e-10:
                    direction = rng.normal(0, 1, 3)
                else:
                    direction = (new_positions[i] - new_positions[j]) / d
                push = (0.5 - d) / 2
                new_positions[i] += push * direction
                new_positions[j] -= push * direction
    
    # Local relaxation
    try:
        energy, relaxed = relax(new_positions, 'dzugutov')
        return energy, relaxed
    except Exception:
        return np.inf, new_positions


def construct(instance):
    """Find low-energy Dzugutov cluster structures using basin hopping from polytetrahedral seeds."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    time_budget = 45.0  # Per-instance time budget (6 instances * 45s = 270s < 300s)
    
    # Generate diverse seeds
    seeds = generate_icosahedron_seeds(size, seed=size)
    
    if not seeds:
        # Fallback: simple random compact
        rng = np.random.default_rng(size)
        radius = 1.1 * size ** (1.0/3.0) * 0.8
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.85 for q in points):
                points.append(p)
        seeds = [np.array(points)]
    
    # Relax all seeds initially
    best_energy = np.inf
    best_positions = seeds[0]
    rng = np.random.default_rng(size * 1000)
    
    for seed_pos in seeds:
        try:
            energy, relaxed = relax(seed_pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
        except Exception:
            pass
    
    # Basin hopping from the best seed
    current_energy = best_energy
    current_positions = best_positions.copy()
    
    # Adaptive step size
    step_size = 0.15
    accept_prob = 1.0
    temperature = 0.5  # Initial "temperature" for Metropolis acceptance
    
    max_iterations = 50
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() - start_time > time_budget:
            break
        
        # Random displacement
        new_positions = current_positions + rng.normal(0, step_size, current_positions.shape)
        
        # Enforce minimum distance
        n = len(new_positions)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(new_positions[i] - new_positions[j])
                if d < 0.5:
                    if d < 1e-10:
                        direction = rng.normal(0, 1, 3)
                    else:
                        direction = (new_positions[i] - new_positions[j]) / d
                    push = (0.5 - d) / 2
                    new_positions[i] += push * direction
                    new_positions[j] -= push * direction
        
        # Local relaxation
        try:
            new_energy, relaxed = relax(new_positions, potential, parameter)
        except Exception:
            new_energy = np.inf
            relaxed = new_positions
        
        # Metropolis acceptance
        if new_energy < current_energy:
            accept = True
        else:
            delta_e = new_energy - current_energy
            accept = rng.random() < np.exp(-delta_e / temperature)
        
        if accept:
            current_energy = new_energy
            current_positions = relaxed.copy()
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = relaxed.copy()
                # Reduce temperature on improvement
                temperature *= 0.85
                # Reduce step size slightly
                step_size *= 0.95
        else:
            # Increase step size occasionally to escape local minima
            step_size *= 1.1
            temperature *= 1.05
        
        # Clamp step size
        step_size = np.clip(step_size, 0.05, 0.5)
        temperature = np.clip(temperature, 0.1, 2.0)
        
        iteration += 1
    
    # Final relaxation to ensure we're at a local minimum
    try:
        final_energy, final_positions = relax(best_positions, potential, parameter)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_positions
    except Exception:
        pass
    
    return best_positions
# EVOLVE-BLOCK-END
