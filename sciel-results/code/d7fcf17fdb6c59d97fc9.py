"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _lj_energy(pos):
    """Compute total Lennard-Jones energy for a set of positions."""
    dists = pdist(pos, 'euclidean')
    if np.any(dists < 1e-12):
        return 1e10
    inv_r2 = 1.0 / (dists * dists)
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    return energy


def _relax(pos):
    """Relax a structure using L-BFGS-B."""
    pos = pos.copy()
    
    def energy_and_grad(x):
        pos_reshaped = x.reshape(-1, 3)
        # Compute energy
        dists = pdist(pos_reshaped, 'euclidean')
        if np.any(dists < 1e-12):
            return 1e10, np.zeros_like(x)
        
        inv_r2 = 1.0 / (dists * dists)
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
        energy = 4.0 * np.sum(inv_r12 - inv_r6)
        
        # Compute gradient
        # For each pair i,j: dE/dr = 4*(-12*r^-13 + 6*r^-7) = 24*r^-7 - 48*r^-13
        # dE/dx_i = -dE/dr * (r_vec / r) where r_vec = pos_i - pos_j
        # Actually, let's compute properly
        # E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
        # dE/dx_i = sum_{j != i} 4 * (-12 * r_ij^-14 * (x_i - x_j)_k + 6 * r_ij^-8 * (x_i - x_j)_k)
        #        = sum_{j != i} 4 * (x_i - x_j)_k * (-12 * r_ij^-14 + 6 * r_ij^-8)
        #        = sum_{j != i} (x_i - x_j)_k * (24 * r_ij^-8 - 48 * r_ij^-14)
        
        n = len(pos_reshaped)
        grad = np.zeros_like(x)
        
        # Use pairwise distances and vectors
        diff = pos_reshaped[:, np.newaxis, :] - pos_reshaped[np.newaxis, :, :]
        dists_3d = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists_3d, np.inf)
        
        inv_r2_3d = 1.0 / (dists_3d ** 2)
        inv_r6_3d = inv_r2_3d ** 3
        inv_r8_3d = inv_r2_3d ** 4
        inv_r14_3d = inv_r2_3d ** 7
        
        # dE/dx_i = sum_j (x_i - x_j) * 4 * (-12 * r^-14 + 6 * r^-8)
        #         = sum_j (x_i - x_j) * (-48 * r^-14 + 24 * r^-8)
        coeff = (-48.0 * inv_r14_3d + 24.0 * inv_r8_3d)
        np.fill_diagonal(coeff, 0.0)
        
        grad_3d = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        return energy, grad_3d.flatten()
    
    try:
        result = minimize(
            energy_and_grad,
            pos.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        relaxed = result.x.reshape(-1, 3)
        return relaxed
    except Exception:
        return pos


def _valid(pos):
    """Check if positions are valid (min distance >= 0.5)."""
    dists = pdist(pos, 'euclidean')
    return np.min(dists) >= 0.5


def _make_icosahedral_seed():
    """Create a Mackay icosahedron seed for 55 atoms."""
    # Mackay icosahedron: 1 center + 12 first shell + 42 second shell
    # First shell: 12 vertices of icosahedron
    # Second shell: 30 edges + 12 vertices of inner dodecahedron... 
    # Actually, the second shell has 42 atoms: 30 on edges + 12 on face centers? No.
    # Mackay icosahedron with 2 shells: 1 + 12 + 42 = 55
    # First shell: 12 vertices of regular icosahedron
    # Second shell: 30 atoms on edges (midpoints? no, at distance 2*a from center) 
    #   and 12 atoms on face centers? Let me use known coordinates.
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron (circumradius R1)
    # Standard icosahedron vertices:
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    verts = np.array(verts, dtype=float)
    norm = np.sqrt(np.sum(verts ** 2, axis=1))
    verts = verts / norm[:, np.newaxis]  # unit sphere
    
    # Scale first shell to appropriate radius
    # For LJ, the first shell radius is about 1.12 * something
    # Let's use R1 = 1.12 (approximate nearest neighbor distance from center to first shell)
    # Actually, in Mackay icosahedron, the first shell atoms are at distance ~1.12 from center
    # and from each other. Let me just scale so that first shell atoms are at distance ~1.12 from center
    R1 = 1.12
    first_shell = verts * R1
    
    # Second shell: 42 atoms
    # 30 atoms at edge midpoints of the icosahedron, scaled to R2
    # 12 atoms at face centers, scaled to R2
    # Actually, the second shell in Mackay icosahedron consists of:
    # - 30 atoms on the 30 edges
    # - 12 atoms on the 20 faces? No, icosahedron has 20 faces.
    # Let me look this up differently. 
    # The 42 second-shell atoms are: 30 on edges + 12 on vertices of a smaller icosahedron? 
    # No. Let me just use a simpler approach: generate points on a sphere of radius R2.
    
    # For now, let's just create a reasonable seed and let relaxation fix it.
    # Place 42 atoms on a sphere of radius R2 = 2.24 (approx 2 * R1)
    R2 = 2.24
    
    # Use Fibonacci sphere for 42 points
    def fibonacci_sphere(n):
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2
            radius = np.sqrt(1 - y ** 2)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    second_shell_points = fibonacci_sphere(42)
    second_shell = second_shell_points * R2
    
    # Combine: center + first shell + second shell
    center = np.array([[0.0, 0.0, 0.0]])
    all_points = np.vstack([center, first_shell, second_shell])
    
    # Ensure minimum distance constraint
    # Check and adjust if needed
    return all_points


def construct():
    """Genetic algorithm to find the global minimum of 55-atom LJ cluster."""
    np.random.seed(42)
    
    # Initial population
    pop_size = 20
    population = []
    
    # Add icosahedral seed
    ico_seed = _make_icosahedral_seed()
    if _valid(ico_seed):
        population.append(ico_seed.copy())
    
    # Add random seeds
    while len(population) < pop_size:
        # Random positions in a sphere of radius 5
        pts = np.random.randn(N, 3) * 2.0
        # Ensure min distance
        for _ in range(100):
            if _valid(pts):
                break
            pts = np.random.randn(N, 3) * 2.0
        population.append(pts)
    
    # Relax all initial population members
    relaxed_population = []
    for pos in population:
        relaxed = _relax(pos)
        if _valid(relaxed):
            relaxed_population.append(relaxed)
        else:
            relaxed_population.append(pos)
    
    # Evaluate fitness
    def fitness(pos):
        return _lj_energy(pos)
    
    best_energy = min(fitness(p) for p in relaxed_population)
    best_pos = relaxed_population[np.argmin([fitness(p) for p in relaxed_population])]
    
    # GA loop
    max_generations = 30
    for gen in range(max_generations):
        # Evaluate current population
        energies = [fitness(p) for p in relaxed_population]
        
        # Select parents (tournament selection)
        def tournament_select(pop, energies, k=3):
            idxs = np.random.choice(len(pop), k, replace=False)
            best_idx = idxs[np.argmin([energies[i] for i in idxs])]
            return pop[best_idx].copy()
        
        new_population = []
        
        # Elitism: keep best
        best_idx = np.argmin(energies)
        new_population.append(relaxed_population[best_idx].copy())
        
        # Generate offspring
        while len(new_population) < pop_size:
            parent1 = tournament_select(relaxed_population, energies)
            parent2 = tournament_select(relaxed_population, energies)
            
            # Crossover: exchange subsets of atoms
            child = parent1.copy()
            n_exchange = np.random.randint(10, 30)
            idx1 = np.random.choice(N, n_exchange, replace=False)
            idx2 = np.random.choice(N, n_exchange, replace=False)
            child[idx1] = parent2[idx2]
            
            # Mutation: small Gaussian displacements
            mutation_scale = 0.5
            child += np.random.randn(N, 3) * mutation_scale
            
            # Relax
            relaxed_child = _relax(child)
            if _valid(relaxed_child):
                new_population.append(relaxed_child)
            else:
                new_population.append(child)
        
        relaxed_population = new_population
        
        # Update best
        gen_energies = [fitness(p) for p in relaxed_population]
        gen_best_idx = np.argmin(gen_energies)
        if gen_energies[gen_best_idx] < best_energy:
            best_energy = gen_energies[gen_best_idx]
            best_pos = relaxed_population[gen_best_idx].copy()
    
    # Final relaxation of the best
    best_pos = _relax(best_pos)
    if not _valid(best_pos):
        # Fallback to icosahedral seed
        best_pos = ico_seed
    
    return best_pos
# EVOLVE-BLOCK-END
