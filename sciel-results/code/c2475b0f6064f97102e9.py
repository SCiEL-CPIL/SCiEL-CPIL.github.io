"""Lennard-Jones cluster global minimum, 55 atoms - Genetic Algorithm approach."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def lj_energy(positions):
    """Compute total Lennard-Jones energy for a set of positions."""
    n = len(positions)
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    # Avoid division by zero
    np.fill_diagonal(dist_sq, 1.0)
    inv_r6 = 1.0 / dist_sq ** 3
    inv_r12 = inv_r6 ** 2
    # Sum over all pairs (upper triangle to avoid double counting)
    iu = np.triu_indices(n, k=1)
    energy = 4.0 * np.sum(inv_r12[iu] - inv_r6[iu])
    return energy


def lj_force(positions):
    """Compute forces (negative gradient of energy) for L-BFGS."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    # Force on atom i from atom j: 4 * (12/r^13 - 6/r^7) * r_vec
    # = 4 * (12*inv_r12 - 6*inv_r6) / r^2 * r_vec
    # Actually: F_ij = 4*(12/r^13 - 6/r^7) * r_ij_vec
    # In terms of inv_r6: F = 4*(12*inv_r12 - 6*inv_r6) * r_vec / r^2
    # = 4*(12*inv_r12 - 6*inv_r6) * inv_r2 * r_vec
    factor = 4.0 * (12.0 * inv_r12 - 6.0 * inv_r6) * inv_r2
    np.fill_diagonal(factor, 0.0)
    # Force on atom i is sum over j of factor[i,j] * (pos[j] - pos[i])
    # = sum_j factor[i,j] * (-diff[i,j])
    forces = -np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    return forces


def relax(positions):
    """Relax a structure using L-BFGS."""
    def objective(x):
        pos = x.reshape(-1, 3)
        return lj_energy(pos)

    def gradient(x):
        pos = x.reshape(-1, 3)
        return -lj_force(pos)

    result = minimize(
        objective,
        positions.flatten(),
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'gtol': 1e-8}
    )
    return result.x.reshape(-1, 3)


def mackay_icosahedron():
    """Generate a Mackay icosahedral structure for N=55 atoms."""
    # Mackay icosahedron: 1 center + 12 first shell + 42 second shell
    # First shell: vertices of icosahedron at distance r1
    # Second shell: 42 atoms on second shell

    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Icosahedron vertices (unit sphere)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    # For LJ, optimal first shell distance ~1.12
    r1 = 1.12
    first_shell = verts * r1

    # Second shell: 42 atoms
    # In Mackay icosahedron, second shell consists of:
    # - 12 atoms at face centers (distance from center = r1 * sqrt(1 + 1/phi^2) approx)
    # - 30 atoms at edge midpoints
    # Actually for Mackay: second shell has 42 atoms at specific positions

    # Let me use a simpler approach: place second shell atoms at distance r2
    # with proper icosahedral symmetry

    # For Mackay icosahedron with 55 atoms:
    # Shell 0: 1 atom at origin
    # Shell 1: 12 atoms (icosahedron vertices)
    # Shell 2: 42 atoms

    # Second shell positions in Mackay icosahedron:
    # The 42 atoms are at positions corresponding to:
    # - 12 face centers of the icosahedron
    # - 30 edge midpoints

    # Face centers of icosahedron (normalized)
    # Each face is a triangle of 3 vertices
    # Icosahedron faces: need to enumerate them

    # Alternative: use known coordinates for 55-atom Mackay icosahedron
    # Let me generate them properly

    # Icosahedron edges (pairs of connected vertices)
    # Two vertices are connected if their dot product is 1/phi
    dot_threshold = 1.0 / phi
    edges = []
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.dot(verts[i], verts[j])
            if abs(d - dot_threshold) < 1e-10:
                edges.append((i, j))

    # Find faces (triangles of mutually connected vertices)
    edge_set = set()
    for i, j in edges:
        edge_set.add((min(i, j), max(i, j)))

    for i in range(12):
        for j in range(i + 1, 12):
            if (i, j) in edge_set:
                for k in range(j + 1, 12):
                    if (j, k) in edge_set and (i, k) in edge_set:
                        faces.append((i, j, k))

    # Face centers
    face_centers = []
    for f in faces:
        fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)

    # Edge midpoints
    edge_midpoints = []
    for i, j in edges:
        em = (verts[i] + verts[j]) / 2.0
        em = em / np.linalg.norm(em)
        edge_midpoints.append(em)

    # Second shell distance
    # For Mackay icosahedron, second shell is at distance r2
    # r2/r1 = sqrt(1 + 1/phi^2) for face centers? Let me compute properly.
    # Actually, the ratio depends on the specific geometry.
    # For LJ, we'll relax anyway, so let's use a reasonable starting distance.
    r2 = r1 * 1.6  # Rough estimate, will be relaxed

    second_shell_atoms = []
    for fc in face_centers:
        second_shell_atoms.append(fc * r2)
    for em in edge_midpoints:
        second_shell_atoms.append(em * r2)

    all_atoms = np.array([[0, 0, 0]])
    all_atoms = np.vstack([all_atoms, first_shell])
    all_atoms = np.vstack([all_atoms, np.array(second_shell_atoms)])

    return all_atoms


def random_structure():
    """Generate a random valid structure."""
    # Start with a rough icosahedral-like structure with random perturbations
    base = mackay_icosahedron()
    noise = np.random.normal(0, 0.3, base.shape)
    return base + noise


def crossover(parent1, parent2):
    """Crossover: exchange subsets of atoms between parents."""
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Randomly select a subset of atoms to swap
    n_swap = np.random.randint(5, 20)
    indices = np.random.choice(N, n_swap, replace=False)

    child1[indices] = parent2[indices]
    child2[indices] = parent1[indices]

    return child1, child2


def mutate(positions, sigma=0.2):
    """Apply Gaussian mutation."""
    return positions + np.random.normal(0, sigma, positions.shape)


def tournament_selection(population, fitnesses, k=3):
    """Tournament selection."""
    indices = np.random.choice(len(population), k, replace=False)
    best_idx = indices[np.argmin(fitnesses[indices])]  # Lower energy is better
    return population[best_idx].copy()


def genetic_algorithm(max_generations=50, pop_size=20, time_limit=110.0):
    """Run genetic algorithm to find low-energy LJ cluster."""
    import time
    start_time = time.time()

    # Initialize population
    population = []
    # Start with Mackay icosahedron and variations
    base = mackay_icosahedron()
    population.append(base.copy())
    population.append(mutate(base, 0.1))
    population.append(mutate(base, 0.2))
    population.append(mutate(base, 0.3))

    # Add random structures
    while len(population) < pop_size:
        pop = random_structure()
        population.append(pop)

    # Relax all initial structures
    relaxed_population = []
    for pos in population:
        try:
            relaxed = relax(pos)
            relaxed_population.append(relaxed)
        except:
            relaxed_population.append(pos)

    population = relaxed_population
    fitnesses = np.array([lj_energy(p) for p in population])

    best_energy = np.min(fitnesses)
    best_pos = population[np.argmin(fitnesses)].copy()

    for gen in range(max_generations):
        if time.time() - start_time > time_limit:
            break

        new_population = []

        # Elitism: keep best few
        sort_idx = np.argsort(fitnesses)
        for i in range(min(3, len(population))):
            new_population.append(population[sort_idx[i]].copy())

        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() - start_time > time_limit:
                break

            # Select parents via tournament
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)

            # Crossover
            c1, c2 = crossover(p1, p2)

            # Mutation
            c1 = mutate(c1, sigma=0.15)
            c2 = mutate(c2, sigma=0.15)

            # Relax
            try:
                rc1 = relax(c1)
                new_population.append(rc1)
            except:
                new_population.append(c1)

            if len(new_population) < pop_size:
                try:
                    rc2 = relax(c2)
                    new_population.append(rc2)
                except:
                    new_population.append(c2)

        population = new_population[:pop_size]
        fitnesses = np.array([lj_energy(p) for p in population])

        gen_best = np.min(fitnesses)
        if gen_best < best_energy:
            best_energy = gen_best
            best_pos = population[np.argmin(fitnesses)].copy()

    return best_pos


def construct():
    """Construct initial positions for 55-atom LJ cluster using genetic algorithm."""
    import time
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin

    # Try genetic algorithm
    try:
        result = genetic_algorithm(
            max_generations=40,
            pop_size=15,
            time_limit=time_budget
        )
    except Exception:
        # Fallback to Mackay icosahedron
        result = mackay_icosahedron()

    # Final relaxation
    try:
        result = relax(result)
    except:
        pass

    return result
# EVOLVE-BLOCK-END
