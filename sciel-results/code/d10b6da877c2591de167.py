"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 285.0  # Leave 45s safety margin for 330s limit


def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT


def _get_icosahedral_points():
    """
    Returns the 12 vertices, 20 face centers, and 30 edge midpoints of a regular icosahedron.
    All points are normalized to unit radius.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    norm = np.linalg.norm(v[0])
    v = v / norm
    
    dists = np.linalg.norm(v[:, np.newaxis, :] - v[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists)
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.isclose(dists[i, j], min_dist):
                edges.append((i, j))
    
    face_centers = []
    seen_faces = set()
    for i, j in edges:
        for k in range(12):
            if k == i or k == j:
                continue
            edge_ik = (i, k) in edges or (k, i) in edges
            edge_jk = (j, k) in edges or (k, j) in edges
            if edge_ik and edge_jk:
                face_key = tuple(sorted([i, j, k]))
                if face_key not in seen_faces:
                    seen_faces.add(face_key)
                    center = (v[i] + v[j] + v[k]) / 3.0
                    center = center / np.linalg.norm(center)
                    face_centers.append(center)
    
    edge_centers = []
    for i, j in edges:
        center = (v[i] + v[j]) / 2.0
        center = center / np.linalg.norm(center)
        edge_centers.append(center)
    
    return v, np.array(face_centers), np.array(edge_centers)


_V, _F, _E = _get_icosahedral_points()

# Build a pool of angular directions for shells
# Each direction is a unit vector; we'll scale by shell radius
def _build_direction_pool():
    """Build a diverse pool of unit vectors for shell placement."""
    pool = []
    # Icosahedron vertices (12)
    pool.extend(_V.tolist())
    # Face centers (20)
    pool.extend(_F.tolist())
    # Edge midpoints (30)
    pool.extend(_E.tolist())
    # Additional: golden ratio based directions for more coverage
    phi = (1 + np.sqrt(5)) / 2
    # Dodecahedron vertices (20) - complementary to icosahedron
    d = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ], dtype=float)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    pool.extend(d.tolist())
    # Cuboctahedron directions (24)
    c = np.array([
        [1,1,0],[1,1,0],[-1,1,0],[-1,1,0],
        [1,-1,0],[1,-1,0],[-1,-1,0],[-1,-1,0],
        [1,0,1],[1,0,1],[-1,0,1],[-1,0,1],
        [1,0,-1],[1,0,-1],[-1,0,-1],[-1,0,-1],
        [0,1,1],[0,1,1],[0,-1,1],[0,-1,1],
        [0,1,-1],[0,1,-1],[0,-1,-1],[0,-1,-1]
    ], dtype=float)
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    pool.extend(c.tolist())
    
    pool = np.array(pool)
    # Remove near-duplicates
    unique = [pool[0]]
    for p in pool[1:]:
        if np.min(np.linalg.norm(p - np.array(unique), axis=1)) > 0.05:
            unique.append(p)
    return np.array(unique)

_DIRECTION_POOL = _build_direction_pool()
_N_DIRECTIONS = len(_DIRECTION_POOL)

# Shell radii options (in reduced units)
# The Dzugutov potential has characteristic distances around 1.5-2.0
_SHELL_RADII = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
_N_SHELLS = len(_SHELL_RADII)


def _decode_genome(genome, n_atoms):
    """
    Decode a genome (array of integers) into 3D positions.
    Genome structure:
    - First n_atoms entries: each encodes (shell_index, direction_index)
      Using base encoding: value = shell_idx * _N_DIRECTIONS + dir_idx
    - Atom 0 is always at origin
    """
    positions = np.zeros((n_atoms, 3))
    
    for i in range(1, n_atoms):
        val = genome[i]
        shell_idx = int(val) // _N_DIRECTIONS
        dir_idx = int(val) % _N_DIRECTIONS
        
        if shell_idx >= _N_SHELLS:
            shell_idx = _N_SHELLS - 1
        if dir_idx >= _N_DIRECTIONS:
            dir_idx = 0
            
        radius = _SHELL_RADII[shell_idx]
        direction = _DIRECTION_POOL[dir_idx]
        positions[i] = direction * radius
    
    return positions


def _genome_from_positions(positions, n_atoms):
    """
    Convert continuous positions to a discrete genome by finding nearest
    shell radius and direction.
    """
    genome = np.zeros(n_atoms, dtype=int)
    for i in range(1, n_atoms):
        p = positions[i]
        r = np.linalg.norm(p)
        
        # Find nearest shell
        shell_idx = int(np.argmin([abs(r - sr) for sr in _SHELL_RADII]))
        
        # Find nearest direction
        if r > 1e-6:
            dir_vec = p / r
            dists = np.linalg.norm(_DIRECTION_POOL - dir_vec, axis=1)
            dir_idx = int(np.argmin(dists))
        else:
            dir_idx = 0
        
        genome[i] = shell_idx * _N_DIRECTIONS + dir_idx
    
    return genome


def _evaluate_genome(genome, n_atoms, potential, parameter):
    """
    Evaluate a genome: decode to positions, relax, return energy.
    """
    try:
        positions = _decode_genome(genome, n_atoms)
        
        # Check minimum distance
        if n_atoms > 1:
            dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < 0.5:
                return np.inf, positions
        
        energy, relaxed = relax(positions, potential, parameter)
        if np.isfinite(energy):
            return energy, relaxed
        return np.inf, positions
    except Exception:
        return np.inf, None


def _genetic_algorithm(n_atoms, potential, parameter, pop_size=40, n_generations=30,
                       mutation_rate=0.15, elite_fraction=0.1, rng=None):
    """
    Discrete genetic algorithm over icosahedral shell assignments.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    max_val = _N_SHELLS * _N_DIRECTIONS
    
    # Initialize population
    population = []
    energies = []
    
    # Seed with some structured initial genomes
    # 1. All atoms on different shells with varied directions
    for init_seed in range(5):
        genome = np.zeros(n_atoms, dtype=int)
        for i in range(1, n_atoms):
            # Spread across shells
            shell = (i + init_seed * 3) % _N_SHELLS
            # Spread directions
            direction = (i * 7 + init_seed * 11) % _N_DIRECTIONS
            genome[i] = shell * _N_DIRECTIONS + direction
        population.append(genome)
    
    # Random genomes
    while len(population) < pop_size:
        genome = np.zeros(n_atoms, dtype=int)
        for i in range(1, n_atoms):
            genome[i] = rng.integers(0, max_val)
        population.append(genome)
    
    # Evaluate initial population
    for idx, genome in enumerate(population):
        if not _check_time():
            break
        energy, _ = _evaluate_genome(genome, n_atoms, potential, parameter)
        energies.append(energy)
    
    # Fill in any unevaluated
    while len(energies) < pop_size:
        energies.append(np.inf)
    
    best_energy = min(energies)
    best_genome = population[energies.index(best_energy)].copy()
    best_positions = None
    try:
        best_positions = _decode_genome(best_genome, n_atoms)
        _, best_positions = relax(best_positions, potential, parameter)
    except Exception:
        pass
    
    for gen in range(n_generations):
        if not _check_time():
            break
        
        # Sort by fitness (energy)
        sorted_idx = np.argsort(energies)
        population = [population[i] for i in sorted_idx]
        energies = [energies[i] for i in sorted_idx]
        
        # Track best
        if energies[0] < best_energy:
            best_energy = energies[0]
            best_genome = population[0].copy()
            try:
                best_positions = _decode_genome(best_genome, n_atoms)
                _, best_positions = relax(best_positions, potential, parameter)
            except Exception:
                pass
        
        # Elitism
        n_elite = max(2, int(pop_size * elite_fraction))
        new_population = [pop.copy() for pop in population[:n_elite]]
        new_energies = energies[:n_elite]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if not _check_time():
                break
            
            # Selection: tournament
            def tournament_select():
                candidates = rng.integers(0, pop_size, size=3)
                best_c = min(candidates, key=lambda i: energies[i])
                return population[best_c]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover: uniform
            if rng.random() < 0.8:
                mask = rng.random(n_atoms) > 0.5
                child = np.where(mask, parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            for i in range(1, n_atoms):
                if rng.random() < mutation_rate:
                    # Mutate one gene
                    if rng.random() < 0.5:
                        # Change shell
                        shell = int(child[i]) // _N_DIRECTIONS
                        direction = int(child[i]) % _N_DIRECTIONS
                        new_shell = rng.integers(0, _N_SHELLS)
                        child[i] = new_shell * _N_DIRECTIONS + direction
                    else:
                        # Change direction
                        shell = int(child[i]) // _N_DIRECTIONS
                        new_dir = rng.integers(0, _N_DIRECTIONS)
                        child[i] = shell * _N_DIRECTIONS + new_dir
            
            new_population.append(child)
        
        # Evaluate new population
        for idx in range(n_elite, len(new_population)):
            if not _check_time():
                break
            energy, _ = _evaluate_genome(new_population[idx], n_atoms, potential, parameter)
            new_energies.append(energy)
        
        while len(new_energies) < pop_size:
            new_energies.append(np.inf)
        
        population = new_population
        energies = new_energies
    
    # Final: relax the best genome's positions
    if best_positions is not None:
        try:
            final_energy, final_pos = relax(best_positions, potential, parameter)
            if np.isfinite(final_energy):
                best_energy = final_energy
                best_positions = final_pos
        except Exception:
            pass
    
    return best_energy, best_positions, best_genome


def _kick_and_relax(positions, potential, parameter, rng, kick_size=0.3):
    """
    Apply a random kick and relax. Returns (energy, positions).
    """
    if not _check_time():
        return np.inf, positions
    
    kicked = positions + rng.normal(0, kick_size, size=positions.shape)
    
    # Check minimum distance
    n = len(kicked)
    if n > 1:
        dists = np.linalg.norm(kicked[:, np.newaxis, :] - kicked[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.5:
            return np.inf, positions
    
    try:
        energy, relaxed = relax(kicked, potential, parameter)
        if np.isfinite(energy):
            return energy, relaxed
    except Exception:
        pass
    return np.inf, positions


def _basin_hopping(positions, potential, parameter, n_kicks=50, rng=None):
    """
    Basin hopping: kick + relax, keep best.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    pos = positions.copy()
    try:
        best_energy, _ = relax(pos, potential, parameter)
    except Exception:
        return np.inf, positions
    
    if not np.isfinite(best_energy):
        return np.inf, positions
    
    best_pos = pos.copy()
    
    for _ in range(n_kicks):
        if not _check_time():
            break
        
        kick_size = 0.2 + 0.2 * rng.random()
        e, new_pos = _kick_and_relax(pos, potential, parameter, rng, kick_size)
        
        if e < best_energy:
            best_energy = e
            best_pos = new_pos.copy()
            pos = new_pos.copy()
        else:
            # Accept with Metropolis-like probability at low "temperature"
            if rng.random() < np.exp(-(e - best_energy) / 0.5):
                pos = new_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """
    Find low-energy Dzugutov cluster using discrete genetic algorithm
    over icosahedral shell assignments, combined with basin hopping.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Generate diverse icosahedral seeds and relax
    seeds = [42, 123, 456, 789, 101]
    scales = [0.95, 1.0, 1.05]
    variants = [0, 3, 1]
    
    candidates = []
    for seed in seeds:
        if not _check_time():
            break
        rng = np.random.default_rng(seed)
        
        for scale in scales:
            if not _check_time():
                break
            for variant in variants:
                if not _check_time():
                    break
                try:
                    pos = _generate_icosahedral(size, rng, variant=variant, scale=scale)
                    if len(pos) == size:
                        energy, positions = _relax_with_fallback(pos, potential, parameter)
                        if np.isfinite(energy):
                            candidates.append((energy, positions.copy()))
                except Exception:
                    pass
    
    if not candidates:
        # Fallback
        rng = np.random.default_rng(size)
        best_positions = rng.uniform(-1, 1, (size, 3))
        try:
            _, best_positions = relax(best_positions, potential, parameter)
        except Exception:
            pass
        return best_positions
    
    # Sort by energy
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:min(len(candidates), 6)]
    
    # Phase 2: Genetic algorithm on the discrete shell representation
    # Use the best candidate's positions to seed the GA
    for idx, (init_energy, init_pos) in enumerate(top_candidates[:3]):
        if not _check_time():
            break
        
        # Convert to genome
        genome = _genome_from_positions(init_pos, size)
        
        # Run GA starting from this genome
        rng_ga = np.random.default_rng(5000 + idx)
        try:
            ga_energy, ga_pos, ga_genome = _genetic_algorithm(
                size, potential, parameter,
                pop_size=30, n_generations=25,
                mutation_rate=0.12, rng=rng_ga
            )
        except Exception:
            continue
        
        if ga_energy < best_energy:
            best_energy = ga_energy
            best_positions = ga_pos.copy()
    
    # Phase 3: Basin hopping from best results
    if best_positions is not None and _check_time():
        rng_bh = np.random.default_rng(9000)
        n_kicks = 40 if _check_time() else 20
        try:
            bh_energy, bh_pos = _basin_hopping(
                best_positions, potential, parameter,
                n_kicks=n_kicks, rng=rng_bh
            )
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
        except Exception:
            pass
    
    # Phase 4: Additional basin hopping from other top candidates
    if _check_time():
        for idx, (init_energy, init_pos) in enumerate(top_candidates[3:]):
            if not _check_time():
                break
            rng_bh = np.random.default_rng(10000 + idx)
            try:
                bh_energy, bh_pos = _basin_hopping(
                    init_pos, potential, parameter,
                    n_kicks=20, rng=rng_bh
                )
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_pos
            except Exception:
                pass
    
    # Final fallback
    if best_positions is None:
        rng = np.random.default_rng(size)
        best_positions = rng.uniform(-1, 1, (size, 3))
        try:
            _, best_positions = relax(best_positions, potential, parameter)
        except Exception:
            pass
    
    return best_positions


def _generate_icosahedral(n, rng, variant=0, scale=1.0):
    """
    Generate n points with icosahedral symmetry.
    variant 0: Center + 12 vertices + structured outer (face/edge centers)
    variant 1: Center + 12 vertices + random outer shell
    variant 2: Random compact with minimum spacing
    variant 3: Mackay-like icosahedral shell
    """
    r1 = 1.05 * scale
    
    points = [np.array([0.0, 0.0, 0.0])]
    shell1 = _V * r1
    points.extend(shell1.tolist())
    
    if n <= len(points):
        return np.array(points[:n])
    
    needed = n - len(points)
    
    if variant == 0:
        r2 = r1 * 1.55
        pool = np.vstack([_F, _E])
        idx = rng.permutation(len(pool))
        selected = pool[idx[:needed]] * r2
        for p in selected:
            points.append(p.tolist())
            
    elif variant == 1:
        r2 = r1 * 1.5
        for _ in range(needed):
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            p = direction * r2
            points.append(p.tolist())
            
    elif variant == 3:
        r2 = r1 * 1.6
        pool = np.vstack([_F, _E])
        idx = rng.permutation(len(pool))
        selected = pool[idx[:needed]] * r2
        for p in selected:
            points.append(p.tolist())
    else:
        radius = 1.2 * (n / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)
        pts = []
        attempts = 0
        while len(pts) < n and attempts < 5000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                ok = True
                for q in pts:
                    if np.linalg.norm(p - np.array(q)) < 0.55:
                        ok = False
                        break
                if ok:
                    pts.append(p.tolist())
            attempts += 1
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            pts.append(p.tolist())
        return np.array(pts)
    
    return np.array(points)


def _relax_with_fallback(positions, potential, parameter):
    """Relax with error handling."""
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return energy, relaxed
    except Exception:
        return np.inf, positions
# EVOLVE-BLOCK-END
