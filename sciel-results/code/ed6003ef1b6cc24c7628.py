"""DE + L-BFGS-B refinement for SC cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _time_left(start_time, deadline=295.0):
    return deadline - (time.time() - start_time)


def _fcc_shells(n, spacing=1.0):
    """Generate n points from FCC lattice shells, centered."""
    if n == 0:
        return np.zeros((0, 3))
    
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 2.5)) + 2
    radius_sq = (n ** (1/3) * spacing * 1.3) ** 2
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                for bx, by, bz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    x = (ix + bx) * spacing
                    y = (iy + by) * spacing
                    z = (iz + bz) * spacing
                    if x*x + y*y + z*z <= radius_sq:
                        points.append([x, y, z])
    
    points = np.array(points)
    if len(points) > n:
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-n**(1/3)*spacing, n**(1/3)*spacing, 3)
            points = np.vstack([points, [p]])
    
    points -= np.mean(points, axis=0)
    return points


def _hcp_shells(n, spacing=1.0):
    """Generate n points from HCP lattice shells."""
    if n == 0:
        return np.zeros((0, 3))
    
    points = []
    a = spacing
    c = np.sqrt(8/3) * a
    max_coord = int(np.ceil(n ** (1/3) * 2.5)) + 2
    radius_sq = (n ** (1/3) * spacing * 1.3) ** 2
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                if iz % 2 == 0:
                    x = ix * a + 0.25 * iy * a
                    y = iy * a * np.sqrt(3)/2
                else:
                    x = ix * a + 0.5*a + 0.25 * iy * a
                    y = iy * a * np.sqrt(3)/2
                z = iz * c / 2
                if x*x + y*y + z*z <= radius_sq:
                    points.append([x, y, z])
    
    points = np.array(points)
    if len(points) > n:
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-n**(1/3)*spacing, n**(1/3)*spacing, 3)
            points = np.vstack([points, [p]])
    
    points -= np.mean(points, axis=0)
    return points


def _icosahedron_based(n, spacing=1.0):
    """Generate points based on icosahedral symmetry with nested shells."""
    phi = (1 + 5 ** 0.5) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    vertices = vertices / np.linalg.norm(vertices[0]) * spacing
    
    points = [[0.0, 0.0, 0.0]]
    points.extend([v.tolist() for v in vertices])
    
    if n > 13:
        dodec = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
            [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
            [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
        ])
        dodec = dodec / np.linalg.norm(dodec[0]) * spacing * 1.8
        for v in dodec:
            if len(points) >= n:
                break
            points.append(v.tolist())
    
    rng = np.random.default_rng(42)
    while len(points) < n:
        base = points[rng.integers(0, len(points))]
        p = np.array(base) + rng.normal(0, 0.3 * spacing, 3)
        points.append(p.tolist())
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _random_compact(n, spacing=1.0, seed=None):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.95
    min_dist = 0.55 * spacing
    
    points = []
    attempts = 0
    max_attempts = n * 300
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - np.array(q)) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points


def _check_validity(positions, min_dist=0.35):
    """Check if positions are valid (finite and no atoms too close)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    # Vectorized distance check
    # Compute pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    # Check off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    if np.any(dists[mask] < min_dist):
        return False
    return True


def _sc_energy_and_grad(x_flat, kind):
    """Energy and gradient for scipy minimize from flat array."""
    pos = x_flat.reshape(-1, 3)
    e, g = sutton_chen_energy_gradient(pos, kind)
    return e, g.flatten()


def _refine(positions, kind, start_time, deadline=295.0, maxiter=300):
    """Refine a structure using L-BFGS-B with the SC gradient."""
    if _time_left(start_time, deadline) < 3.0:
        e, _ = sutton_chen_energy_gradient(positions, kind)
        return e, positions
    try:
        res = minimize(_sc_energy_and_grad, positions.flatten(), jac=True,
                       method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': 1e-7,
                                                     'maxfun': maxiter * 10})
        if np.isfinite(res.fun):
            return res.fun, res.x.reshape(-1, 3)
        return sutton_chen_energy_gradient(positions, kind)
    except Exception:
        try:
            return relax(positions, 'sutton_chen', kind)
        except Exception:
            return sutton_chen_energy_gradient(positions, kind)


def _make_de_candidates(pop, rng, k=1, scale=0.3):
    """Generate k differential evolution offspring from population pop."""
    n = len(pop)
    offspring = []
    for _ in range(k):
        if n < 3:
            # Fallback: random perturbation of a random member
            idx = rng.integers(0, n)
            child = pop[idx] + rng.normal(0, scale, pop[idx].shape)
            offspring.append(child)
        else:
            idx_a = rng.integers(0, n)
            while True:
                idx_b = rng.integers(0, n)
                if idx_b != idx_a:
                    break
            while True:
                idx_c = rng.integers(0, n)
                if idx_c != idx_a and idx_c != idx_b:
                    break
            F = 0.5 + 0.5 * rng.random()  # F in [0.5, 1.0]
            child = pop[idx_a] + F * (pop[idx_b] - pop[idx_c])
            offspring.append(child)
    return offspring


def construct(instance):
    """
    DE + multi-start L-BFGS-B for SC cluster global minima.
    Uses diverse structural templates, then differential evolution with
    local minimization to explore the energy landscape.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    start_time = time.time()
    deadline = 295.0
    
    # Generate diverse starting structures (Reduced count for speed)
    starters = [
        _fcc_shells(size, spacing=1.0),
        _hcp_shells(size, spacing=1.0),
        _icosahedron_based(size, spacing=1.0),
        _random_compact(size, spacing=1.0, seed=size),
    ]
    
    # Phase 1: Relax all starters
    population = []
    for i, starter in enumerate(starters):
        if _time_left(start_time, deadline) < 15.0:
            break
        try:
            energy, positions = _refine(starter, kind, start_time, deadline, maxiter=80)
            if np.isfinite(energy):
                population.append((energy, positions.copy()))
        except Exception:
            continue
    
    if not population:
        # Emergency fallback
        best_positions = _random_compact(size, spacing=1.0, seed=size)
        return best_positions
    
    # Sort by energy
    population.sort(key=lambda x: x[0])
    best_energy, best_positions = population[0]
    
    # Keep top 3 for DE population
    pop_size = min(3, len(population))
    de_pop = [pos for _, pos in population[:pop_size]]
    
    # Phase 2: Differential Evolution with local minimization
    rng = np.random.default_rng(size + 100)
    de_generations = 0
    max_generations = 50
    stall_count = 0
    stall_limit = 10
    
    # Target ~20s for DE phase
    while _time_left(start_time, deadline) > 20.0 and de_generations < max_generations:
        de_generations += 1
        
        # Generate offspring via DE (1 offspring per generation to save time)
        candidates = _make_de_candidates(de_pop, rng, k=1, scale=0.35)
        
        # Relax each candidate
        new_solutions = []
        for cand in candidates:
            if _time_left(start_time, deadline) < 20.0:
                break
            try:
                energy, positions = _refine(cand, kind, start_time, deadline, maxiter=50)
                if np.isfinite(energy):
                    new_solutions.append((energy, positions.copy()))
            except Exception:
                continue
        
        if not new_solutions:
            break
        
        # Combine with existing population and keep best
        all_solutions = population + new_solutions
        all_solutions.sort(key=lambda x: x[0])
        
        new_best = all_solutions[0][0]
        if new_best < best_energy - 1e-6:
            best_energy = new_best
            best_positions = all_solutions[0][1].copy()
            stall_count = 0
        else:
            stall_count += 1
        
        # Update DE population to top pop_size
        de_pop = [pos for _, pos in all_solutions[:pop_size]]
        population = all_solutions[:pop_size]
        
        # Reset if stalled too long
        if stall_count >= stall_limit:
            # Inject new random diversity
            if _time_left(start_time, deadline) > 25.0:
                random_start = _random_compact(size, spacing=1.0, seed=rng.integers(0, 10**9))
                try:
                    energy, positions = _refine(random_start, kind, start_time, deadline, maxiter=50)
                    if np.isfinite(energy):
                        population.append((energy, positions.copy()))
                        de_pop.append(positions.copy())
                except Exception:
                    pass
            population.sort(key=lambda x: x[0])
            de_pop = [pos for _, pos in population[:pop_size]]
            population = population[:pop_size]
            stall_count = 0
    
    # Phase 3: Final fine-tuning
    if _time_left(start_time, deadline) > 5.0:
        try:
            final_energy, final_pos = _refine(best_positions, kind, start_time, deadline, maxiter=100)
            if np.isfinite(final_energy) and final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos.copy()
        except Exception:
            pass
    
    # Final validity check
    if not _check_validity(best_positions):
        rng_fix = np.random.default_rng(size + 500)
        for _ in range(20):
            fixed = best_positions + rng_fix.normal(0, 0.1, best_positions.shape)
            if _check_validity(fixed):
                best_positions = fixed
                break
    
    return best_positions
# EVOLVE-BLOCK-END
