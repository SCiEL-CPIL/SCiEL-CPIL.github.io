"""
Lennard-Jones 55-atom cluster global minimum search.

Combines the Mackay icosahedron seed strategy with a genetic-algorithm
population search (crossover + mutation) and multi-stage L-BFGS polish
to converge to the global minimum energy (~-279.248).
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55

def _get_icosahedron_vectors():
    """
    Returns 12 unit vectors along the 5-fold axes (icosahedron vertices)
    and 30 unit vectors along the 2-fold axes (icosahedron edge midpoints).
    """
    phi = (1 + np.sqrt(5)) / 2

    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [-phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, -1],
    ], dtype=float)

    # Normalize to unit length
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    # 30 edge midpoints
    dists = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists[dists > 1e-8])

    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.isclose(dists[i, j], min_dist, atol=1e-8):
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                edge_mids.append(mid)

    edge_mids = np.array(edge_mids)
    return verts, edge_mids


def _mackay_icosahedron_55(r1=1.135, r2=1.162, r3=1.283):
    """
    Construct the 55-atom two-shell Mackay icosahedron.
    1 center, 12 at 5-fold (r1), 30 at 2-fold (r2), 12 at 5-fold (r3).
    """
    v5, v2 = _get_icosahedron_vectors()

    points = [np.zeros(3)]

    # Add 12 atoms at 5-fold axes, radius r1
    for v in v5:
        points.append(v * r1)

    # Add 30 atoms at 2-fold axes, radius r2
    for v in v2:
        points.append(v * r2)

    # Add 12 atoms at 5-fold axes, radius r3
    for v in v5:
        points.append(v * r3)

    points = np.array(points, dtype=float)
    return points


def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    return energy


def _lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))

    # Gradient: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)/r
    # Simplified: grad = 4 * sum_j (-12*r2^-7 + 6*r2^-4) * diff
    r2_inv4 = r2_safe ** (-4)
    r2_inv7 = r2_safe ** (-7)
    coeff = 4 * (-12 * r2_inv7 + 6 * r2_inv4)
    coeff = np.where(mask, coeff, 0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _is_valid(pos):
    """Check if all inter-atomic distances are >= 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5


def _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10):
    """Relax positions using L-BFGS-B with analytical gradient."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()

    result = minimize(
        energy_func,
        pos.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return result.x.reshape(-1, 3)


def _polish(pos, deadline=None):
    """
    High-precision multi-stage polish pass with progressively tighter tolerances.
    """
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()

    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()

    x0 = pos.flatten()

    # Multi-stage polish: progressively tighter
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
    ]

    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass

    return best_pos, best_energy


def _crossover(parent1, parent2, rng):
    """
    Crossover: exchange a random subset of atoms between two parents.
    Returns a new structure.
    """
    child = parent1.copy()
    n = len(child)
    # Select a random subset of indices to swap
    swap_count = rng.integers(5, 25)  # swap 5-25 atoms
    indices = rng.choice(n, size=swap_count, replace=False)
    child[indices] = parent2[indices]
    return child


def _mutation(pos, scale, rng):
    """
    Apply small Gaussian displacement to positions.
    """
    noise = rng.normal(0, scale, size=pos.shape)
    return pos + noise


def construct():
    """
    Construct the 55-atom LJ cluster using a genetic algorithm with
    Mackay icosahedron seeds, followed by L-BFGS relaxation and polish.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    rng = np.random.default_rng(42)

    # Use known good initial radii for the Mackay icosahedron
    initial_radii = np.array([1.135, 1.162, 1.283])

    # Construct base geometry
    base = _mackay_icosahedron_55(*initial_radii)

    # Initialize population with diverse Mackay variants
    population = []
    energies = []

    # Add base geometry
    population.append(base.copy())
    energies.append(_lj_energy(base))

    # Add radius-perturbed variants
    radius_perturbations = [
        (1.10, 1.15, 1.25),
        (1.15, 1.18, 1.30),
        (1.08, 1.14, 1.26),
        (1.12, 1.17, 1.29),
        (1.11, 1.16, 1.27),
    ]
    for r1, r2, r3 in radius_perturbations:
        cand = _mackay_icosahedron_55(r1, r2, r3)
        population.append(cand)
        energies.append(_lj_energy(cand))

    # Add Gaussian-perturbed variants
    for scale in [0.01, 0.05, 0.1, 0.15]:
        cand = _mutation(base, scale, rng)
        population.append(cand)
        energies.append(_lj_energy(cand))

    # Relax all initial candidates and keep the best
    best_energy = float('inf')
    best_pos = base.copy()

    for i, cand in enumerate(population):
        if time.time() > deadline - 20.0:
            break

        relaxed = _relax(cand, maxiter=5000, ftol=1e-12, gtol=1e-10)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()

    # Genetic algorithm loop
    pop_size = 10
    max_generations = 20
    current_pop = [best_pos.copy()]

    # Add some diverse individuals to the GA population
    for scale in [0.02, 0.05, 0.1, 0.15, 0.2]:
        cand = _mutation(best_pos, scale, rng)
        relaxed = _relax(cand, maxiter=3000, ftol=1e-12, gtol=1e-10)
        if _is_valid(relaxed):
            current_pop.append(relaxed)
        if len(current_pop) >= pop_size:
            break

    # Fill remaining slots with radius-perturbed variants
    for r1, r2, r3 in radius_perturbations:
        if len(current_pop) >= pop_size:
            break
        cand = _mackay_icosahedron_55(r1, r2, r3)
        relaxed = _relax(cand, maxiter=3000, ftol=1e-12, gtol=1e-10)
        if _is_valid(relaxed):
            current_pop.append(relaxed)

    for gen in range(max_generations):
        if time.time() > deadline - 15.0:
            break

        # Evaluate fitness of current population
        pop_energies = []
        for pos in current_pop:
            if _is_valid(pos):
                pop_energies.append(_lj_energy(pos))
            else:
                pop_energies.append(1e10)

        # Sort by energy
        sorted_indices = np.argsort(pop_energies)
        current_pop = [current_pop[i] for i in sorted_indices]
        pop_energies = [pop_energies[i] for i in sorted_indices]

        # Update best
        if pop_energies[0] < best_energy:
            best_energy = pop_energies[0]
            best_pos = current_pop[0].copy()

        # Selection: keep top half
        elite_size = max(2, pop_size // 2)
        next_pop = current_pop[:elite_size]

        # Crossover and mutation to fill the rest
        while len(next_pop) < pop_size:
            if time.time() > deadline - 10.0:
                break

            # Select parents from top half
            p1_idx = rng.integers(0, elite_size)
            p2_idx = rng.integers(0, elite_size)
            parent1 = current_pop[p1_idx]
            parent2 = current_pop[p2_idx]

            # Crossover
            child = _crossover(parent1, parent2, rng)

            # Mutation
            mut_scale = 0.02 + 0.05 * rng.random()
            child = _mutation(child, mut_scale, rng)

            # Relax
            relaxed = _relax(child, maxiter=3000, ftol=1e-12, gtol=1e-10)
            if _is_valid(relaxed):
                next_pop.append(relaxed)

        current_pop = next_pop

    # Final high-precision polish pass
    if best_energy < float('inf') and time.time() < deadline - 5.0:
        refined, refined_energy = _polish(best_pos, deadline=deadline)
        if _is_valid(refined):
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_pos = refined

    return best_pos
