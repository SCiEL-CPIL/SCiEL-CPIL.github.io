"""Dzugutov cluster global minima search.
Combines high-quality polytetrahedral (Mackay icosahedron) seeding with parallel
multi-basin exploration using multiprocessing and the robust relax API."""

import numpy as np
import time
import multiprocessing as mp
from scipy.optimize import minimize
import autoresearch_suite.chem_clusters as chem_clusters
from autoresearch_suite.chem_clusters import relax

# Global deadline guard
TOTAL_TIME_LIMIT = 310.0
_start_time = None
_best_results = {}

# Dzugutov Potential Parameters
A_POT = 5.82
B_POT = 1.28
A_CUT = 1.87
B_CUT = 1.94
C_EXP = 1.10
D_EXP = 0.27
M_EXP = 16


def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time


def _remaining():
    return max(0.0, TOTAL_TIME_LIMIT - _elapsed())


def _dzugutov_energy_gradient(positions):
    """Compute Dzugutov energy and gradient."""
    try:
        energy, grad = chem_clusters.dzugutov_energy_gradient(positions)
        return energy, grad
    except Exception:
        pass

    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)

    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r < 1e-8:
                continue

            if r < A_CUT:
                V = (A_POT * (r ** (-M_EXP) - B_POT)) * np.exp(C_EXP / (r - A_CUT))
                dVdr = A_POT * (-M_EXP * r ** (-M_EXP - 1) - C_EXP / (r - A_CUT) ** 2 * (r ** (-M_EXP) - B_POT)) * np.exp(C_EXP / (r - A_CUT))
            elif r < B_CUT:
                V = B_POT * np.exp(D_EXP / (r - B_CUT))
                dVdr = B_POT * np.exp(D_EXP / (r - B_CUT)) * (-D_EXP / (r - B_CUT) ** 2)
            else:
                continue

            energy += V
            force_factor = dVdr / r
            grad[i] -= force_factor * r_vec
            grad[j] += force_factor * r_vec

    return energy, grad


def _make_mackay_icosahedron(n, rng):
    """
    Generate a Mackay icosahedron-like structure.
    Shell 0: 1 center atom
    Shell 1: 12 vertices (radius ~ 1.0)
    Shell 2: 30 edge midpoints (radius ~ 1.701)
    Shell 3: 20 face centers (radius ~ 2.201)
    Total: 1 + 12 + 30 + 20 = 63 atoms for full Mackay
    """
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)

    shells = [np.zeros((1, 3))]
    remaining = n - 1

    if remaining > 0:
        r1 = 1.0
        shell1 = v * r1
        take1 = min(12, remaining)
        shells.append(shell1[:take1])
        remaining -= take1

    if remaining > 0:
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(v[i] - v[j])
                if d < 1.1:
                    mid = (v[i] + v[j]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    edges.append(mid)

        unique_edges = []
        for e in edges:
            if all(np.linalg.norm(e - ue) > 0.1 for ue in unique_edges):
                unique_edges.append(e)

        r2 = np.sqrt(1 + 2 / phi)
        shell2 = np.array(unique_edges) * r2
        take2 = min(len(shell2), remaining)
        shells.append(shell2[:take2])
        remaining -= take2

    if remaining > 0:
        face_indices = [
            [0, 1, 8], [0, 1, 11], [0, 2, 8], [0, 2, 9],
            [0, 3, 9], [0, 3, 11], [1, 4, 8], [1, 4, 10],
            [1, 5, 10], [1, 5, 11], [2, 4, 8], [2, 4, 9],
            [2, 5, 9], [2, 5, 10], [3, 6, 9], [3, 6, 11],
            [3, 7, 11], [3, 7, 10], [6, 4, 8], [6, 7, 10]
        ]

        face_centers = []
        for f in face_indices:
            fc = (v[f[0]] + v[f[1]] + v[f[2]]) / 3.0
            fc = fc / np.linalg.norm(fc)
            face_centers.append(fc)

        r3 = np.sqrt(1 + 2 / phi + 2 / (phi * phi))
        shell3 = np.array(face_centers) * r3
        take3 = min(len(shell3), remaining)
        shells.append(shell3[:take3])
        remaining -= take3

    if remaining > 0:
        r4 = 2.8
        for _ in range(remaining):
            p = rng.normal(size=3)
            p = p / np.linalg.norm(p) * r4
            shells.append(p.reshape(1, 3))

    positions = np.vstack(shells)

    if len(positions) > n:
        radii = np.linalg.norm(positions, axis=1)
        indices = np.argsort(radii)[:n]
        positions = positions[indices]

    return positions


def _make_fcc_cluster(n, spacing, rng):
    """Generate a compact FCC-like cluster."""
    positions = [np.zeros(3)]
    remaining = n - 1

    a = spacing * 0.9
    lattice_pts = []
    for ix in range(-4, 5):
        for iy in range(-4, 5):
            for iz in range(-4, 5):
                p = np.array([ix, iy, iz]) * a
                lattice_pts.append(p)
                for fc in [(0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pc = np.array([ix + fc[0], iy + fc[1], iz + fc[2]]) * a
                    lattice_pts.append(pc)

    lattice_pts = np.array(lattice_pts)
    dists = np.linalg.norm(lattice_pts, axis=1)
    order = np.argsort(dists)
    lattice_pts = lattice_pts[order]

    nonzero_mask = dists > 0.1
    lattice_pts = lattice_pts[nonzero_mask]

    for i in range(min(remaining, len(lattice_pts))):
        positions.append(lattice_pts[i])

    while len(positions) < n:
        p = rng.uniform(-4 * spacing, 4 * spacing, 3)
        positions.append(p)

    return np.array(positions[:n])


def _make_random_compact(n, spacing, rng):
    """Generate a random compact cluster."""
    positions = []
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    min_dist = 0.6 * spacing
    max_attempts = 5000

    for _ in range(max_attempts):
        if len(positions) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in positions):
                positions.append(p)

    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        positions.append(p)

    return np.array(positions[:n])


def _enforce_min_distance(positions, rng, min_dist=0.5):
    """Separate any atoms closer than min_dist."""
    pos = positions.copy()
    for _ in range(20):
        needs_fix = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    needs_fix = True
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-8:
                        diff = rng.normal(0, 1, 3)
                    diff = diff / np.linalg.norm(diff)
                    push = (min_dist - d) / 2 + 0.01
                    pos[i] += diff * push
                    pos[j] -= diff * push
        if not needs_fix:
            break
    return pos


def _try_relax(positions, potential, parameter):
    """Try using the provided relax API."""
    try:
        energy, pos = relax(positions, potential, parameter)
        return energy, pos
    except Exception:
        return None, None


def _optimize_positions(positions, potential, parameter, time_budget):
    """Optimize positions using L-BFGS-B as fallback."""
    def objective_with_grad(x):
        pos = x.reshape(-1, 3)
        try:
            energy, grad = _dzugutov_energy_gradient(pos)
            return energy, grad.flatten()
        except Exception:
            return 1e10, np.zeros_like(x)

    x0 = positions.flatten()
    best_energy = 1e10
    best_x = x0.copy()

    try:
        result = minimize(
            objective_with_grad, x0, method='L-BFGS-B',
            options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_x = result.x.copy()
    except Exception:
        pass

    return best_x.reshape(-1, 3)


def _worker_relax(args):
    """Worker function for multiprocessing: relax a single candidate."""
    positions, potential, parameter = args
    try:
        energy, pos = relax(positions, potential, parameter)
        return (energy, pos)
    except Exception:
        return (None, None)


def _parallel_relax(candidates, potential, parameter, n_workers=4):
    """Relax multiple candidates in parallel using multiprocessing."""
    if len(candidates) == 0:
        return []

    # Limit workers to available CPUs and number of candidates
    n_workers = min(n_workers, len(candidates), mp.cpu_count())
    if n_workers <= 1:
        # Sequential fallback
        results = []
        for cand in candidates:
            results.append(_worker_relax((cand, potential, parameter)))
        return results

    try:
        pool = mp.Pool(processes=n_workers)
        tasks = [(cand, potential, parameter) for cand in candidates]
        results = pool.map(_worker_relax, tasks)
        pool.close()
        pool.join()
        return results
    except Exception:
        # Fallback to sequential
        results = []
        for cand in candidates:
            results.append(_worker_relax((cand, potential, parameter)))
        return results


def construct(instance):
    """Find low-energy structures of Dzugutov clusters."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))

    if n in _best_results:
        return _best_results[n]

    instances_to_do = [38, 45, 50, 55, 57, 60]
    idx = instances_to_do.index(n) if n in instances_to_do else 0
    base_time_per_instance = TOTAL_TIME_LIMIT / 6.0
    time_budget = min(_remaining() * 0.9, base_time_per_instance * 1.5)
    time_budget = max(time_budget, 10.0)

    rng = np.random.default_rng(42 + n)
    spacing = 1.1

    # Generate diverse initial structures
    candidates = []

    # 1. Mackay icosahedron (primary seed)
    mackay_pos = _make_mackay_icosahedron(n, rng)
    candidates.append(mackay_pos)

    # 2. FCC-like
    fcc_pos = _make_fcc_cluster(n, spacing, rng)
    candidates.append(fcc_pos)

    # 3. Random compact (multiple variants)
    for seed_offset in range(4):
        rng_local = np.random.default_rng(42 + n + seed_offset * 100)
        rand_pos = _make_random_compact(n, spacing, rng_local)
        candidates.append(rand_pos)

    # 4. Perturbed Mackay (multiple scales)
    for scale in [0.90, 0.95, 1.05, 1.10]:
        perturbed = mackay_pos * scale + rng.normal(0, 0.05, mackay_pos.shape)
        candidates.append(perturbed)

    # 5. Scaled FCC variants
    for scale in [0.95, 1.05]:
        fcc_scaled = fcc_pos * scale + rng.normal(0, 0.03, fcc_pos.shape)
        candidates.append(fcc_scaled)

    # Enforce minimum distance on all candidates
    processed_candidates = []
    for cand in candidates:
        cand = _enforce_min_distance(cand, rng)
        processed_candidates.append(cand)

    # Parallel relaxation of all candidates
    relax_results = _parallel_relax(processed_candidates, potential, parameter, n_workers=4)

    best_energy = np.inf
    best_positions = None

    # Collect results from parallel relaxation
    for i, (energy, pos) in enumerate(relax_results):
        if energy is not None and pos is not None:
            if energy < best_energy:
                best_energy = energy
                best_positions = pos

    # If no relax succeeded, try sequential with L-BFGS
    if best_positions is None:
        for i, cand in enumerate(processed_candidates):
            if _remaining() < 3:
                break

            # Try relax again sequentially
            rel_energy, rel_pos = _try_relax(cand, potential, parameter)
            if rel_energy is not None and rel_pos is not None:
                if rel_energy < best_energy:
                    best_energy = rel_energy
                    best_positions = rel_pos
                continue

            # L-BFGS fallback
            if _remaining() > 5:
                opt_time = min(_remaining() * 0.2, 8.0)
                opt_pos = _optimize_positions(cand, potential, parameter, opt_time)
                try:
                    final_energy, _ = _dzugutov_energy_gradient(opt_pos)
                    if final_energy < best_energy:
                        best_energy = final_energy
                        best_positions = opt_pos
                except Exception:
                    pass

    # Second pass: try relax on the best found so far with small perturbations
    if best_positions is not None and _remaining() > 10:
        for pert_scale in [0.02, 0.05, 0.10]:
            if _remaining() < 5:
                break
            perturbed = best_positions + rng.normal(0, pert_scale, best_positions.shape)
            perturbed = _enforce_min_distance(perturbed, rng)
            rel_energy, rel_pos = _try_relax(perturbed, potential, parameter)
            if rel_energy is not None and rel_pos is not None:
                if rel_energy < best_energy:
                    best_energy = rel_energy
                    best_positions = rel_pos

    # Final fallback
    if best_positions is None:
        if len(processed_candidates) > 0:
            best_positions = processed_candidates[0]
        else:
            best_positions = rng.uniform(-3, 3, (n, 3))

    # Ensure minimum distance
    final_pos = _enforce_min_distance(best_positions, rng)

    _best_results[n] = final_pos
    return final_pos
