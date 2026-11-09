"""Dzugutov cluster global minima search.
Combines high-quality polytetrahedral seeding with basin hopping global optimization
and parallel basin exploration via multiprocessing."""

import numpy as np
import time
import multiprocessing as mp
from scipy.optimize import minimize
import autoresearch_suite.chem_clusters as chem_clusters
from autoresearch_suite.chem_clusters import relax

# Dzugutov Potential Parameters
A_POT = 5.82
B_POT = 1.28
A_CUT = 1.87
B_CUT = 1.94
C_EXP = 1.10
D_EXP = 0.27
M_EXP = 16

# Global deadline guard
TOTAL_TIME_LIMIT = 305.0
_start_time = None
_best_results = {}


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


def _make_mackay_icosahedron(n, rng=None):
    """Generate a Mackay icosahedron cluster with n atoms using proper shell radii."""
    if rng is None:
        rng = np.random.default_rng(42)

    phi = (1 + np.sqrt(5)) / 2

    # 12 vertices of icosahedron (unit radius)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    # Shell 0: Center (1 atom)
    positions = [np.zeros(3)]
    remaining = n - 1

    # Shell 1: 12 vertices at radius r1
    r1 = 1.0
    shell1 = verts * r1
    take1 = min(12, remaining)
    positions.extend([p.copy() for p in shell1[:take1]])
    remaining -= take1

    # Shell 2: 30 edge midpoints at radius r2
    if remaining > 0:
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.25:
                    mid = (verts[i] + verts[j]) / 2.0
                    mid_norm = np.linalg.norm(mid)
                    if mid_norm > 1e-8:
                        mid = mid / mid_norm
                        edges.append(mid)

        unique_edges = []
        for e in edges:
            if all(np.linalg.norm(e - ue) > 0.1 for ue in unique_edges):
                unique_edges.append(e)

        r2 = np.sqrt(2 * (1 + 1.0 / phi))
        shell2 = np.array(unique_edges) * r2
        take2 = min(len(shell2), remaining)
        positions.extend([p.copy() for p in shell2[:take2]])
        remaining -= take2

    # Shell 3: 20 face centers at radius r3
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
            fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            fc_norm = np.linalg.norm(fc)
            if fc_norm > 1e-8:
                fc = fc / fc_norm
                face_centers.append(fc)

        r3 = np.sqrt(3 * (1 + 1.0 / phi))
        shell3 = np.array(face_centers) * r3
        take3 = min(len(shell3), remaining)
        positions.extend([p.copy() for p in shell3[:take3]])
        remaining -= take3

    # If still remaining, add atoms on a larger shell
    if remaining > 0:
        r4 = 2.8
        for _ in range(remaining):
            p = rng.normal(size=3)
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-8:
                p = np.array([1.0, 0.0, 0.0])
            else:
                p = p / p_norm
            positions.append(p * r4)

    # If too many atoms, remove from outermost shell
    if len(positions) > n:
        pos_arr = np.array(positions)
        radii = np.linalg.norm(pos_arr, axis=1)
        indices = np.argsort(radii)[:n]
        positions = [pos_arr[i] for i in indices]

    return np.array(positions)


def _make_fcc_cluster(n, rng=None):
    """Generate a compact FCC-like cluster."""
    if rng is None:
        rng = np.random.default_rng(42)

    positions = [np.zeros(3)]
    remaining = n - 1

    a = 1.0
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
        p = rng.uniform(-4, 4, 3)
        positions.append(p)

    return np.array(positions[:n])


def _make_bcc_cluster(n, rng=None):
    """Generate a compact BCC-like cluster."""
    if rng is None:
        rng = np.random.default_rng(42)

    positions = [np.zeros(3)]
    remaining = n - 1

    a = 1.0
    lattice_pts = []
    for ix in range(-3, 4):
        for iy in range(-3, 4):
            for iz in range(-3, 4):
                p = np.array([ix, iy, iz]) * a
                lattice_pts.append(p)
                pc = np.array([ix + 0.5, iy + 0.5, iz + 0.5]) * a
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
        p = rng.uniform(-3, 3, 3)
        positions.append(p)

    return np.array(positions[:n])


def _make_random_compact(n, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng(42)

    positions = []
    radius = 1.2 * n ** (1.0 / 3.0)
    min_dist = 0.6
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


def _enforce_min_distance(positions, min_dist=0.5):
    """Separate any atoms closer than min_dist."""
    pos = positions.copy()
    for _ in range(30):
        needs_fix = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    needs_fix = True
                    diff = pos[i] - pos[j]
                    diff_norm = np.linalg.norm(diff)
                    if diff_norm < 1e-8:
                        diff = np.array([1.0, 0.0, 0.0])
                    else:
                        diff = diff / diff_norm
                    push = (min_dist - d) / 2 + 0.01
                    pos[i] += diff * push
                    pos[j] -= diff * push
        if not needs_fix:
            break
    return pos


def _relax_positions(positions):
    """Relax positions using the relax API, with L-BFGS fallback."""
    try:
        energy, pos = relax(positions, 'dzugutov')
        if energy is not None and pos is not None:
            return energy, pos
    except Exception:
        pass

    # Fallback: L-BFGS optimization
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
            options={'maxiter': 50, 'maxfun': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_x = result.x.copy()
    except Exception:
        pass

    return best_energy, best_x.reshape(-1, 3)


def _basin_hopping(positions, n_kicks, kick_scale, time_budget, seed):
    """Basin hopping: kick, relax, accept/reject with Metropolis criterion."""
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_positions(positions)
    best_energy = current_energy
    best_pos = current_pos.copy()

    t_start = time.time()
    T = 0.5  # Initial temperature for Metropolis acceptance

    for kick_idx in range(n_kicks):
        if time.time() - t_start > time_budget:
            break
        if _remaining() < 5:
            break

        # Adaptive kick scale: larger kicks early, smaller later
        progress = kick_idx / max(1, n_kicks)
        current_kick_scale = kick_scale * (1.0 - 0.5 * progress)

        # Kick: random perturbation
        kick = rng.normal(0, current_kick_scale, positions.shape)
        kicked = current_pos + kick
        kicked = _enforce_min_distance(kicked)

        # Relax
        new_energy, new_pos = _relax_positions(kicked)

        if new_energy is None:
            continue

        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-delta_e / T)

        if accept:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()

        # Cool down
        T *= 0.97

    return best_energy, best_pos


def _worker_basin_hopping(args):
    """Worker for parallel basin hopping."""
    positions, n_kicks, kick_scale, time_budget, seed = args
    try:
        best_energy, best_pos = _basin_hopping(positions, n_kicks, kick_scale, time_budget, seed)
        return best_energy, best_pos
    except Exception:
        return None, None


def construct(instance):
    """Find low-energy structures of Dzugutov clusters."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = instance["n"]

    if n in _best_results:
        return _best_results[n]

    # Time budget for this instance
    instances_to_do = [38, 45, 50, 55, 57, 60]
    idx = instances_to_do.index(n) if n in instances_to_do else 0
    base_time_per_instance = TOTAL_TIME_LIMIT / 6.0
    time_budget = min(_remaining() * 0.9, base_time_per_instance * 1.5)
    time_budget = max(time_budget, 15.0)

    # Generate diverse initial structures
    rng = np.random.default_rng(42 + n)
    candidates = []

    # 1. Mackay icosahedron (primary seed)
    mackay_pos = _make_mackay_icosahedron(n, rng)
    candidates.append(mackay_pos)

    # 2. Perturbed Mackay variants
    for seed_offset in [1, 2, 3]:
        perturbed = mackay_pos + rng.normal(0, 0.1, mackay_pos.shape)
        candidates.append(perturbed)

    # 3. FCC-like
    fcc_pos = _make_fcc_cluster(n, rng)
    candidates.append(fcc_pos)

    # 4. BCC-like
    bcc_pos = _make_bcc_cluster(n, rng)
    candidates.append(bcc_pos)

    # 5. Random compact
    random_pos = _make_random_compact(n, rng)
    candidates.append(random_pos)

    # 6. Another random compact with different seed
    rng2 = np.random.default_rng(1000 + n)
    random_pos2 = _make_random_compact(n, rng2)
    candidates.append(random_pos2)

    best_energy = np.inf
    best_positions = None

    # First: relax all candidates to get initial energies
    relaxed_candidates = []
    for cand in candidates:
        pos = _enforce_min_distance(cand)
        energy, relaxed = _relax_positions(pos)
        if energy is not None and relaxed is not None:
            relaxed_candidates.append((energy, relaxed))
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed

    if not relaxed_candidates:
        # Fallback
        pos = _enforce_min_distance(candidates[0])
        energy, relaxed = _relax_positions(pos)
        if relaxed is not None:
            best_positions = relaxed
            best_energy = energy
        else:
            best_positions = pos

    # Now do basin hopping from the best few starting points
    relaxed_candidates.sort(key=lambda x: x[0])
    top_candidates = relaxed_candidates[:min(4, len(relaxed_candidates))]

    # Parallel basin hopping
    num_workers = min(4, len(top_candidates))
    per_worker_time = time_budget * 0.65 / num_workers

    # Number of kicks depends on time budget and n
    base_kicks = max(8, int(40 / (n / 40)))
    kick_scale = 0.12 + 0.04 * (n / 60)

    worker_args = []
    for i, (energy, pos) in enumerate(top_candidates):
        seed = 2000 + n * 10 + i
        n_kicks = base_kicks
        worker_args.append((pos, n_kicks, kick_scale, per_worker_time, seed))

    try:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_worker_basin_hopping, worker_args)

        for energy, pos in results:
            if energy is not None and pos is not None:
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos

    except Exception:
        # Fallback: sequential basin hopping on best candidate
        if top_candidates:
            pos = top_candidates[0][1]
            energy, new_pos = _basin_hopping(pos, base_kicks, kick_scale, time_budget * 0.5, seed=9999)
            if energy is not None and energy < best_energy:
                best_energy = energy
                best_positions = new_pos

    # Final relaxation of the best result
    if best_positions is not None and _remaining() > 5:
        final_energy, final_pos = _relax_positions(best_positions)
        if final_energy is not None and final_pos is not None:
            best_energy = final_energy
            best_positions = final_pos

    # Ensure minimum distance
    if best_positions is not None:
        best_positions = _enforce_min_distance(best_positions)
    else:
        best_positions = _enforce_min_distance(candidates[0])

    _best_results[n] = best_positions
    return best_positions
