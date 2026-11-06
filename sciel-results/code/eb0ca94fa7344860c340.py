"""Dzugutov cluster global minima: icosahedral seeds + simulated annealing + polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 280.0  # safety margin under 300s kill
_PER_INSTANCE = 45.0  # per-instance budget with margin


def _elapsed():
    return time.time() - _START


def _remaining():
    return TIME_LIMIT - _elapsed()


# Dzugutov parameters
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16


def _dzugutov_energy(pos):
    """Vectorised Dzugutov energy."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    r = r[iu]
    e = np.zeros_like(r)
    m1 = r < _a
    m2 = (r >= _a) & (r < _b)
    if np.any(m1):
        rm = r[m1]
        e[m1] = _A * (rm ** (-_m) - _B) * np.exp(_c / (rm - _a))
    if np.any(m2):
        rm = r[m2]
        e[m2] = _B * np.exp(_d / (rm - _b))
    return float(np.sum(e))


def _generate_icosahedron(n, rng):
    """Generate n atoms in a 12-icosahedral (polytetrahedral) arrangement."""
    # 12 icosahedral vertices (golden ratio based)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    # Scale to typical Dzugutov nearest-neighbour distance ~1.0
    scale = 1.0
    verts *= scale

    # Build a multi-shell icosahedral cluster
    points = [np.zeros(3)]  # center
    # Shell 1: 12 vertices
    points.extend(verts.tolist())
    # Shell 2: 30 (edge midpoints) + 20 (face centers) = 50
    # Edge midpoints: midpoints of 30 icosahedron edges
    # For simplicity, generate more shells by scaling
    shells = [1.0, 1.0 * np.sqrt(2), 1.0 * np.sqrt(3)]
    # Additional vertices at larger radii
    for s in shells[1:]:
        # Use icosahedral directions scaled
        points.extend((verts * s).tolist())
    # Fill remaining with random points in a spherical shell
    target = n
    while len(points) < target:
        # Random point on sphere
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        r = 1.0 + 0.3 * rng.random()
        points.append(v * r)
    points = np.array(points[:target])
    # Small random perturbation to break symmetry
    points += rng.normal(scale=0.02, size=points.shape)
    return points


def _generate_bcc_cluster(n, rng):
    """Generate n atoms in a BCC-like compact cluster."""
    # BCC lattice points in a cube
    a = 1.0
    points = []
    # Cubic + BCC
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                points.append([i * a, j * a, k * a])
                points.append([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a])
    # Add more shells
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                points.append([i * a, j * a, k * a])
                points.append([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a])
    rng.shuffle(points)
    points = np.array(points[:n])
    # Center
    points -= points.mean(axis=0)
    points += rng.normal(scale=0.02, size=points.shape)
    return points


def _generate_random_compact(n, rng):
    """Random compact cluster."""
    radius = 1.1 * n ** (1.0 / 3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _simulated_annealing(pos, n, budget, rng):
    """Simulated annealing with gradient-based moves."""
    T0 = 2.0
    T_min = 0.001
    steps = max(100, int(budget * 20))
    cooling = (T_min / T0) ** (1.0 / steps)
    T = T0
    E = _dzugutov_energy(pos)
    best_pos = pos.copy()
    best_E = E
    step_size = 0.1
    t0 = time.time()
    for it in range(steps):
        if time.time() - t0 > budget:
            break
        # Gradient step
        try:
            grad = dzugutov_energy_gradient(pos)
            pos_new = pos - step_size * grad
        except Exception:
            pos_new = pos + rng.normal(scale=step_size, size=pos.shape)
        # Add noise
        pos_new = pos_new + rng.normal(scale=0.05 * T, size=pos_new.shape)
        # Check minimum distance
        if n > 1:
            diff = pos_new[:, None, :] - pos_new[None, :, :]
            r = np.linalg.norm(diff, axis=2)
            iu = np.triu_indices(n, k=1)
            if np.min(r[iu]) < 0.4:
                pos_new = pos + rng.normal(scale=0.05, size=pos.shape)
        E_new = _dzugutov_energy(pos_new)
        delta = E_new - E
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            pos = pos_new
            E = E_new
            if E < best_E:
                best_E = E
                best_pos = pos.copy()
        T *= cooling
        step_size = max(0.01, step_size * 0.999)
    return best_pos, best_E


def _polish(pos, budget, rng):
    """High-precision polish using the provided relax function."""
    try:
        energy, positions = relax(pos, 'dzugutov', None)
        return positions, energy
    except Exception:
        return pos, _dzugutov_energy(pos)


def construct(instance):
    """Icosahedral seeds + simulated annealing + polish."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(size)

    t_start = time.time()
    per_budget = min(_PER_INSTANCE, max(2.0, _remaining() - 5.0))

    # Generate diverse seeds
    seeds = []
    seeds.append(_generate_icosahedron(size, rng))
    seeds.append(_generate_icosahedron(size, rng))
    seeds.append(_generate_bcc_cluster(size, rng))
    seeds.append(_generate_random_compact(size, rng))
    seeds.append(_generate_random_compact(size, rng))

    # Budget allocation: SA on each seed, then polish best
    sa_budget = per_budget * 0.6 / len(seeds)
    results = []
    for i, seed in enumerate(seeds):
        if time.time() - t_start > per_budget * 0.7:
            break
        budget_i = min(sa_budget, max(0.5, per_budget * 0.7 - (time.time() - t_start)))
        pos_sa, E_sa = _simulated_annealing(seed.copy(), size, budget_i, rng)
        results.append((E_sa, pos_sa))

    # Also try direct relax on a couple seeds (fast local convergence)
    for i in range(min(2, len(seeds))):
        if time.time() - t_start > per_budget * 0.85:
            break
        try:
            energy, positions = relax(seeds[i].copy(), potential, parameter)
            results.append((energy, positions))
        except Exception:
            pass

    if not results:
        # Fallback: just relax the first seed
        try:
            energy, positions = relax(seeds[0].copy(), potential, parameter)
            return positions
        except Exception:
            return seeds[0]

    # Pick best and polish
    results.sort(key=lambda x: x[0])
    best_E, best_pos = results[0]

    # Polish the top few candidates
    polish_budget = max(0.5, per_budget - (time.time() - t_start) - 2.0)
    best_final_E = best_E
    best_final_pos = best_pos
    for E_c, pos_c in results[:3]:
        if time.time() - t_start > per_budget - 1.0:
            break
        pb = max(0.3, polish_budget / 3.0)
        pos_p, E_p = _polish(pos_c.copy(), pb, rng)
        if E_p < best_final_E:
            best_final_E = E_p
            best_final_pos = pos_p

    return best_final_pos
# EVOLVE-BLOCK-END
