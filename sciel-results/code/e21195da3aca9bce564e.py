"""Morse cluster global minima: lattice-seeded simulated annealing + gradient polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global deadline guard (evaluator kills at 330 s wall-clock).
GLOBAL_DEADLINE = time.time() + 315.0  # safety margin


def _remaining():
    return max(0.0, GLOBAL_DEADLINE - time.time())


def _morse_energy(pos, rho):
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    dist = dist[mask]
    x = rho * (1.0 - dist)
    ex = np.exp(x)
    return float(np.sum(ex * (ex - 2.0)))


def _morse_gradient(pos, rho):
    n = pos.shape[0]
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-16)
    mask = ~np.eye(n, dtype=bool)
    x = rho * (1.0 - dist)
    ex = np.exp(x)
    dE_dr = -2.0 * rho * ex * (ex - 1.0)
    dE_dr = np.where(mask, dE_dr, 0.0)
    inv = np.where(dist > 1e-12, 1.0 / dist, 0.0)
    forces = dE_dr[:, :, None] * diff * inv[:, :, None]
    return np.sum(forces, axis=1)


def _local_polish(pos, rho, maxiter=300):
    """Gradient-based local minimization of the pure Morse energy."""
    n = pos.shape[0]

    def obj_grad(x):
        p = x.reshape(-1, 3)
        return _morse_energy(p, rho), _morse_gradient(p, rho).flatten()

    try:
        res = minimize(obj_grad, pos.reshape(-1), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "gtol": 1e-9, "ftol": 1e-12})
        return float(res.fun), res.x.reshape(-1, 3)
    except Exception:
        return _morse_energy(pos, rho), pos


def _separate(pos, min_dist=0.5):
    """Geometrically enforce a minimum pair distance (no energy penalty)."""
    n = pos.shape[0]
    pos = pos.copy()
    for _ in range(80):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                r = np.linalg.norm(d)
                if r < min_dist:
                    if r > 1e-10:
                        push = (min_dist - r) / r * 0.5
                        pos[i] += d * push
                        pos[j] -= d * push
                    else:
                        pos[i] += np.array([0.5, 0.0, 0.0])
                    moved = True
        if not moved:
            break
    return pos


def _compact_seed(n, rng):
    """Build a compact, near-closed-shell packing that already respects the floor.

    Strategy: generate a dense icosahedral/FCC-like shell packing by placing
    atoms on concentric shells with jitter, then relax the geometry to a local
    minimum. This starts the search in a physically plausible basin.
    """
    # Concentric shell model: fill shells of an icosahedral/FCC packing.
    # Approximate shell radii for a near-spherical cluster of n atoms.
    points = []
    # Core
    points.append(np.zeros(3))
    # Layered shells (12, 20, 30, 42, ...) with Fibonacci-sphere distribution
    shell_counts = [12, 20, 30, 42, 56, 72, 90, 110]
    radii = [1.0, 1.62, 2.24, 2.86, 3.48, 4.1, 4.72, 5.34]
    idx = 1
    for sc, rad in zip(shell_counts, radii):
        for k in range(sc):
            if idx >= n:
                break
            # Fibonacci sphere
            phi = np.arccos(1.0 - 2.0 * (k + 0.5) / sc)
            theta = 2.0 * np.pi * k / sc
            p = rad * np.array([np.sin(phi) * np.cos(theta),
                                np.sin(phi) * np.sin(theta),
                                np.cos(phi)])
            p += rng.normal(0, 0.05, 3)
            points.append(p)
            idx += 1
        if idx >= n:
            break
    # Fill remaining with a random compact placement
    while len(points) < n:
        r = np.cbrt(len(points) / n) * 4.0
        p = rng.normal(0, r / 2.0, 3)
        points.append(p)
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    points = _separate(points, 0.5)
    return points


def _anneal(pos, rho, n, t_limit, rng):
    """Simulated annealing with geometric distance floor + periodic polish."""
    pos = _separate(pos, 0.5).copy()
    e = _morse_energy(pos, rho)
    best_e = e
    best_pos = pos.copy()

    e_scale = max(1.0, abs(e) / n)
    T0 = 0.35 * e_scale
    T_end = 1e-4 * e_scale
    t_start = time.time()

    # Number of anneal steps scaled to the time budget
    step_budget = int(t_limit / 0.01)  # ~100 ms per 100 steps
    step_budget = max(200, min(step_budget, 20000))
    cool = (T_end / T0) ** (1.0 / max(1, step_budget))

    T = T0
    step = 0
    while T > T_end and (time.time() - t_start) < t_limit and _remaining() > 5.0:
        # Langevin kick
        amp = np.sqrt(2.0 * T * 0.05)
        trial = pos + rng.normal(0, amp, (n, 3))
        trial = _separate(trial, 0.5)
        e_trial = _morse_energy(trial, rho)
        de = e_trial - e
        if de < 0 or rng.random() < np.exp(-de / max(T, 1e-9)):
            pos = trial
            e = e_trial
            if e < best_e:
                best_e = e
                best_pos = pos.copy()
        T *= cool
        step += 1
        # Periodic gradient polish to stay in a good basin
        if step % 400 == 0 and _remaining() > 10.0:
            e_pol, pos_pol = _local_polish(pos, rho, maxiter=120)
            pos_pol = _separate(pos_pol, 0.5)
            if e_pol < best_e:
                best_e = e_pol
                best_pos = pos_pol.copy()
            e = e_pol
            pos = pos_pol

    # Final polish
    if _remaining() > 5.0:
        e_final, pos_final = _local_polish(pos, rho, maxiter=400)
        pos_final = _separate(pos_final, 0.5)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final.copy()
    return best_e, best_pos


def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))

    base_rng = np.random.default_rng(hash((n, rho)) % (2 ** 32))

    # Build diverse compact seeds
    seeds = []
    for _ in range(4):
        seeds.append(_compact_seed(n, base_rng))

    # Time budget per chain
    per_chain = max(2.0, _remaining() * 0.9 / 4.0)

    best_e = np.inf
    best_pos = seeds[0].copy()

    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = []
        for i, s in enumerate(seeds):
            r = np.random.default_rng(hash((n, rho, i)) % (2 ** 32))
            futs.append(ex.submit(_anneal, s, rho, n, per_chain, r))
        for f in as_completed(futs):
            try:
                e, p = f.result()
                if e < best_e:
                    best_e = e
                    best_pos = p.copy()
            except Exception:
                pass

    # Final global polish if time allows
    if _remaining() > 5.0:
        e, p = _local_polish(best_pos, rho, maxiter=500)
        p = _separate(p, 0.5)
        if e < best_e:
            best_e = e
            best_pos = p

    return best_pos
# EVOLVE-BLOCK-END
