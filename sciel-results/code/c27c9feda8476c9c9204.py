"""Morse cluster global minima: structured lattice seeds + basin hopping + gradient polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from scipy.optimize import minimize

_START = time.time()
_HARD_DEADLINE = _START + 295.0  # safety margin before 300s limit / 330s kill


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy(pos, rho):
    """Exact evaluator Morse energy: sum_{i<j} exp(rho(1-r))(exp(rho(1-r))-2)."""
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    iu = np.triu_indices(n, 1)
    d = d[iu]
    x = np.exp(rho * (1.0 - d))
    return float(np.sum(x * (x - 2.0)))


def _min_dist(pos):
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    if n < 2:
        return 1e10
    diff = pos[:, None, :] - pos[None, :, :]
    d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    iu = np.triu_indices(n, 1)
    return float(d[iu].min())


def _energy_grad(pos, rho):
    res = morse_energy_gradient(np.asarray(pos, dtype=float), rho)
    a, b = res
    if np.ndim(a) == 0:
        return float(a), np.asarray(b, dtype=float)
    else:
        return float(b), np.asarray(a, dtype=float)


def _fcc_points(n, a=np.sqrt(2.0)):
    pts = []
    L = int(np.ceil(n ** (1.0 / 3.0))) + 2
    offs = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]), np.array([0.0, 0.5, 0.5])]
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                base = np.array([i, j, k]) * a
                for off in offs:
                    pts.append(base + off)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:n]
    return pts[idx]


def _bcc_points(n, a=2.0 / np.sqrt(3.0)):
    pts = []
    L = int(np.ceil(n ** (1.0 / 3.0))) + 2
    offs = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                base = np.array([i, j, k]) * a
                for off in offs:
                    pts.append(base + off)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:n]
    return pts[idx]


def _random_compact(n, rng, spacing=1.0):
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    pts = []
    # Use a coarse grid to speed up rejection
    grid = np.round(pts)  # not useful initially
    attempts = 0
    min_d = 0.7 * spacing
    while len(pts) < n and attempts < 5000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < min_d:
                    ok = False
                    break
            if ok:
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        pts.append(rng.uniform(-radius, radius, 3))
    return np.array(pts[:n])


def _enforce_min_dist(pos, min_d=0.5, iters=50):
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    if dist < 1e-9:
                        d = np.array([1e-6, 0, 0])
                        dist = 1e-6
                    push = d / dist * (min_d - dist) * 0.5
                    pos[i] += push
                    pos[j] -= push
                    moved = True
        if not moved:
            break
    return pos


def _basin_hop(start_pos, rho, rng, time_budget, kick_start, kick_end, T_start, T_end):
    pos = np.array(start_pos, dtype=float, copy=True)
    try:
        e, _ = relax(pos, 'morse', rho)
    except Exception:
        e = _morse_energy(pos, rho)
    best_pos = pos.copy()
    best_e = e
    t0 = time.time()
    n_kicks = 0
    while (time.time() - t0) < time_budget and _time_left() > 3.0:
        frac = min(1.0, (time.time() - t0) / max(time_budget, 1e-9))
        kick = kick_start + (kick_end - kick_start) * frac
        T = T_start + (T_end - T_start) * frac
        new_pos = pos + rng.normal(0.0, kick, pos.shape)
        try:
            new_e, new_relaxed = relax(new_pos, 'morse', rho)
        except Exception:
            break
        n_kicks += 1
        new_relaxed = np.asarray(new_relaxed, dtype=float)
        if new_e < e:
            pos, e = new_relaxed, new_e
            if e < best_e:
                best_e = e
                best_pos = pos.copy()
        else:
            if T > 1e-9 and rng.random() < np.exp(-min(50.0, (new_e - e) / T)):
                pos, e = new_relaxed, new_e
    return best_pos, best_e, n_kicks


def _polish(pos, rho, time_budget):
    pos = np.array(pos, dtype=float, copy=True)
    try:
        e0, _ = relax(pos, 'morse', rho)
        e0 = _morse_energy(pos, rho)
    except Exception:
        e0 = _morse_energy(pos, rho)
    best_pos = pos.copy()
    best_e = e0
    t0 = time.time()
    # Gradient-based L-BFGS-B polish
    try:
        def fun(x):
            e, g = _energy_grad(x.reshape(-1, 3), rho)
            return e, g
        res = minimize(fun, pos.ravel(), method='L-BFGS-B', jac=True,
                       options={'maxiter': 300, 'maxfun': 3000,
                                'gtol': 1e-8, 'ftol': 1e-12})
        new_pos = res.x.reshape(-1, 3)
        if _min_dist(new_pos) >= 0.5:
            new_e = _morse_energy(new_pos, rho)
            if new_e < best_e:
                best_e = new_e
                best_pos = new_pos
    except Exception:
        pass
    # Final relax for safety
    if _time_left() - (time.time() - t0) > 0.5 and (time.time() - t0) < time_budget:
        try:
            _, p2 = relax(best_pos, 'morse', rho)
            p2 = np.asarray(p2, dtype=float)
            if _min_dist(p2) >= 0.5:
                e2 = _morse_energy(p2, rho)
                if e2 < best_e:
                    best_e = e2
                    best_pos = p2
        except Exception:
            pass
    return best_pos, best_e


def construct(instance):
    n = int(instance["n"])
    rho = instance.get("rho", instance.get("kind"))
    rho = float(rho)

    inst_start = time.time()
    inst_budget = 48.0

    def inst_left():
        return min(inst_budget, _time_left() - 3.0) - (time.time() - inst_start)

    rng = np.random.default_rng(n * 7919 + int(round(rho * 10)))

    # --- Seeds ---
    seeds = []
    try:
        seeds.append(_fcc_points(n))
        seeds.append(_bcc_points(n))
        seeds.append(_random_compact(n, rng))
        seeds.append(_fcc_points(n) + rng.normal(0, 0.15, (n, 3)))
        seeds.append(_bcc_points(n) + rng.normal(0, 0.15, (n, 3)))
    except Exception:
        seeds.append(_random_compact(n, rng))

    best_pos = None
    best_e = np.inf
    seed_results = []
    for seed in seeds:
        if inst_left() < 2.0:
            break
        try:
            e, p = relax(seed, 'morse', rho)
            p = np.asarray(p, dtype=float)
            e = _morse_energy(p, rho)
        except Exception:
            continue
        if e < best_e:
            best_e = e
            best_pos = p
        seed_results.append((e, p))

    if best_pos is None:
        # Fallback
        best_pos = _random_compact(n, rng)
        try:
            _, best_pos = relax(best_pos, 'morse', rho)
        except Exception:
            pass

    # Sort seeds by energy for chain starts
    seed_results.sort(key=lambda x: x[0])

    # --- Basin hopping ---
    bh_time = max(5.0, inst_left() - 4.0)
    n_chains = 3
    chain_budget = bh_time / n_chains
    schedules = [
        (0.15, 0.04, 2.0, 0.05),
        (0.35, 0.10, 5.0, 0.10),
        (0.60, 0.20, 10.0, 0.20),
    ]
    for ci in range(n_chains):
        if inst_left() < 3.0:
            break
        # Start chain from best available (rotate through top seeds for diversity)
        start = seed_results[ci % len(seed_results)][1] if seed_results else best_pos
        ks, ke, Ts, Te = schedules[ci % len(schedules)]
        cb = min(chain_budget, inst_left() - 2.0)
        if cb < 1.0:
            break
        try:
            cpos, ce, _ = _basin_hop(start, rho, rng, cb, ks, ke, Ts, Te)
        except Exception:
            continue
        if ce < best_e:
            best_e = ce
            best_pos = cpos

    # --- Final polish ---
    if inst_left() > 2.0 and best_pos is not None:
        try:
            ppol, epol = _polish(best_pos, rho, max(1.0, inst_left() - 1.0))
            if epol < best_e:
                best_e = epol
                best_pos = ppol
        except Exception:
            pass

    # --- Safety: enforce min distance ---
    best_pos = np.asarray(best_pos, dtype=float)
    if _min_dist(best_pos) < 0.5:
        best_pos = _enforce_min_dist(best_pos, 0.5)
        # re-relax after enforcement
        try:
            _, best_pos = relax(best_pos, 'morse', rho)
        except Exception:
            pass
        if _min_dist(best_pos) < 0.5:
            # last resort: use best seed
            if seed_results:
                best_pos = seed_results[0][1]

    return best_pos
# EVOLVE-BLOCK-END
