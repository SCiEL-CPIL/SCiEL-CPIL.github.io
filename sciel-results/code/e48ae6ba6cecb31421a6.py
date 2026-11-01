"""Optimized packing: Langevin repulsion search + hex warm-start + NM polish."""
import numpy as np
import time

N = 200
_TIME_LIMIT = 150.0


def compute_radius(c):
    c = np.asarray(c, dtype=np.float64)
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    r_pair = 0.5 * d.min()
    r_wall = (0.5 - np.abs(c)).min()
    return float(min(r_pair, r_wall))


def smooth_obj(c, soft):
    # negative of a soft-min surrogate of the packing radius (smooth, differentiable)
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    d = np.clip(d, 1e-9, None)
    lp = np.logaddexp.reduce(np.log(d))
    lw = np.logaddexp.reduce(np.log(np.clip(0.5 - np.abs(c), 1e-9, None)))
    lp_soft = soft * np.log(np.exp(lp / soft) - 1.0)
    lw_soft = soft * np.log(np.exp(lw / soft) - 1.0)
    m = np.minimum(lp_soft, lw_soft)
    return -m


def hex_initial():
    # Dense hexagonal lattice (warm start), clipped to the square.
    m = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(-0.5, 0.5, m)
    dx = xs[1] - xs[0]
    dy = dx * np.sqrt(3.0) / 2.0
    pts = []
    row = 0
    while len(pts) < N:
        off = 0.5 * dx if (row % 2) else 0.0
        y = -0.5 + dy * (row + 0.5)
        if y > 0.5:
            break
        for x in xs:
            p = (x + off, y)
            if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5:
                pts.append(p)
            if len(pts) >= N:
                break
        row += 1
    if len(pts) < N:
        # top up with a fine grid
        g = int(np.ceil(np.sqrt(N)))
        tg = (np.arange(g) + 0.5) / g - 0.5
        for yy in tg:
            for xx in tg:
                if len(pts) >= N:
                    break
                if -0.5 <= xx <= 0.5 and -0.5 <= yy <= 0.5:
                    pts.append((xx, yy))
    return np.array(pts[:N], dtype=np.float64)


def forces(c, soft):
    # Repulsive force = gradient of smooth_obj (drives circles apart / away from walls).
    diff = c[:, None, :] - c[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(d, 1.0)
    inv = 1.0 / np.clip(d, 1e-9, None)
    rep = (diff * (inv[..., None] / soft))
    rep = rep.sum(axis=1)
    wall = np.sign(c) / soft
    return rep + wall


def langevin(c, t0, t_end, n_iter, max_step, seed):
    rng = np.random.default_rng(seed)
    c = c.copy()
    best = c.copy()
    best_f = smooth_obj(best, 1e-3)
    for i in range(n_iter):
        frac = i / max(1, n_iter - 1)
        T = t0 * (t_end / t0) ** frac
        f = forces(c, T)
        step = max_step * (1.0 - frac)
        noise = rng.standard_normal(c.shape) * np.sqrt(T) * step
        c = c + f * step + noise
        c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
        if (i % 25 == 0) or i == n_iter - 1:
            f = smooth_obj(c, 1e-3)
            if f > best_f:
                best_f = f
                best = c.copy()
    return best


def polish(c, time_left):
    # Nelder-Mead polish on the smooth objective; bail before deadline.
    from scipy.optimize import minimize
    t_start = time.time()
    try:
        res = minimize(
            lambda x: smooth_obj(x, 1e-3),
            c.ravel(),
            method="Nelder-Mead",
            options={
                "maxiter": 4000,
                "xatol": 1e-10,
                "fatol": 1e-12,
                "adaptive": True,
                "initial_simplex": 1e-4,
                "callback": lambda x: None if (time.time() - t_start) < time_left else (_ for _ in ()).throw(RuntimeError("stop")),
            },
        )
        if res.success or res.fun < smooth_obj(c, 1e-3):
            return np.clip(res.x.reshape(N, 2), -0.5 + 1e-9, 0.5 - 1e-9)
    except Exception:
        pass
    return c


def construct():
    t_start = time.time()
    deadline = t_start + _TIME_LIMIT - 8.0  # safety margin
    c = hex_initial()
    best = c.copy()
    best_f = smooth_obj(best, 1e-3)

    # Coarse-to-fine Langevin annealing (larger soft = smoother global moves early).
    schedule = [
        (3e-2, 3e-4, 6000, 3e-3),
        (1e-2, 1e-4, 8000, 2e-3),
        (3e-3, 5e-5, 12000, 1e-3),
        (1e-3, 1e-5, 15000, 5e-4),
    ]
    for si, (t0, t_end, n_iter, max_step) in enumerate(schedule):
        if time.time() > deadline - 25.0:
            break
        for seed in range(3):
            if time.time() > deadline - 25.0:
                break
            cand = langevin(c, t0, t_end, n_iter, max_step, seed=si * 100 + seed)
            cf = smooth_obj(cand, 1e-3)
            if cf > best_f:
                best_f = cf
                best = cand.copy()
                c = cand.copy()

    # Final polish with remaining budget.
    time_left = max(1.0, deadline - time.time())
    best = polish(best, time_left)

    # Hard guard: never exceed the wall-clock limit.
    if time.time() > _TIME_LIMIT - 2.0:
        pass  # return best as-is

    return best
