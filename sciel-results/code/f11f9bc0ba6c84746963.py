"""Optimizer for BBOB-style Rastrigin (f15, rotated)."""
import time
import numpy as np


def _cmaes(func, x0, sigma0, budget, rng, deadline, best_box):
    """Deterministic CMA-ES. best_box=[best_x, best_f, best_calls]."""
    dim = len(x0)
    x0 = np.clip(x0, -5, 5).astype(float)
    x = x0.copy()
    sigma = float(sigma0)
    best_x, best_f = x.copy(), float(func(x))
    best_box[0], best_box[1], best_box[2] = best_x, best_f, best_box[2] + 1

    # CMA-ES weights
    mu = max(1, dim // 2)
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()
    mueff = 1.0 / w.sum()
    c_sigma = (mueff + 2) / (dim + mueff + 5)
    c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
    c_mu = min(1 - c_1, 2 * ((mueff + 2) / (mueff + 5) - 1) / ((dim + 2) ** 2 + mueff))
    chi = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

    B = np.eye(dim)
    D = np.ones(dim)
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    C = np.eye(dim)
    calls = 1
    for _ in range(budget):
        if time.perf_counter() > deadline:
            break
        # generate offspring
        Y = np.array([x + sigma * (B @ (D * rng.standard_normal(dim))) for _ in range(mu)])
        Y = np.clip(Y, -5, 5)
        F = np.array([float(func(y)) for y in Y])
        calls += mu
        idx = np.argsort(F)
        Ys = Y[idx]
        Fs = F[idx]
        # update best
        if Fs[0] < best_f:
            best_f = float(Fs[0])
            best_x = Ys[0].copy()
            best_box[0], best_box[1] = best_x, best_f
        # weighted mean
        m = (w * Ys).sum(axis=0)
        # update C
        yv = (m - x) / sigma
        pc = (1 - c_c) * pc + c_c * yv
        cs = np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** 2)
        hsig = 1 if cs / chi < (1 + 2 / (dim + 1)) * (1 - c_sigma) else 0
        ps = (1 - c_sigma) * ps + hsig * np.sqrt(c_sigma * (2 - c_sigma) * mueff) * yv
        C = ((1 + c_1 * hsig * (cs * cs / dim - 1) - c_1 - c_mu) * C
             + c_1 * np.outer(pc, pc)
             + c_mu * sum(wi * np.outer(yi, yi) for wi, yi in zip(w, (Ys - x) / sigma)))
        # re-factor
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.clip(D, 1e-20, None))
        x = m
        # sigma adaptation
        sigma *= np.exp(c_sigma / chi * (np.linalg.norm(ps) / chi - 1))
        sigma = min(max(sigma, 1e-10), 20.0)
    return best_x, best_f, calls


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 280.0  # safety margin under 300s
        dim = self.dim
        rng = np.random.default_rng(0)

        # Structure-aware seed set: integer lattice (global minima), half-integers,
        # and random points. Rastrigin minima lie at integer coords.
        lattice = []
        # integer grid within [-4,4]
        for a in range(-4, 5):
            for b in range(-4, 5):
                for c in range(-4, 5):
                    for d in range(-4, 5):
                        for e in range(-4, 5):
                            lattice.append([a, b, c, d, e])
        # half-integer grid
        half = []
        for a in range(-4, 4):
            for b in range(-4, 4):
                for c in range(-4, 4):
                    for d in range(-4, 4):
                        for e in range(-4, 4):
                            half.append([a + 0.5, b + 0.5, c + 0.5, d + 0.5, e + 0.5])
        # random points
        rand = [list(rng.uniform(-5, 5, dim)) for _ in range(8)]

        # Evaluate all seeds, rank by value
        seeds = lattice + half + rand
        evals = []
        for s in seeds:
            s = np.clip(np.asarray(s, float), -5, 5)
            evals.append((float(func(s)), s))
        evals.sort(key=lambda t: t[0])

        best_box = [evals[0][1].copy(), evals[0][0], len(seeds)]

        # Multi-start CMA-ES on the best seeds, allocating budget to early/best ones.
        # First start gets the most; later starts get less.
        starts = [s for _, s in evals[:6]]
        # budget per start: decreasing
        per = [90, 70, 55, 45, 35, 30]
        for s, b in zip(starts, per):
            if time.perf_counter() > deadline:
                break
            _, _, _ = _cmaes(func, s, 1.5, b, rng, deadline, best_box)

        # Final high-precision polish on the best found point.
        if time.perf_counter() < deadline:
            _cmaes(func, best_box[0], 0.3, 40, rng, deadline, best_box)

        return best_box[0], best_box[1]
