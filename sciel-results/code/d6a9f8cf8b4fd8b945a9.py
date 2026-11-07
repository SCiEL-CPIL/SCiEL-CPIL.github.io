"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global DE-like phase + CMA-ES local phase, with a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 300.0

        best_x = None
        best_f = np.inf
        calls = 0

        def feval(x):
            nonlocal best_x, best_f, calls
            if time.time() > deadline:
                raise TimeoutError()
            calls += 1
            fx = func(np.clip(x, -5.0, 5.0))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # ---------------- GLOBAL PHASE (DE-like) ----------------
        rng = np.random.default_rng(0)
        N = 12
        F = 0.8
        CR = 0.5

        pop = np.empty((N, dim))
        pf = np.empty(N)
        for i in range(N):
            pop[i] = rng.uniform(-5.0, 5.0, dim)
            pf[i] = feval(pop[i])

        order = np.argsort(pf)
        pop = pop[order]
        pf = pf[order]
        best_x = pop[0].copy()
        best_f = pf[0]
        best_i = 0
        t_since = 0
        gen = 0

        g = 0
        while calls < 300 and g < 2000:
            if time.time() > deadline:
                break
            i = best_i
            x_i = pop[i]
            f_i = pf[i]
            t_since += 1
            if t_since > 20 * N:
                pop[-1] = rng.uniform(-5.0, 5.0, dim)
                pf[-1] = feval(pop[-1])
                order = np.argsort(pf)
                pop = pop[order]
                pf = pf[order]
                best_i = 0
                best_x = pop[0].copy()
                best_f = pf[0]
                t_since = 0
                continue

            r1, r2, r3 = np.random.choice(N, 3, replace=False)
            v = x_i + F * (pop[r1] - pop[r2])
            v = np.clip(v, -5.0, 5.0)
            jrand = rng.integers(0, dim)
            u = np.where(rng.random(dim) < CR, v, x_i)
            u[jrand] = v[jrand]
            fu = feval(u)
            if fu <= f_i:
                pop[i] = u
                pf[i] = fu
                if fu <= best_f:
                    best_f = fu
                    best_x = u.copy()
                    best_i = i
                    t_since = 0
            gen += 1
            if gen % 3 == 0:
                order = np.argsort(pf)
                pop = pop[order]
                pf = pf[order]
                best_i = 0
                best_x = pop[0].copy()
                best_f = pf[0]
                t_since = 0

        # ---------------- LOCAL PHASE (CMA-ES) ----------------
        x0 = best_x.copy()
        scale = 3.0

        def zfeval(z):
            return feval(x0 + scale * z)

        z0 = np.zeros(dim)
        sigma = 1.0
        lambda_ = 4 + int(np.floor(3 * np.log(dim)))
        weights = np.log(lambda_ - np.arange(lambda_) + 0.5)
        mu = int(np.floor(lambda_ / 2))
        weights = weights[:mu]
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()
        c1 = (2 * (mueff - 2 + 2.0 / dim)) / ((dim + 2) ** 2 + mueff)
        cmu = (2 * (2 - c1) * (mueff - 2 + 1.0 / mueff)) / ((dim + 2) ** 2 + mueff)
        cs = (mueff + 2) / (dim + mueff + 5)
        csigma = 1.0 / (1.0 + 2 * cs + mueff / dim)
        chi = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        zmean = z0.copy()

        remaining = budget - calls
        if remaining > 0:
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                zs = np.empty((lambda_, dim))
                fs = np.empty(lambda_)
                for k in range(lambda_):
                    step = sigma * (B @ (D * rng.standard_normal(dim)))
                    zs[k] = zmean + step
                    fs[k] = zfeval(zs[k])
                idx = np.argsort(fs)
                zs = zs[idx]
                fs = fs[idx]
                zmean_old = zmean
                zmean = np.sum(weights[:, None] * zs[:mu], axis=0)
                diff = (zmean - zmean_old) / sigma
                pc = (1 - c1) * pc + c1 * diff
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c1) ** 2)) < (1.4 + 2 / (dim + 1)) * chi
                ps = (1 - csigma) * ps + hsig * np.sqrt(csigma * (2 - csigma) * mueff) * diff
                Y = (zs[:mu] - zmean_old) / sigma
                Cmat = (1 - c1 - cmu) * np.eye(dim) + c1 * np.outer(pc, pc) + cmu * np.sum(
                    weights[:, None, None] * Y[:, :, None] * Y[:, None, :].transpose(0, 2, 1), axis=0
                )
                eigval, eigvec = np.linalg.eigh(Cmat)
                eigval = np.maximum(eigval, 1e-12)
                B = eigvec
                D = np.sqrt(eigval)
                sigma *= np.exp(cs / csigma * (np.linalg.norm(ps) / chi - 1))
                sigma = np.clip(sigma, 1e-8, 50.0)

        # ---------------- FINAL POLISH ----------------
        if best_x is None:
            best_x = np.zeros(dim)
        rng2 = np.random.default_rng(1)
        s = 1e-3
        for _ in range(30):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            cand = np.clip(best_x + s * rng2.standard_normal(dim), -5.0, 5.0)
            fc = feval(cand)
            if fc < best_f:
                best_f = fc
                best_x = cand.copy()
            s *= 0.7

        return best_x, best_f
# EVOLVE-BLOCK-END
