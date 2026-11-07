# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (robust, standard) with a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280.0   # safety margin below 300s kill
        n = self.dim
        budget = self.budget

        # CMA-ES hyper-parameters
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = (2 - cs - min(1, (mueff - 2 + 1 / n) / (n + 3))) / ((n + 1) ** 2 + 2)
        cmu = min(1 - c1, 2 * (mueff - 2 + 2 / mueff) / ((n + 2) ** 2 + 2))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # initial state
        xmean = np.zeros(n)
        sigma = 2.0
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        count_eval = 0
        count_steps = 0

        # evaluation wrapper: record + time guard
        def evaluate(x):
            nonlocal count_eval
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            count_eval += 1
            return float(func(x))

        # initial point
        x = evaluate(xmean)
        best_x, best_f = xmean.copy(), x
        it = 0

        while count_eval < budget:
            if time.time() > deadline:
                break

            # sample
            z = np.random.standard_normal((lam, n))
            pop = xmean + sigma * (z @ (B * D))
            fvals = np.empty(lam)
            for j in range(lam):
                if count_eval >= budget:
                    break
                if time.time() > deadline:
                    break
                fvals[j] = evaluate(pop[j])
            if count_eval >= budget or time.time() > deadline:
                break

            order = np.argsort(fvals)
            x_sorted = pop[order]
            f_sorted = fvals[order]

            # update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()

            # recombination
            wsum = np.sum(w[:mu])
            y_w = np.sum(w[:mu, None] * x_sorted[:mu], axis=0)
            y_w = y_w / wsum

            # update step-size control
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) / damps) * (
                (B @ (D * ((B.T @ y_w) / sigma)))
            )
            sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chiN - 1))

            # update evolution path and covariance
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc)) * y_w
            C = (B @ (np.outer(D, D)) @ B.T)
            H1 = np.outer(p_c, p_c)
            H2 = np.sum(w[:mu, None, None] * np.outer(x_sorted[:mu] - xmean,
                                                     x_sorted[:mu] - xmean), axis=0)
            C = (1 - (c1 + cmu)) * C + c1 * (H1 + (cc * (2 - cc)) * C) \
                + cmu / wsum * H2

            # ensure symmetry
            C = 0.5 * (C + C.T)

            # recompute eigen-decomposition
            try:
                eigvals, B = np.linalg.eigh(C)
            except np.linalg.LinAlgError:
                eigvals, B = np.linalg.eigh(0.5 * (C + C.T))
            eigvals = np.maximum(eigvals, 1e-16)
            D = np.sqrt(eigvals)

            # update mean
            xmean = xmean + y_w
            it += 1
            count_steps += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
