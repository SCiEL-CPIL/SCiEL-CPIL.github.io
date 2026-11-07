import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        calls = 0
        best_x = None
        best_f = np.inf
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                raise TimeoutError("Time limit exceeded")
            calls += 1
            if calls > budget:
                raise Exception("Budget exceeded")
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
            return val

        # Global search with CMA-ES
        global_budget = int(0.70 * budget)
        lam = 16
        mu = lam // 2
        sigma0 = 2.2
        c1 = 2 / ((dim + 1.3) ** 2 + mu)
        cmu = (min(mu - 1, 2) / (dim + mu)) * (1 - c1)
        cs = (mu + 2) / (dim + mu + 5)
        damps = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + cs
        cc = (4 + dim / mu) / (dim + 4 + 2 * mu / dim)

        # Initial population: Latin hypercube + random for diversity
        rng = np.random.default_rng(42)
        x_m = rng.uniform(-4, 4, size=dim)
        B = rng.standard_normal((dim, lam))
        np.linalg.matrix_rank(B)
        C = np.eye(dim)
        ps = np.zeros(dim)
        chi = np.sqrt(dim) * (1 + 1/(2*dim) - 1/(4*dim**2))

        # Generate initial population
        P = np.zeros((lam, dim))
        for i in range(lam):
            P[i] = x_m + sigma0 * (B[:, i] / np.linalg.norm(B[:, i]))
        P = np.clip(P, -5, 5)

        # Evaluate initial population
        F = np.array([wrapped_func(P[i]) for i in range(lam)])
        idx = np.argsort(F)
        # Sort
        P = P[idx]
        F = F[idx]
        x_w = P[:mu]

        # CMA-ES main loop
        for gen in range(max(1, int(global_budget / lam))):
            if not time_ok() or calls >= budget:
                break

            # Update weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1 / np.sum(w ** 2)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, (mu_eff - 2) / (dim + mu_eff - 2)) * (1 - c1)

            # Update mean
            x_m = np.sum(w[:, None] * x_w, axis=0)

            # Evolution path
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C @ (np.sqrt(sigma0) * (w[:, None] * x_w) @ np.linalg.inv(C).T)

            # Check path norm for sigma control
            if np.linalg.norm(ps) / chi > 1.4 + 2 / (dim + 1):
                hsig = 0
            else:
                hsig = 1

            # Update covariance
            C = (1 - c1 - cmu + hsig * cs * (1 - cs)) * C
            C += c1 * np.outer(ps, ps)
            C += cmu * np.sum(w[:, None, None] * np.outer(x_w, x_w), axis=0)

            # Ensure positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Compute sigma update
            hsig_vec = np.zeros(dim)
            if hsig == 1:
                hsig_vec = (2 / damps - 1) * x_m
            sigma0 *= np.exp(cs / damps * (np.linalg.norm(ps) / chi - 1))
            sigma0 = np.clip(sigma0, 1e-12, 10.0)

            # Generate new population
            D = np.sqrt(np.diag(C))
            B = np.zeros((dim, lam))
            for i in range(lam):
                z = rng.standard_normal(dim)
                # Whitened
                y = C @ z / np.sqrt(np.sum((C @ z) * z)) if np.sum((C @ z) * z) > 1e-20 else z
                B[:, i] = y / np.linalg.norm(y) if np.linalg.norm(y) > 1e-10 else z / np.linalg.norm(z)

            P_new = np.zeros((lam, dim))
            for i in range(lam):
                P_new[i] = x_m + sigma0 * B[:, i]
            P_new = np.clip(P_new, -5, 5)

            # Evaluate
            F_new = np.array([wrapped_func(P_new[i]) for i in range(lam)])
            idx = np.argsort(F_new)
            P_new = P_new[idx]
            F_new = F_new[idx]
            x_w = P_new[:mu]

        # Local refinement with (1+1)-ES from best point
        remaining = budget - calls
        if remaining > 5 and best_x is not None and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            local_rng = np.random.default_rng(7)
            
            for _ in range(remaining):
                if not time_ok() or calls >= budget:
                    break
                y = x + sigma * local_rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = wrapped_func(y)
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.8
                if sigma < 1e-10:
                    break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
