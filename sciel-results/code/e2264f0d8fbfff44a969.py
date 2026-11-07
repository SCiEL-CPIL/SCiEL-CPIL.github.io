"""Bootstrap spawn: CMA-ES + random exploration for 5D Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with coarse random exploration for 5D Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return float(func(x))

        def time_ok():
            return calls[0] < budget and time.time() < deadline

        # Phase 1: Coarse random exploration to find a good basin
        xs = []
        fs = []
        best_x = None
        best_f = np.inf

        n_explore = 60
        for i in range(n_explore):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = call_func(x)
            xs.append(x)
            fs.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        x = best_x.copy()
        fx = best_f

        # Phase 2: CMA-ES to navigate the rotated landscape
        # CMA-ES adapts covariance to the hidden rotation
        n_evals_cma = min(350, budget - calls[0])
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # ~11 for dim=5
        sigma = 1.0  # initial step size

        # CMA-ES state
        mean = x.copy()
        C = np.eye(dim)
        D = np.sqrt(np.diag(C))
        B = np.eye(dim)
        sigma = 1.0
        cs = (np.sqrt(dim + 1.5) - 1) / (dim + 1 + np.sqrt(2))
        ds = 1 + 2 * max(0, (np.sqrt((dim + 1) / (dim + 3)) - 1)) + cs
        cc = (4 + dim / (dim + 4)) / ((dim + 2) + 4 + dim / (dim + 4))
        cmu = min(1 - cc, 2 / ((dim + 1.3) ** 2 + dim))
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        for gen in range(n_evals_cma):
            if not time_ok():
                break

            # Generate population
            pop = []
            pop_f = []
            for i in range(pop_size):
                if not time_ok():
                    break
                z = rng.standard_normal(dim)
                # Transform z using B * D
                xi = mean + sigma * (B @ (D * z))
                xi = np.clip(xi, -5, 5)
                fi = call_func(xi)
                pop.append(xi)
                pop_f.append(fi)
                if fi < fx:
                    fx = fi
                    x = xi.copy()

            if len(pop) < 2:
                break

            pop = np.array(pop)
            pop_f = np.array(pop_f)

            # Sort by fitness
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]

            # Update best
            if pop_f[0] < fx:
                fx = pop_f[0]
                x = pop[0].copy()

            # Update mean
            mu = pop_size // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            mean_old = mean.copy()
            mean = np.sum(w[:, None] * pop[:mu], axis=0)

            # Update covariance
            y = (mean - mean_old) / sigma
            pc = (1 - cs) * pc + cs * np.sqrt(cc * (2 - cc) / mueff) * y
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n
            ps = (1 - ds) * ps + float(hsig) * np.sqrt(ds * (2 - ds) + cc * (2 - cc) / mueff) * y

            C = ((1 - cmu * (hsig - 1) * cc * (2 - cc)) * C +
                 np.outer(ps, ps) * (1 - hsig) +
                 np.sum(w[:, None, None] * np.outer(pop[:mu] - mean_old, pop[:mu] - mean_old), axis=0) / sigma ** 2 * cmu)

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Regularize
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += (1e-10 - np.min(eigvals)) * np.eye(dim)

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update B and D
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                D = np.sqrt(np.diag(C))
                B = np.eye(dim)

        # Phase 3: Fine (1+1)-ES polish
        sigma = 0.05
        for i in range(min(100, budget - calls[0])):
            if not time_ok():
                break
            step = sigma * rng.standard_normal(dim)
            x_trial = np.clip(x + step, -5, 5)
            fx_trial = call_func(x_trial)
            if fx_trial < fx:
                x = x_trial
                fx = fx_trial
                sigma *= 1.05
            else:
                sigma *= 0.9

        return x, fx
# EVOLVE-BLOCK-END
