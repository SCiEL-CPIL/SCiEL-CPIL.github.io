"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f13 (Sharp Ridge) with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        calls = [0]

        def safe_call(x):
            if calls[0] >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            try:
                val = float(func(x))
                calls[0] += 1
                return val
            except:
                calls[0] += 1
                return float('inf')

        best_x = None
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def cma_es(x0, max_iter=200):
            nonlocal best_x, best_f, calls, budget
            if calls[0] >= budget or time.time() > deadline:
                return x0, best_f

            dim = self.dim
            lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))

            # CMA-ES parameters
            cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + (4 + dim / (dim + 4)))
            cmu = min(1 - cc, 2 * (dim - 2 + 1. / dim) / ((dim + 2.) ** 2 + 1))
            sigma = 1.0
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            c_sigma = 1.0
            B = np.eye(dim)
            D = np.ones(dim)
            C = np.eye(dim)

            # Initial population
            pop = []
            fvals = []
            for i in range(lam):
                if calls[0] >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                xi = x0 + sigma * (B @ (D * z))
                xi = np.clip(xi, -5.0, 5.0)
                fi = safe_call(xi)
                pop.append(xi)
                fvals.append(fi)
                update_best(xi, fi)

            if len(pop) < 2:
                return x0, best_f

            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1. / np.sum(weights ** 2)

            for it in range(max_iter):
                if calls[0] >= budget or time.time() > deadline:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                pop = [pop[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if abs(fvals[0] - fvals[-1]) < 1e-12 * max(1.0, abs(fvals[0])):
                    break

                # Weighted mean
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += weights[i] * pop[i]

                # Update evolution paths
                y = (x_new - x0) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (it + 1))) / \
                       np.sqrt(dim) * (1 + cmu / 2) <= 1.4 + 2 / (dim + 1)

                pc = (1 - cc) * pc + hsig * cc * y
                x0 = x_new

                # Update covariance matrix
                for i in range(mu):
                    zi = (pop[i] - x_new) / sigma
                    C += cmu / (sigma ** 2) * np.outer(zi, zi)
                delta_sigma = (np.linalg.norm(pc) ** 2 / dim - 1) * cc * (2 - cc)
                C += (2 * cmu * delta_sigma) * np.outer(pc, pc)

                # Update step size
                sigma *= np.exp((c_sigma / 5) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

                # Eigenvalue decomposition for sampling
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))

                # Generate new population
                new_pop = []
                new_fvals = []
                for i in range(lam):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    xi = x0 + sigma * (B @ (D * z))
                    xi = np.clip(xi, -5.0, 5.0)
                    fi = safe_call(xi)
                    new_pop.append(xi)
                    new_fvals.append(fi)
                    update_best(xi, fi)

                if len(new_pop) < 2:
                    break

                pop = new_pop
                fvals = new_fvals

            # Return best point
            order = np.argsort(fvals)
            return pop[order[0]], fvals[order[0]]

        # Main loop
        # 1. Start with a random point
        x_current = rng.uniform(-4, 4, dim)
        f_current = safe_call(x_current)
        update_best(x_current, f_current)

        # 2. Run CMA-ES
        if calls[0] < budget - 5:
            x_polished, f_polished = cma_es(x_current, max_iter=150)
            update_best(x_polished, f_polished)

        # 3. If budget remains, try a few more random starts if we haven't converged well
        if calls[0] < budget - 20:
            for _ in range(3):
                if calls[0] >= budget - 10 or time.time() > deadline:
                    break
                x_new = rng.uniform(-4, 4, dim)
                f_new = safe_call(x_new)
                update_best(x_new, f_new)
                if f_new < best_f + 1e-6:
                    if calls[0] < budget - 10:
                        x_polished, f_polished = cma_es(x_new, max_iter=50)
                        update_best(x_polished, f_polished)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
