import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        def eval_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or not time_ok():
                return np.inf
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        pop_size = 10  # lambda
        sigma = 2.0
        c1 = (4 + np.log(dim)) / (dim + 4 + 3 * np.log(dim))
        cmu = min(0.2, (1 - c1) / 2)
        cs = (dim + 1.3) / (dim + 5.5)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial center: start at origin (minimizer is in [-4,4]^5)
        mean = np.zeros(dim)
        first_eval = eval_func(mean)
        if first_eval < best_f:
            best_f = first_eval
            best_x = mean.copy()

        # Initialize population
        pop = mean + sigma * rng.standard_normal((pop_size, dim))
        pop = np.clip(pop, -5.0, 5.0)
        fitness = np.array([eval_func(x) for x in pop])

        # Sort by fitness
        order = np.argsort(fitness)
        pop = pop[order]
        fitness = fitness[order]

        # CMA-ES state
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        step_size = sigma
        damps = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)

        mu = pop_size // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 2) + c_sigma) / ((dim + 2.1) ** 2 + mueff + c_sigma)
        c_mu = min(0.5, (mueff - 2 + 2 / dim) / ((dim + 2) ** 2 + mueff))

        while calls < budget - pop_size and time_ok():
            # Compute new mean
            new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

            # Update evolution path
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (new_mean - mean) / step_size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (new_mean - mean) / step_size

            # Update covariance matrix
            delta_sigma = (np.linalg.norm(ps) / chi_n - 1) * c_sigma * damps
            step_size *= np.exp(delta_sigma)

            # Rank-one and rank-mu updates
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (delta_sigma * (2 - delta_sigma)) * np.eye(dim))
            C += c_mu * np.sum(weights[i] * np.outer(pop[i] - mean, pop[i] - mean) for i in range(mu))

            # Ensure positive semi-definite
            C = (C + C.T) / 2

            # Update mean
            mean = new_mean
            if mean[np.argmin(fitness)] < best_f:
                pass  # already handled in eval_func

            # Generate new population
            # Compute eigenvalues for sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.sqrt(eigvals)

            new_pop = mean[None, :] + step_size * (eigvecs @ (D[:, None] * rng.standard_normal((dim, pop_size))))
            new_pop = np.clip(new_pop, -5.0, 5.0)

            # Evaluate new population
            new_fitness = np.array([eval_func(x) for x in new_pop])

            # Sort
            order = np.argsort(new_fitness)
            pop = new_pop[order]
            fitness = new_fitness[order]

            # Update best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = pop[0].copy()

        return best_x, best_f
