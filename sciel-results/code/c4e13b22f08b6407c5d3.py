"""Optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: global exploration with restarts, then CMA-ES-like local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Phase 1: Global exploration with restarts
        # Use a population-based approach with stagnation restarts
        global_budget = int(budget * 0.4)
        local_budget = budget - global_budget

        best_x = None
        best_f = np.inf
        best_calls = 0

        # Track calls globally
        call_count = [0]

        def eval_func(x):
            call_count[0] += 1
            return func(x)

        # Global phase: (mu+lambda) with restarts
        mu = 4
        lam = 8
        sigma = 3.0
        stagnation_limit = 50
        no_improve = 0
        phase_calls = 0

        # Initial population
        pop = [rng.uniform(-5, 5, n) for _ in range(lam)]
        pop_f = [eval_func(p) for p in pop]
        phase_calls += lam

        for _ in range(1, 10000):
            if phase_calls + lam > global_budget:
                break
            if time.time() > deadline:
                break

            # Sort and select parents
            idx = np.argsort(pop_f)
            parents = [pop[i] for i in idx[:mu]]
            best_in_gen = pop_f[idx[0]]

            if best_in_gen < best_f:
                best_f = best_in_gen
                best_x = np.copy(pop[idx[0]])
                no_improve = 0
            else:
                no_improve += 1

            # Generate offspring
            new_pop = []
            for i in range(lam):
                parent = parents[i % mu]
                child = parent + sigma * rng.standard_normal(n)
                child = np.clip(child, -5, 5)
                new_pop.append(child)

            new_f = [eval_func(p) for p in new_pop]
            phase_calls += lam

            # Combine and select
            combined = list(zip(pop, pop_f)) + list(zip(new_pop, new_f))
            combined.sort(key=lambda x: x[1])
            pop = [c[0] for c in combined[:lam]]
            pop_f = [c[1] for c in combined[:lam]]

            # Adapt sigma
            if no_improve < stagnation_limit:
                sigma *= 1.05
            else:
                sigma *= 0.9

            # Restart if stuck
            if no_improve >= stagnation_limit:
                # Try a random restart
                x_new = rng.uniform(-5, 5, n)
                f_new = eval_func(x_new)
                phase_calls += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = np.copy(x_new)
                    no_improve = 0
                else:
                    # Keep searching locally around best
                    for _ in range(5):
                        x_try = best_x + 1.0 * rng.standard_normal(n)
                        x_try = np.clip(x_try, -5, 5)
                        f_try = eval_func(x_try)
                        phase_calls += 1
                        if f_try < best_f:
                            best_f = f_try
                            best_x = np.copy(x_try)

            if best_x is None:
                best_x = np.copy(pop[0])
                best_f = pop_f[0]

        # Phase 2: Local refinement with CMA-ES-like strategy
        if best_x is None:
            best_x = rng.uniform(-5, 5, n)
            best_f = eval_func(best_x)

        # CMA-ES parameters
        c1 = 1 / (n + 4)
        cmu = min(1 - c1, 2 * (mu - 2 + 0.5) / ((n + 2) ** 2 + 1))
        c_sigma = (mu + 2) / (n + mu + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initialize CMA-ES
        x_mean = np.copy(best_x)
        f_mean = best_f
        sigma_cma = 0.5
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        hs = chi_n * (1 + c_sigma / (2 * d_sigma))

        # Compute inverse square root of C
        def inv_sqrt(C_mat):
            w, V = np.linalg.eigh(C_mat)
            w = np.clip(w, 1e-10, None)
            return V @ np.diag(1 / np.sqrt(w)) @ V.T

        C_inv_sqrt = inv_sqrt(C)

        local_calls = 0
        max_local_calls = local_budget

        for _ in range(1, 50000):
            if local_calls + lam > max_local_calls:
                break
            if time.time() > deadline:
                break

            # Generate lambda samples
            z = rng.standard_normal((lam, n))
            # Standard normalization
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            # Transform to C-normalized
            y = (z @ B) / D
            # Apply inverse sqrt of C
            y = y @ C_inv_sqrt
            # Apply sigma
            x_samples = x_mean[None, :] + sigma_cma * y

            # Clip to bounds
            x_samples = np.clip(x_samples, -5, 5)

            # Evaluate
            f_samples = np.array([eval_func(x) for x in x_samples])
            local_calls += lam

            # Sort
            idx = np.argsort(f_samples)
            # Weighted recombination
            w = np.log(lam / 2) - np.log(np.arange(1, mu + 1))
            w /= w.sum()

            # Update mean
            x_new_mean = np.sum(w[:, None] * x_samples[idx[:mu]], axis=0)
            f_new_mean = f_samples[idx[0]]

            # Update step size control
            y_sigma = (x_new_mean - x_mean) / sigma_cma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma_cma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / hs - 1))
            sigma_cma = np.clip(sigma_cma, 1e-10, 50)

            # Update covariance matrix
            y_c = (x_new_mean - x_mean) / sigma_cma
            p_c = (1 - c1) * p_c + np.sqrt(c1 * (2 - c1) * mu_eff) * y_c

            delta_h = (1 + c1) * np.dot(p_c, p_c) / n - (1 - c1)
            C = ((1 + c1 * delta_h - c1 - cmu) * C
                 + c1 * np.outer(p_c, p_c)
                 + cmu * np.sum(w[:, None, None] * np.outer(y_samples[idx[:mu]], y_samples[idx[:mu]]), axis=0))

            # Ensure symmetry
            C = (C + C.T) / 2

            # Recompute B, D
            w_eig, V = np.linalg.eigh(C)
            w_eig = np.clip(w_eig, 1e-10, None)
            B = V
            D = np.sqrt(w_eig)
            C_inv_sqrt = inv_sqrt(C)

            # Update best
            if f_new_mean < best_f:
                best_f = f_new_mean
                best_x = np.copy(x_new_mean)
                f_mean = f_new_mean

            # Update mean
            x_mean = x_new_mean

            # Check convergence
            if sigma_cma < 1e-8:
                break

        return best_x, best_f


# Helper for mu_eff
def _mu_eff(mu):
    return mu * (mu - 1) / (mu + 1)


# Monkey-patch or inline mu_eff
# Actually, let's just compute it inline in the loop
# Redefine with inline mu_eff
class Optimizer:
    """Multi-phase optimizer: global exploration with restarts, then CMA-ES-like local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        global_budget = int(budget * 0.4)
        local_budget = budget - global_budget

        best_x = None
        best_f = np.inf
        call_count = [0]

        def eval_func(x):
            call_count[0] += 1
            return func(x)

        # Phase 1: Global exploration
        mu = 4
        lam = 8
        sigma = 3.0
        stagnation_limit = 50
        no_improve = 0
        phase_calls = 0

        pop = [rng.uniform(-5, 5, n) for _ in range(lam)]
        pop_f = [eval_func(p) for p in pop]
        phase_calls += lam

        for _ in range(1, 10000):
            if phase_calls + lam > global_budget:
                break
            if time.time() > deadline:
                break

            idx = np.argsort(pop_f)
            parents = [pop[i] for i in idx[:mu]]
            best_in_gen = pop_f[idx[0]]

            if best_in_gen < best_f:
                best_f = best_in_gen
                best_x = np.copy(pop[idx[0]])
                no_improve = 0
            else:
                no_improve += 1

            new_pop = []
            for i in range(lam):
                parent = parents[i % mu]
                child = parent + sigma * rng.standard_normal(n)
                child = np.clip(child, -5, 5)
                new_pop.append(child)

            new_f = [eval_func(p) for p in new_pop]
            phase_calls += lam

            combined = list(zip(pop, pop_f)) + list(zip(new_pop, new_f))
            combined.sort(key=lambda x: x[1])
            pop = [c[0] for c in combined[:lam]]
            pop_f = [c[1] for c in combined[:lam]]

            if no_improve < stagnation_limit:
                sigma *= 1.05
            else:
                sigma *= 0.9

            if no_improve >= stagnation_limit:
                x_new = rng.uniform(-5, 5, n)
                f_new = eval_func(x_new)
                phase_calls += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = np.copy(x_new)
                    no_improve = 0
                else:
                    for _ in range(5):
                        x_try = best_x + 1.0 * rng.standard_normal(n)
                        x_try = np.clip(x_try, -5, 5)
                        f_try = eval_func(x_try)
                        phase_calls += 1
                        if f_try < best_f:
                            best_f = f_try
                            best_x = np.copy(x_try)

            if best_x is None:
                best_x = np.copy(pop[0])
                best_f = pop_f[0]

        # Phase 2: CMA-ES local refinement
        if best_x is None:
            best_x = rng.uniform(-5, 5, n)
            best_f = eval_func(best_x)

        x_mean = np.copy(best_x)
        sigma_cma = 0.5
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        c1 = 1 / (n + 4)
        mu_eff = mu * (mu - 1) / (mu + 1)
        cmu = min(1 - c1, 2 * (mu - 2 + 0.5) / ((n + 2) ** 2 + 1))
        c_sigma = (mu + 2) / (n + mu + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + c_sigma
        hs = chi_n * (1 + c_sigma / (2 * d_sigma))

        def inv_sqrt(C_mat):
            w, V = np.linalg.eigh(C_mat)
            w = np.clip(w, 1e-10, None)
            return V @ np.diag(1 / np.sqrt(w)) @ V.T

        C_inv_sqrt = inv_sqrt(C)

        local_calls = 0
        max_local_calls = local_budget

        for _ in range(1, 50000):
            if local_calls + lam > max_local_calls:
                break
            if time.time() > deadline:
                break

            z = rng.standard_normal((lam, n))
            norms = np.linalg.norm(z, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-10, None)
            z = z / norms
            y = (z @ B) / D
            y = y @ C_inv_sqrt
            x_samples = x_mean[None, :] + sigma_cma * y
            x_samples = np.clip(x_samples, -5, 5)

            f_samples = np.array([eval_func(x) for x in x_samples])
            local_calls += lam

            idx = np.argsort(f_samples)
            w = np.log(lam / 2) - np.log(np.arange(1, mu + 1))
            w /= w.sum()

            x_new_mean = np.sum(w[:, None] * x_samples[idx[:mu]], axis=0)
            f_new_mean = f_samples[idx[0]]

            y_sigma = (x_new_mean - x_mean) / sigma_cma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma_cma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / hs - 1))
            sigma_cma = np.clip(sigma_cma, 1e-10, 50)

            y_c = (x_new_mean - x_mean) / sigma_cma
            p_c = (1 - c1) * p_c + np.sqrt(c1 * (2 - c1) * mu_eff) * y_c

            delta_h = (1 + c1) * np.dot(p_c, p_c) / n - (1 - c1)
            C = ((1 + c1 * delta_h - c1 - cmu) * C
                 + c1 * np.outer(p_c, p_c)
                 + cmu * np.sum(w[:, None, None] * np.outer(y[idx[:mu]], y[idx[:mu]]), axis=0))

            C = (C + C.T) / 2

            w_eig, V = np.linalg.eigh(C)
            w_eig = np.clip(w_eig, 1e-10, None)
            B = V
            D = np.sqrt(w_eig)
            C_inv_sqrt = inv_sqrt(C)

            if f_new_mean < best_f:
                best_f = f_new_mean
                best_x = np.copy(x_new_mean)

            x_mean = x_new_mean

            if sigma_cma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
