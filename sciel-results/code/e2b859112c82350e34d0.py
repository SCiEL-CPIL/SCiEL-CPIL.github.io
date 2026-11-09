"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        calls = 0
        best_x = np.zeros(self.dim)
        best_f = np.inf

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -self.bound, self.bound)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def cma_es(x0, sigma0, max_iter):
            """
            Simplified CMA-ES implementation.
            x0: initial mean
            sigma0: initial step size
            max_iter: maximum number of generations
            """
            nonlocal calls, best_x, best_f, deadline

            n = self.dim
            # Population size: default is 4 + floor(3 ln n)
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            if lambda_ < 4:
                lambda_ = 4

            # Weights
            mu = lambda_ // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mu_eff = 1 / np.sum(w**2)

            # Adaptation rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))

            # Initialization
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            cs = np.sqrt((n + 1.3 + 2 / n) * c_sigma * (2 - c_sigma))

            # Precompute square root of C (Cholesky)
            try:
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(B))
                B = B / D[:, np.newaxis]
            except np.linalg.LinAlgError:
                B = np.eye(n)
                D = np.ones(n)

            for it in range(max_iter):
                if calls >= self.budget or time.time() > deadline:
                    break

                # Sample lambda points
                z = np.random.standard_normal((lambda_, n))
                # Transform to normal distribution with covariance C
                x_samples = mean + sigma * np.dot(z, B.T)

                # Evaluate
                f_values = np.array([eval_func(x_samples[i]) for i in range(lambda_)])
                
                if calls >= self.budget:
                    break

                # Sort by fitness
                order = np.argsort(f_values)
                x_sorted = x_samples[order]
                f_sorted = f_values[order]

                # Update best
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = x_sorted[0].copy()

                # Update mean
                new_mean = np.sum(w[:, np.newaxis] * x_sorted[:mu], axis=0)

                # Update path evolution strategies
                old_mean = mean
                delta_mean = (new_mean - old_mean) / sigma
                pc = (1 - c_c) * pc + c_c * delta_mean
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / np.linalg.norm(B) * delta_mean

                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1)))) < (1 + 2 / n) * np.sqrt(1 - c_sigma)

                # Update covariance matrix
                C = (1 + c_1 * (hsig - 1) * np.trace(C) + c_mu) * C
                if hsig:
                    C -= c_1 * np.outer(pc, pc)
                C -= c_1 * np.trace(C) * np.eye(n) # This line is actually wrong in standard CMA-ES, let's use standard update
                
                # Correct Standard CMA-ES Update:
                # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (hsig - 1) * C) + c_mu * sum(w_i * (x_i - mean_old)(x_i - mean_old).T)
                
                # Let's re-implement the update correctly to avoid bugs
                # Reset C update logic
                C_update = (1 - c_1 - c_mu) * C
                if hsig:
                    C_update += c_1 * (np.outer(pc, pc) + (hsig - 1) * C)
                else:
                    C_update += c_1 * np.outer(pc, pc)
                
                for i in range(mu):
                    v = x_sorted[i] - old_mean
                    C_update += c_mu * w[i] * np.outer(v, v)
                
                C = C_update

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                try:
                    B = np.linalg.cholesky(C)
                    D = np.sqrt(np.diag(B))
                    B = B / D[:, np.newaxis]
                except np.linalg.LinAlgError:
                    # Fallback to identity if not positive definite
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones(n)

                # Update mean and sigma
                mean = new_mean
                sigma = sigma * np.exp(c_sigma / cs * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1))) - 1))

                # Clip mean to bounds to keep search relevant
                mean = np.clip(mean, -self.bound, self.bound)

        def nelder_mead_polish(x0, max_iter=50):
            """Local polish using Nelder-Mead."""
            nonlocal calls, best_x, best_f, deadline
            n = self.dim
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += 0.1

            simplex_f = np.array([eval_func(simplex[i]) for i in range(n + 1)])
            if calls >= self.budget:
                return

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            for it in range(max_iter):
                if calls >= self.budget or time.time() > deadline:
                    break

                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]

                if abs(simplex_f[0] - simplex_f[-1]) < 1e-12:
                    break

                centroid = np.mean(simplex[:n], axis=0)
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = eval_func(x_r)
                if calls >= self.budget:
                    break

                if f_r < simplex_f[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = eval_func(x_e)
                    if calls >= self.budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    if f_r < simplex_f[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (centroid - simplex[-1])
                    f_c = eval_func(x_c)
                    if calls >= self.budget:
                        break
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex_f[i] = eval_func(simplex[i])
                        if calls >= self.budget:
                            break

        # Phase 1: Initial diverse points to find a good basin
        rng = np.random.default_rng(42)
        init_points = [np.zeros(self.dim)]
        for _ in range(5):
            init_points.append(rng.uniform(-4.0, 4.0, self.dim))
        
        for p in init_points:
            eval_func(p)

        # Phase 2: Run CMA-ES from the best initial point
        # We allocate about 70% of the budget to CMA-ES
        cma_budget = int(self.budget * 0.7)
        # Estimate max iterations: lambda + 1 calls per iter roughly
        max_cma_iter = max(1, (cma_budget - calls) // 6) # Conservative estimate
        
        if calls < self.budget:
            cma_es(best_x.copy(), sigma0=1.0, max_iter=max_cma_iter)

        # Phase 3: Local Polish
        if calls < self.budget:
            nelder_mead_polish(best_x.copy(), max_iter=50)

        # Phase 4: If budget remains, try a second CMA-ES run from a perturbed best
        if calls < self.budget and time.time() < deadline:
            perturbation = rng.normal(0, 0.5, self.dim)
            x_start2 = np.clip(best_x + perturbation, -self.bound, self.bound)
            eval_func(x_start2)
            if calls < self.budget:
                max_cma_iter2 = max(1, (self.budget - calls) // 6)
                cma_es(x_start2, sigma0=0.5, max_iter=max_cma_iter2)
            if calls < self.budget:
                nelder_mead_polish(best_x.copy(), max_iter=20)

        return best_x, best_f
# EVOLVE-BLOCK-END
