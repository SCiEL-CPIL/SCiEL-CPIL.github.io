import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)

        # CMA-ES parameters tuned for 5D sphere with budget 500
        # Use a larger population for better exploration early on
        lam = 12  # larger than default for better exploration
        mu = lam // 2  # number of recombination weights

        # Initial sigma: domain is [-5,5], use moderate initial step
        sigma = 2.0

        # Initial mean: center of the box
        mean = np.zeros(dim)
        fx_mean = func(mean)
        best_f = fx_mean
        best_x = mean.copy()
        calls_made = 1

        # CMA-ES weights (log-weights)
        weights = np.log(np.arange(lam, 0, -1)) - np.log(np.arange(lam, 0, -1)).sum()
        weights = weights[:mu] / weights[:mu].sum()

        # Effective population size
        mueff = 1.0 / np.sum(weights**2)

        # Learning rates (gentler adaptation as per cell findings)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))

        # Path variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Covariance matrix
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        C_inv = np.eye(dim)

        # Helper: eigendecomposition of C
        def update_covariance():
            nonlocal C, C_inv, C_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        update_covariance()

        # Track stagnation
        no_improve_count = 0
        stagnation_threshold = 30  # generations without improvement before restart

        for gen in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate lambda candidates
            z_samples = rng.standard_normal((lam, dim))
            candidates = []
            for j in range(lam):
                y = C_inv_sqrt @ z_samples[j]
                x_j = mean + sigma * y
                x_j = np.clip(x_j, -5, 5)
                candidates.append(x_j)

            # Evaluate all candidates
            fvals = []
            gen_best_f = best_f
            for x_j in candidates:
                if time.time() > deadline:
                    fvals.append(best_f)
                    continue
                f_j = func(x_j)
                calls_made += 1
                fvals.append(f_j)

                if f_j < gen_best_f:
                    gen_best_f = f_j

            fvals = np.array(fvals)

            # Check for improvement
            if gen_best_f < best_f - 1e-10:
                best_f = gen_best_f
                # Find the best x
                best_idx = np.argmin(fvals)
                best_x = candidates[best_idx].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Sort by fitness
            order = np.argsort(fvals)
            x_selected = np.array([candidates[i] for i in order[:mu]])

            # Update mean
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * x_selected, axis=0)

            # Success check
            if np.linalg.norm(mean - mean_old) > 1e-10:
                # Update path p_sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_inv_sqrt @ (mean - mean_old))
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))

                # Update path p_c
                p_c = (1 - cs) * p_c + hsig * np.sqrt(cs * (2 - cs) * mueff) / sigma * (mean - mean_old)

                # Rank-one and rank-mu updates
                delta_hsig = (1.0 if hsig else 0.0)
                C = (1 + c1 * delta_hsig + cmu) * C - c1 * np.outer(p_c, p_c)
                C -= cmu * np.sum(weights[:, None, None] * np.outer((C_inv_sqrt @ (x_selected - mean_old).T).T, (C_inv_sqrt @ (x_selected - mean_old).T)), axis=0)

                # Ensure positive definiteness
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                # Adapt sigma
                sigma = sigma * np.exp(cs / (dim + 3) * (np.linalg.norm(p_sigma) / np.sqrt((1 - (1 - cs)**(gen + 1))) - 1))

                # Clip sigma
                sigma = np.clip(sigma, 1e-6, 10.0)

            # Stagnation restart
            if no_improve_count >= stagnation_threshold:
                # Restart from best with larger sigma
                mean = best_x.copy()
                sigma = max(sigma * 2.0, 1.0)
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_count = 0

            if calls_made >= budget:
                break

        return best_x, best_f
