# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Adaptive Hyperparameter Annealing (AHA) with a single population of 10 candidates.
    
    Single continuous search where mutation step size and covariance adaptation
    rates are annealed based on the current improvement rate (success rate)
    rather than time or iteration count.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        pop_size = 10

        rng = np.random.default_rng(12345)

        # --- Initialization: diverse spread across the box ---
        # Start with a well-spread initial population to cover the search space
        # Use Latin-hypercube-like stratification for better coverage
        x_init = np.zeros((pop_size, dim))
        for d in range(dim):
            # Stratified random per dimension
            strata = np.linspace(-5, 5, pop_size + 1)
            x_init[:, d] = rng.uniform(strata[:-1], strata[1:])
        
        population = []
        f_values = []
        calls_made = 0
        f_best = np.inf
        x_best = None

        for i in range(pop_size):
            x = x_init[i]
            f = func(x)
            calls_made += 1
            population.append(x)
            f_values.append(f)
            if f < f_best:
                f_best = f
                x_best = x.copy()

        # Sort initial population by fitness
        idx = np.argsort(f_values)
        population = [population[i] for i in idx]
        f_values = [f_values[i] for i in idx]

        # --- CMA-ES core parameters ---
        mu = pop_size // 2  # 5 parents
        # Logarithmic weights
        w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = w_raw / w_raw.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Initial step size: scaled to explore the box
        sigma = 2.0
        min_sigma = 1e-10
        max_sigma = 15.0

        # Covariance matrix and its square root
        C = np.eye(dim)
        Csqr = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Standard CMA-ES learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff) / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # --- AHA: Adaptive annealing state ---
        # Track improvement over a sliding window
        success_window = []
        window_size = 15
        # Track consecutive generations without improvement for stagnation detection
        stagnation_count = 0

        # Main loop
        while calls_made < budget:
            # Compute weighted mean of top mu
            m = np.zeros(dim)
            for i in range(mu):
                m += weights[i] * population[i]

            # Generate offspring
            offspring = []
            offspring_f = []
            improved_this_gen = False

            for i in range(pop_size):
                if calls_made >= budget:
                    break
                z = rng.standard_normal(dim)
                x_new = m + sigma * (Csqr @ z)
                x_new = np.clip(x_new, -5, 5)
                offspring.append(x_new)

                f_val = func(x_new)
                calls_made += 1
                offspring_f.append(f_val)

                if f_val < f_best:
                    f_best = f_val
                    x_best = x_new.copy()
                    improved_this_gen = True

            # Pad offspring_f if budget ran out mid-generation
            while len(offspring_f) < pop_size:
                offspring_f.append(f_best)

            # Elitist selection: keep best pop_size from parents + offspring
            combined = population + offspring
            combined_f = f_values + offspring_f
            sel_idx = np.argsort(combined_f)[:pop_size]
            population = [combined[i] for i in sel_idx]
            f_values = [combined_f[i] for i in sel_idx]

            # --- AHA: Update success rate and stagnation ---
            success_window.append(1.0 if improved_this_gen else 0.0)
            if len(success_window) > window_size:
                success_window = success_window[-window_size:]

            success_rate = np.mean(success_window)

            if improved_this_gen:
                stagnation_count = 0
            else:
                stagnation_count += 1

            # --- AHA: Adaptive sigma modulation ---
            # Standard CMA-ES sigma update
            y_sigma = np.zeros(dim)
            for i in range(mu):
                y_sigma += weights[i] * (population[i] - m) / sigma

            norm_y_sigma = np.linalg.norm(y_sigma)
            h_sigma = float(norm_y_sigma / chi_n > 1.4 + 2 / (dim + 1))

            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * h_sigma * (y_sigma / chi_n)

            # Standard CMA-ES sigma adaptation
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma * (norm_y_sigma / chi_n - 1))

            # AHA modulation: adjust sigma based on success rate and stagnation
            # High success rate -> allow larger steps (exploration)
            # Low success rate + stagnation -> shrink steps (exploitation)
            # Use a smooth modulation that's gentle
            if success_rate > 0.3:
                # Improving: allow moderate expansion
                sigma_mod = 1.0 + 0.15 * (success_rate - 0.3) / 0.7
            elif success_rate < 0.1 and stagnation_count > 5:
                # Stagnant: contract to exploit local structure
                sigma_mod = 0.85
            else:
                sigma_mod = 1.0

            sigma *= sigma_mod
            sigma = np.clip(sigma, min_sigma, max_sigma)

            # --- AHA: Adaptive covariance rates ---
            # When improving, adapt covariance faster to follow the landscape
            # When stagnant, slow down to avoid oscillation
            # Smooth annealing based on success rate
            if success_rate > 0.3:
                rate_scale = 1.0 + 0.3 * (success_rate - 0.3) / 0.7
            elif success_rate < 0.1 and stagnation_count > 5:
                rate_scale = 0.5
            else:
                rate_scale = 0.8 + 0.2 * (success_rate / 0.3)

            c_c_eff = c_c * rate_scale
            c_1_eff = c_1 * rate_scale
            c_mu_eff = c_mu * rate_scale

            # --- Update covariance matrix ---
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (population[i] - m)

            y_w_norm = y_w / sigma
            p_c = (1 - c_c_eff) * p_c + c_c_eff * h_sigma * y_w_norm

            # Rank-1 update
            C = (1 + c_1_eff * (c_mu_eff * mu_eff - 1)) * C + c_1_eff * np.outer(p_c, p_c)

            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (population[i] - m) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            C += c_mu_eff * rank_mu

            # Symmetrize and ensure positive definiteness
            C = (C + C.T) / 2.0

            # Eigen-decompose for Csqr and regularization
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, 1e12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            Csqr = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        return x_best, f_best
# EVOLVE-BLOCK-END
