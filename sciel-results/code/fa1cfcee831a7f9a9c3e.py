# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive population size for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        n = self.dim
        budget = self.budget

        state = {
            'calls': 0,
            'best_f': np.inf,
            'best_x': np.zeros(n)
        }

        def wrapped_func(x):
            if state['calls'] >= budget or time.time() > deadline:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            state['calls'] += 1
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        x_mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(n)

        p_c = np.zeros(n)
        p_sigma = np.zeros(n)

        # Standard CMA-ES adaptation rates
        c_sigma = (n + 2) / (n + 5 + 2 * n / (n + 1))
        d_sigma = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + c_sigma
        c_c = (4 + n / n) / (4 + n + n)
        c_1 = 2 / ((n + 1.3) ** 2 + n)

        # Adaptive population size parameters
        pop_size = 10
        min_pop_size = 4
        max_gens = 100

        gen = 0
        no_improve_count = 0
        best_f_prev = np.inf

        while state['calls'] < budget - 10 and gen < max_gens and time.time() < deadline:
            # Calculate mu and weights for current pop_size
            if pop_size < 4:
                pop_size = min_pop_size
            mu = max(2, pop_size // 2)
            
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Calculate c_mu based on current mu_eff
            c_mu = min(1 - c_1, 2 * ((n + 1.) / (n + 4.)) ** 2 - 1)

            offspring = []
            for i in range(pop_size):
                if state['calls'] >= budget - 10 or time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                x_sample = x_mean + sigma * (L @ z)
                x_sample = np.clip(x_sample, -5, 5)
                fx = wrapped_func(x_sample)
                offspring.append((x_sample, fx))
            
            if len(offspring) < 2:
                break
                
            offspring.sort(key=lambda t: t[1])
            
            # Check for improvement
            if offspring[0][1] < best_f_prev:
                no_improve_count = 0
                best_f_prev = offspring[0][1]
            else:
                no_improve_count += 1
            
            if offspring[0][1] < state['best_f']:
                state['best_f'] = offspring[0][1]
                state['best_x'] = offspring[0][0].copy()
            
            # Use top mu for update
            parents = [offspring[i][0] for i in range(min(mu, len(offspring)))]
            if len(parents) < 2:
                break
                
            # Recalculate weights if we have fewer parents than mu
            if len(parents) < mu:
                mu = len(parents)
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mu_eff = 1.0 / np.sum(weights ** 2)
                c_mu = min(1 - c_1, 2 * ((n + 1.) / (n + 4.)) ** 2 - 1)

            new_mean = np.sum([weights[i] * parents[i] for i in range(mu)], axis=0)
            
            y = (new_mean - x_mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            
            rank_one = np.outer(y, y)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                y_i = (parents[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            # Update covariance matrix
            C = (1 + c_1 * (c_mu - c_1) / c_1 - c_1) * C + c_1 * (rank_one + rank_mu)
            
            # Ensure positive semi-definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
            
            # Update sigma
            h_sigma = 1.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L.T @ (L @ y))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            x_mean = new_mean
            gen += 1
            
            # Adaptive population size: shrink if stagnating
            if no_improve_count > 5 and pop_size > min_pop_size:
                pop_size = max(min_pop_size, pop_size - 1)
                no_improve_count = 0 # Reset counter after shrinking
            
            # Stop if sigma is too small
            if sigma < 1e-4:
                break

        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END
