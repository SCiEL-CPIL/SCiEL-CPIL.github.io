"""Optimized BBOB f15 (Rastrigin rotated) solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(μ+λ)-ES with adaptive step-size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0

        def timed_call(x):
            nonlocal calls_made
            val = func(x)
            calls_made += 1
            return val

        # Try multiple restarts with different strategies
        # Strategy: (μ+λ)-ES with μ=4, λ=12, 1/5 success rule for sigma
        # Each restart gets a fraction of the budget

        rng = np.random.default_rng(12345)

        # We'll do a single long run but with periodic resets if stuck
        # Actually, let's do: first a coarse search, then refine

        # Phase 1: Coarse exploration with larger sigma
        # Phase 2: Fine search

        # Let's use a (4+12)-ES
        mu = 4
        lam = 12
        sigma = 3.0
        alpha_mu = np.full(mu, 1.0 / mu)  # weights
        mu_eff = 1.0 / np.sum(alpha_mu ** 2)
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / (self.dim ** 1.8 + 10)
        c_mu = min(1 - c_1, (mu_eff - 2 + 2 / mu_eff) / (self.dim + 2))
        chi = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

        # Initial population
        x_mean = rng.uniform(-4, 4, self.dim)
        pop = x_mean + sigma * rng.standard_normal((lam, self.dim))
        pop = np.clip(pop, -5, 5)

        # Evaluate initial population
        fitness = np.array([timed_call(pop[i]) for i in range(lam)])

        # Track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        # Sort by fitness
        order = np.argsort(fitness)
        pop_sorted = pop[order]
        fitness_sorted = fitness[order]

        # Update mean
        x_mean = np.sum(alpha_mu[:, None] * pop_sorted[:mu], axis=0)

        # Initialize evolution paths
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)

        # Success counter for 1/5 rule (simplified: use path-based adaptation)
        for gen in range(self.budget - lam):
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break

            # Generate offspring
            new_pop = x_mean + sigma * rng.standard_normal((lam, self.dim))
            new_pop = np.clip(new_pop, -5, 5)

            # Evaluate
            new_fitness = np.array([timed_call(new_pop[i]) for i in range(lam)])

            # Combine and sort
            all_pop = np.vstack([pop_sorted[:mu], new_pop])
            all_fit = np.concatenate([fitness_sorted[:mu], new_fitness])
            all_order = np.argsort(all_fit)
            all_pop_sorted = all_pop[all_order]
            all_fit_sorted = all_fit[all_order]

            # Update best
            if all_fit_sorted[0] < best_f:
                best_f = all_fit_sorted[0]
                best_x = all_pop_sorted[0].copy()

            # New mean
            new_x_mean = np.sum(alpha_mu[:, None] * all_pop_sorted[:mu], axis=0)

            # Update evolution paths
            diff = new_x_mean - x_mean
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * diff
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * diff

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean with covariance matrix adaptation (simplified: just use new mean)
            x_mean = new_x_mean

            # Update population
            pop_sorted = all_pop_sorted[:mu]
            fitness_sorted = all_fit_sorted[:mu]

        return best_x, best_f
# EVOLVE-BLOCK-END
