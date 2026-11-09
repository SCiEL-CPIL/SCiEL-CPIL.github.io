"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Rank-1 + Rank-mu Natural Gradient Descent with diagonal step-size vector."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit with 10s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Diagonal step-size vector (per-dimension sigma)
        sigma = np.ones(dim) * 2.0

        # Population size for rank-mu
        lam = 4  # population size
        mu = 2   # number of best individuals to use for rank-mu update

        # Call counter
        calls_made = 1

        # Pre-allocate for population
        pop_x = np.empty((lam, dim))
        pop_f = np.empty(lam)

        # Generation loop
        gen = 0
        while calls_made < budget and time.time() < deadline:
            # Check time more frequently
            if time.time() > deadline - 5.0:
                break

            # Generate population
            for i in range(lam):
                if calls_made >= budget or time.time() > deadline - 2.0:
                    break
                # Sample with per-dimension sigma
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                pop_x[i] = y
                pop_f[i] = func(y)
                calls_made += 1

                if pop_f[i] < best_fx:
                    best_fx = pop_f[i]
                    best_x = y.copy()

            # Sort by fitness
            order = np.argsort(pop_f)
            sorted_x = pop_x[order]
            sorted_f = pop_f[order]

            # Rank-1 update (weight the best individual)
            w1 = 1.0 / mu
            # Rank-mu update (weights for the mu best)
            # Standard weights: w_i = log(mu + 0.5) - log(i) for i=1..mu, normalized
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()
            wmu = weights.sum() / mu  # effective weight sum

            # Rank-1: mean of best
            mean_best = sorted_x[0]
            # Rank-mu: weighted mean of mu best
            mean_mu = np.zeros(dim)
            for i in range(mu):
                mean_mu += weights[i] * sorted_x[i]

            # Update position
            # p_sigma (cumulation for sigma) - simplified: use rank-1 direction
            # For diagonal sigma, we update each dimension independently

            # Compute update directions
            # Rank-1 contribution
            d1 = mean_best - x
            # Rank-mu contribution
            dmu = mean_mu - x

            # Update x with combined direction
            # Use a step size that balances rank-1 and rank-mu
            alpha1 = 1.0
            alphamu = 1.0
            x = x + alpha1 * d1 + alphamu * dmu
            x = np.clip(x, -5, 5)

            # Update sigma (per-dimension)
            # For rank-1: if the best move was successful, increase sigma in that direction
            # For simplicity, use a heuristic based on progress
            for d in range(dim):
                # If the best individual improved in this dimension, increase sigma
                # Otherwise, decrease
                if sorted_f[0] < fx:
                    # Improvement: increase sigma in dimensions where best moved
                    if abs(d1[d]) > 1e-10:
                        sigma[d] *= 1.1
                    else:
                        sigma[d] *= 0.95
                else:
                    sigma[d] *= 0.9

                # Clamp sigma
                sigma[d] = np.clip(sigma[d], 0.01, 10.0)

            fx = sorted_f[0]  # Update current best function value

            gen += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
