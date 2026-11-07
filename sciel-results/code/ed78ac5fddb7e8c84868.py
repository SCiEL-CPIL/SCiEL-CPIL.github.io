"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with adaptive step sizes and population-based search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        B = self.budget

        # Initial population size
        lam = max(4, int(4 + 3 * np.log(d)))
        # Ensure we don't exceed budget
        if lam * 2 > B:
            lam = max(2, B // 2)

        # Initial point: center of box
        x = np.zeros(d)
        # Initial sigma: large enough to explore
        sigma = 3.0

        # Best found so far
        best_x = x.copy()
        best_f = func(best_x)

        # CMA-ES-like parameters
        # We'll use a simplified CMA-ES with population
        # Track mean and covariance-like structure via step sizes per dimension
        # For simplicity and robustness, use a diagonal CMA-ES (sep-ES)

        # Per-dimension step sizes
        sigmas = np.full(d, sigma)

        # Success counters for 1/5 rule per dimension
        successes = np.zeros(d)
        total = np.zeros(d)

        calls_made = 1  # We already made one call for best_x

        # Main loop
        while calls_made < B:
            # Generate population
            offspring = []
            fitnesses = []

            # Number of offspring to generate in this round
            n_off = min(lam, B - calls_made)

            for i in range(n_off):
                # Sample from multivariate normal with diagonal covariance
                # Add some diversity: mix of large and small steps
                if i == 0:
                    # One large step for exploration
                    step = sigmas * 2.0 * rng.standard_normal(d)
                elif i == 1:
                    # One medium step
                    step = sigmas * 1.0 * rng.standard_normal(d)
                else:
                    # Standard step
                    step = sigmas * rng.standard_normal(d)

                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1

                offspring.append(y)
                fitnesses.append(fy)

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Sort by fitness
            sorted_idx = np.argsort(fitnesses)
            sorted_offspring = [offspring[i] for i in sorted_idx]
            sorted_fitnesses = [fitnesses[i] for i in sorted_idx]

            # Update mean: weighted average of top half
            mu = max(1, n_off // 2)
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(i + 1) - np.log(mu)
            weights /= weights.sum()

            new_x = np.zeros(d)
            for i in range(mu):
                new_x += weights[i] * sorted_offspring[i]

            # Update per-dimension step sizes using 1/5 rule
            for j in range(d):
                # Count successes: how many of the top mu have better j-th coordinate
                # Simple heuristic: if the best offspring has a different j-th coord, 
                # and it's an improvement, count as success
                # More robust: compare mean movement
                delta_j = new_x[j] - x[j]
                # If the movement is in the direction that improved fitness on average
                # Use a simpler rule: if the best offspring moved in dimension j
                # and the overall fitness improved, count as success
                # For diagonal CMA-ES, use the ratio of successful steps
                # Here we use a simplified approach:
                # If the new mean is closer to the best in dimension j, it's a success
                if abs(new_x[j] - best_x[j]) < abs(x[j] - best_x[j]):
                    successes[j] += 1
                total[j] += 1

            # Adapt sigmas based on success rate
            for j in range(d):
                if total[j] >= 5:
                    rate = successes[j] / total[j]
                    if rate > 0.5:
                        sigmas[j] *= 1.2
                    elif rate < 0.2:
                        sigmas[j] *= 0.8
                    # Reset counters periodically
                    if total[j] >= 20:
                        successes[j] = 0
                        total[j] = 0

            # Clamp sigmas
            sigmas = np.clip(sigmas, 0.01, 10.0)

            x = new_x

        return best_x, best_f
# EVOLVE-BLOCK-END
