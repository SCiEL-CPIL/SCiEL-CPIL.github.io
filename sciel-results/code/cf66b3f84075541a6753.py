"""Optimized Rank-1 + Rank-mu Natural Gradient Descent with diagonal step-size."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Rank-1 + Rank-mu Natural Gradient Descent with diagonal step-size vector.
    
    Models the search distribution as an axis-aligned ellipsoid (diagonal covariance)
    and updates it via natural gradient descent on the log-diagonal parameters.
    This avoids full covariance matrix overhead while adapting per-dimension scales
    to rapidly flatten anisotropic landscapes like the Discus function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        B = self.budget

        # Initial point in the center of the box
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Diagonal step-size (sigma per dimension), start with a moderate value
        sigma = np.full(d, 1.5)

        # Population size: use larger population for better gradient estimation
        mu = 15  # number of offspring per generation
        # Learning rates for natural gradient updates (tuned for better convergence)
        lr_rank1 = 0.3 / (mu * d)       # rank-1 update rate
        lr_rankmu = 0.15 / (mu * d)     # rank-mu update rate

        # Momentum for smoother updates
        momentum_s = np.zeros(d)
        momentum_x = np.zeros(d)
        beta = 0.8  # momentum coefficient

        calls_used = 1  # first evaluation

        gen = 0
        while calls_used < B:
            gen += 1
            # Adapt population size: start larger, shrink as we converge
            if gen < 5:
                mu_gen = mu
            elif gen < 20:
                mu_gen = max(5, mu - (gen - 5))
            else:
                mu_gen = max(3, mu - (gen - 5) - 10)

            if calls_used + mu_gen > B:
                mu_gen = B - calls_used
            if mu_gen <= 0:
                break

            # Generate mu offspring
            offspring = []
            offspring_f = []
            for i in range(mu_gen):
                if calls_used >= B:
                    break
                # Sample from N(x, diag(sigma^2))
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                offspring.append(y)
                offspring_f.append(fy)

            if len(offspring) == 0:
                break

            # Sort by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_f = [offspring_f[i] for i in indices]

            # Update best
            if sorted_f[0] < best_fx:
                best_fx = sorted_f[0]
                best_x = sorted_offspring[0].copy()

            # Natural gradient update for log(sigma)
            s = np.log(sigma)

            # Rank-1 update: direction towards best offspring
            y_best = np.array(sorted_offspring[0])
            delta_r1 = (y_best - x) / (sigma + 1e-10)
            
            # Rank-mu update: weighted average of top offspring
            n_top = max(1, mu_gen // 2)
            top_offspring = np.array(sorted_offspring[:n_top])
            weights = np.arange(n_top, 0, -1, dtype=float)
            weights /= weights.sum()
            avg_y = np.average(top_offspring, axis=0, weights=weights[:, None])
            delta_rm = (avg_y - x) / (sigma + 1e-10)

            # Apply updates with momentum
            update_s = lr_rank1 * delta_r1 + lr_rankmu * delta_rm
            momentum_s = beta * momentum_s + (1 - beta) * update_s
            s += momentum_s

            # Clamp sigma to reasonable bounds
            sigma = np.exp(np.clip(s, np.log(0.005), np.log(8.0)))

            # Update mean x: move towards best offspring with adaptive step
            # Use a step size that adapts based on recent progress
            step_size = 0.5 / (1 + gen * 0.02)
            update_x = y_best - x
            momentum_x = beta * momentum_x + (1 - beta) * update_x
            x = x + step_size * momentum_x
            x = np.clip(x, -5, 5)

            # If stuck, reduce sigma and try a random restart direction
            if gen % 20 == 0 and sorted_f[0] >= best_fx * 0.999:
                sigma *= 0.8

        # Final local refinement around best_x
        if calls_used < B:
            local_sigma = np.maximum(sigma * 0.2, 0.01)
            for _ in range(B - calls_used):
                if calls_used >= B:
                    break
                z = rng.standard_normal(d)
                y = best_x + local_sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    local_sigma *= 0.85
                else:
                    local_sigma *= 0.95

        return best_x, best_fx
# EVOLVE-BLOCK-END
