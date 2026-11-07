"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 30 and adaptive stopping for f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size lambda
        lam = 30
        # Number of selected individuals mu
        mu = lam // 2  # 15
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial parameters
        rng = np.random.default_rng(42)
        # Start near center for better initial coverage, but slightly randomized
        x_mean = np.zeros(dim)
        sigma = 2.5  # Initial step size, covers a good portion of [-5,5]^5

        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        # Precompute C^{-1/2} initially as I
        C_inv_sqrt = np.eye(dim)

        # Track best
        best_x = x_mean.copy()
        best_f = func(best_x)
        call_count = 1

        # Generate initial population
        def generate_population(x_mean, sigma, C_inv_sqrt, rng):
            z = rng.standard_normal((lam, dim))
            # Transform z to have covariance C
            # y = C_inv_sqrt @ z
            y = z @ C_inv_sqrt.T
            # x = x_mean + sigma * y
            pop = x_mean + sigma * y
            return pop

        # Main loop
        gen = 0
        while call_count < budget:
            if time.time() > deadline:
                break

            # Generate population
            pop = generate_population(x_mean, sigma, C_inv_sqrt, rng)

            # Evaluate population
            fvals = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                if call_count >= budget:
                    break
                x_i = np.clip(pop[i], -5, 5)
                fvals[i] = func(x_i)
                call_count += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x_i.copy()

            # If we broke out early due to time/budget, we might have incomplete fvals
            # But the loop structure above breaks the inner loop, so we need to handle that.
            # Let's restructure to ensure we only process complete generations or handle partial ones.
            # For simplicity and safety, let's assume we complete the generation if we haven't hit deadline/budget.
            # The inner loop breaks, but then we proceed to update which might be wrong if fvals is incomplete.
            # Better: check if we have a full generation.
            if call_count < budget and time.time() < deadline:
                # Only if we actually evaluated all lam points in this generation
                # The loop above increments call_count. If it broke early, call_count < budget might be true,
                # but fvals has uninitialized values.
                # Let's check if the inner loop completed.
                # Actually, the inner loop runs for i in range(lam). If it breaks, we exit the for loop.
                # We need a flag.
                pass

            # Let's rewrite the evaluation part to be safer.
            
            # --- Revised Evaluation Block ---
            # We are inside the while loop.
            # Let's just check if we can afford another full generation.
            if call_count + lam > budget:
                # Do a few more individual evaluations if we have budget left
                while call_count < budget and time.time() < deadline:
                    # Simple local step from best_x
                    y = np.clip(best_x + 0.1 * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    call_count += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y
                break

            # Full generation
            pop = generate_population(x_mean, sigma, C_inv_sqrt, rng)
            fvals = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    # If time runs out during generation, we stop.
                    # We can't update CMA-ES state with partial data easily.
                    # Just return best found so far.
                    return best_x, best_f
                x_i = np.clip(pop[i], -5, 5)
                fvals[i] = func(x_i)
                call_count += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x_i.copy()

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Update mean
            w = weights
            x_mean_new = np.dot(w, pop_sorted[:mu, :])

            # Update step size sigma
            # Calculate norm of ps
            # First update ps
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * C_inv_sqrt @ (x_mean_new - x_mean)
            delta_mean = x_mean_new - x_mean
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_inv_sqrt @ delta_mean)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) / np.sqrt(1 - (1 - c_sigma) ** 2)
            hsig = min(hsig, 1.4 + 2 / (dim + 1))

            # Update pc
            # pc = (1 - c_c) * pc + hsig * sqrt(c_c * (2 - c_c) * mu_eff) * delta_mean / sigma
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_mean / sigma

            # Update Covariance Matrix C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (hsig > 1 - c_c) * (1 - c_c) * (2 - c_c)) + c_mu * sum(w_i * (y_i - y_mean) @ (y_i - y_mean).T)
            # y_i = (x_i - x_mean_old) / sigma
            # y_mean = (x_mean_new - x_mean_old) / sigma = delta_mean / sigma
            y = (pop_sorted[:mu, :] - x_mean) / sigma
            y_mean = delta_mean / sigma
            
            # Rank-one update from ps
            # Note: The term (hsig > 1 - c_c) is a boolean, effectively 1 or 0
            rank_one_ps = pc @ pc.T + (hsig > 1 - c_c) * (1 - c_c) * (2 - c_c)
            
            # Rank-mu update
            # sum(w_i * (y_i - y_mean) @ (y_i - y_mean).T)
            diff = y - y_mean
            rank_mu = np.einsum('i,ij,ik->jk', w, diff, diff)

            C = (1 - c_1 - c_mu) * C + c_1 * rank_one_ps + c_mu * rank_mu

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update C_inv_sqrt
            # We need C^{-1/2}. We can compute it via eigen decomposition.
            # Since we just did it for regularization, we can reuse.
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) - np.sqrt(dim) * (1 + (1 - c_c) / (2 * c_c))))
            # Actually, the standard formula is:
            # sigma = sigma * exp(c_sigma / d_sigma * (||ps|| / chi_N - 1))
            # where chi_N = sqrt(dim) * (1 - 1/(4*dim) - 1/(25*dim^2))
            chi_N = np.sqrt(dim) * (1 - 1.0 / (4 * dim) - 1.0 / (25 * dim ** 2))
            ps_norm = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_N - 1))

            # Update mean
            x_mean = x_mean_new

            # Adaptive stopping criterion
            # Check if progress is negligible
            if gen > 5:
                # If the best fitness hasn't improved significantly in last 10 generations
                # Or if sigma is very small
                if sigma < 1e-10:
                    break
                # Simple stagnation check: if the range of best 10 fitness values is very small
                # We don't store history, so let's just check if the improvement in last step was tiny
                # For now, let's rely on budget and time.
                # A better check: if the difference between best and second best is tiny and sigma is small.
                if fvals_sorted[1] - fvals_sorted[0] < 1e-10 * (1 + abs(fvals_sorted[0])):
                    # Converged
                    break

            gen += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
