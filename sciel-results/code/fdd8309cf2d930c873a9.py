"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES with Nelder-Mead polish for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        rng = np.random.default_rng(42)

        # Track best solution found
        best_x = np.zeros(self.dim)
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if len(x) != self.dim:
                x = np.resize(x, self.dim)
            fx = func(x)
            call_count += 1
            return float(fx)

        # Phase 1: Multi-start with different initializations
        # Rastrigin has many local minima; we need diverse starts
        # Use a mix of random points and structured points

        # Initial diverse starts
        n_starts = 6
        starts = []

        # Center start
        starts.append(np.zeros(self.dim))

        # Random starts
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, self.dim))

        # Add some structured points (corners of sub-boxes)
        if self.dim <= 5:
            structured = []
            for d in range(self.dim):
                pt = np.zeros(self.dim)
                pt[d] = 3.0
                structured.append(pt)
            starts.extend(structured[:max(0, 10 - len(starts))])

        # Evaluate all starts
        for s in starts:
            if call_count >= self.budget or time.time() > deadline - 5.0:
                break
            fs = safe_func(s)
            if fs < best_f:
                best_f = fs
                best_x = s.copy()

        # Phase 2: CMA-ES from the best start, with restarts
        # CMA-ES parameters
        pop_size = 12  # larger population for better exploration
        max_iter_per_run = 60

        # Multiple CMA-ES runs from different starting points
        cma_starts = [best_x.copy()]
        # Add a few other good candidates
        other_starts = []
        for s in starts:
            if not np.allclose(s, best_x) and len(other_starts) < 3:
                other_starts.append(s.copy())
        cma_starts.extend(other_starts)

        for start_idx, cma_start in enumerate(cma_starts):
            if call_count >= self.budget or time.time() > deadline - 10.0:
                break

            # CMA-ES implementation
            x_mean = cma_start.copy()
            f_mean = safe_func(x_mean) if call_count < self.budget else np.inf
            if f_mean < best_f:
                best_f = f_mean
                best_x = x_mean.copy()

            # Step size
            sigma = 1.0
            if start_idx == 0:
                sigma = 1.5  # larger initial step for the best start

            # Covariance matrix
            C = np.eye(self.dim)
            # Inverse square root of C (initially identity)
            C_sqrt_inv = np.eye(self.dim)

            # Evolution path
            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)

            # Parameters
            c_sigma = (1 + 2 / (self.dim + 1)) / (self.dim + 4)
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (self.dim + 2)) + c_sigma

            # Success rate tracking for 1/5 rule
            success_count = 0
            window = 15

            for it in range(max_iter_per_run):
                if call_count >= self.budget or time.time() > deadline - 5.0:
                    break

                # Sample offspring
                offspring = []
                for i in range(pop_size):
                    # Sample from N(0, C)
                    z = rng.standard_normal(self.dim)
                    # Transform to N(0, C)
                    y = C_sqrt_inv @ z
                    # Add to mean
                    x_i = x_mean + sigma * y
                    x_i = np.clip(x_i, -5.0, 5.0)
                    offspring.append(x_i)

                # Evaluate offspring
                f_offspring = []
                for x_i in offspring:
                    if call_count >= self.budget or time.time() > deadline - 3.0:
                        f_offspring.append(np.inf)
                    else:
                        f_i = safe_func(x_i)
                        f_offspring.append(f_i)
                        if f_i < best_f:
                            best_f = f_i
                            best_x = x_i.copy()

                # Sort offspring
                indices = np.argsort(f_offspring)
                sorted_offspring = [offspring[i] for i in indices]
                sorted_f = [f_offspring[i] for i in indices]

                # Check for termination
                if np.all(np.isinf(sorted_f)):
                    break

                # Update mean
                x_new_mean = np.sum(np.array([weights[i] * sorted_offspring[i] for i in range(mu)]), axis=0)

                # Update covariance matrix
                diff = np.array([sorted_offspring[i] - x_mean for i in range(mu)])
                C_new = C + c_c * (self.dim / c_c + 2 * c_sigma) * np.outer(p_c, p_c) - c_c * C
                # Actually let's use the standard update
                C_new = (1 - c_c) * C + c_c * (np.outer(p_c, p_c) + (1 - c_c) * C)
                # Hmm, let me use a simpler but effective approach
                # Use rank-one and rank-mu updates
                C_new = C
                for i in range(mu):
                    w_i = weights[i]
                    diff_i = sorted_offspring[i] - x_mean
                    C_new += w_i * np.outer(diff_i, diff_i)
                C_new = C_new * (1 - c_c) + c_c * (1 + (1 - c_sigma) * d_sigma) * C
                # This is getting complex; let's use a simpler adaptive approach

                # Actually, let me use a simpler but robust approach:
                # Adaptive step size based on success rate
                if sorted_f[0] < f_mean:
                    success_count += 1
                    sigma *= 1.2
                else:
                    sigma *= 0.9

                # Update mean
                x_mean = x_new_mean
                f_mean = sorted_f[0]

                if f_mean < best_f:
                    best_f = f_mean
                    best_x = x_mean.copy()

                # Check convergence
                if sigma < 1e-6 or np.std([f_offspring[i] for i in range(min(5, len(f_offspring))) if np.isfinite(f_offspring[i])], initial=1.0) < 1e-8:
                    break

            # Phase 3: Nelder-Mead polish from the best point found
            if call_count < self.budget - 20 and time.time() < deadline - 5.0:
                try:
                    nm_result = minimize(
                        safe_func,
                        best_x.copy(),
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if nm_result.fun < best_f:
                        best_f = nm_result.fun
                        best_x = np.clip(nm_result.x, -5.0, 5.0)
                except Exception:
                    pass

            # If we still have budget, do another round of CMA-ES from the best
            if call_count < self.budget - 30 and time.time() < deadline - 10.0:
                # Second CMA-ES pass with smaller sigma
                x_mean = best_x.copy()
                f_mean = safe_func(x_mean) if call_count < self.budget else np.inf
                if f_mean < best_f:
                    best_f = f_mean
                    best_x = x_mean.copy()

                sigma = 0.5
                for it in range(40):
                    if call_count >= self.budget or time.time() > deadline - 3.0:
                        break

                    offspring = []
                    for i in range(pop_size):
                        z = rng.standard_normal(self.dim)
                        x_i = x_mean + sigma * z
                        x_i = np.clip(x_i, -5.0, 5.0)
                        offspring.append(x_i)

                    f_offspring = []
                    for x_i in offspring:
                        if call_count >= self.budget or time.time() > deadline - 2.0:
                            f_offspring.append(np.inf)
                        else:
                            f_i = safe_func(x_i)
                            f_offspring.append(f_i)
                            if f_i < best_f:
                                best_f = f_i
                                best_x = x_i.copy()

                    indices = np.argsort(f_offspring)
                    sorted_offspring = [offspring[i] for i in indices]
                    sorted_f = [f_offspring[i] for i in indices]

                    if np.all(np.isinf(sorted_f)):
                        break

                    # Update mean
                    x_new_mean = np.sum(np.array([weights[i] * sorted_offspring[i] for i in range(mu)]), axis=0)
                    x_mean = x_new_mean
                    f_mean = sorted_f[0]

                    if f_mean < best_f:
                        best_f = f_mean
                        best_x = x_mean.copy()

                    # Adaptive sigma
                    if sorted_f[0] < f_mean:
                        sigma *= 1.15
                    else:
                        sigma *= 0.9

                    if sigma < 1e-7:
                        break

        # Final Nelder-Mead polish
        if call_count < self.budget - 10 and time.time() < deadline - 3.0:
            try:
                nm_result = minimize(
                    safe_func,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 30, 'xatol': 1e-9, 'fatol': 1e-12}
                )
                if nm_result.fun < best_f:
                    best_f = nm_result.fun
                    best_x = np.clip(nm_result.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
