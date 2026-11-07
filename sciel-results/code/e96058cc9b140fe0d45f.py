"""Optimized search for BBOB f13 Sharp Ridge using CMA-ES with coordinate descent polish."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with coordinate descent polish for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        calls = [0]

        def safe_call(x):
            if calls[0] >= budget:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            try:
                val = float(func(x))
                calls[0] += 1
                return val
            except:
                calls[0] += 1
                return float('inf')

        best_x = np.zeros(dim)
        best_f = float('inf')

        # Phase 1: CMA-ES to find the basin and learn landscape geometry
        # CMA-ES handles the hidden rotation through covariance adaptation
        
        # Initialize CMA-ES
        x_mean = rng.uniform(-3.0, 3.0, dim)
        f_mean = safe_call(x_mean)
        best_x = x_mean.copy()
        best_f = f_mean

        # CMA-ES parameters
        n = dim
        lam = max(4, int(np.ceil(4 + 3 * np.log(n))))  # population size
        sigma = 1.5  # Initial step size
        c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + 10)
        cmu = min(0.5 - c1, 2 * ((2 - c1) ** 2 - 1) / ((n + 2) ** 2 + 12))
        cs = (n + 1.3) / (n + 5)
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        # Covariance matrix C (start as identity)
        C = np.eye(n)
        # Eigen decomposition for sampling
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-12))

        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Weights for recombination
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # CMA-ES loop
        cma_calls_used = 0
        max_cma_calls = min(budget // 2, 200)  # Use half the budget for CMA-ES
        
        while cma_calls_used < max_cma_calls and calls[0] < budget and time.time() < deadline - 30:
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(lam):
                if calls[0] >= budget:
                    break
                z = rng.standard_normal(n)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5.0, 5.0)
                offspring.append(y)
                fy = safe_call(y)
                cma_calls_used += 1
                offspring_f.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(offspring) < lam:
                break

            # Sort by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_f = [offspring_f[i] for i in indices]

            # Check for stagnation
            if sorted_f[0] - best_f < 1e-10:
                # No improvement, reduce sigma
                sigma *= 0.8
                if sigma < 1e-6:
                    break
                continue

            # Update mean
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += w[i] * sorted_offspring[i]
            x_new = y_w

            # Update sigma (CSA)
            z_sigma = (B.T @ (D * (B.T @ (x_new - x_mean)))) / sigma if sigma > 1e-15 else np.zeros(n)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * z_sigma
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (lam + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            if hsig:
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update C (covariance matrix)
            y_w_diff = x_new - x_mean
            x_mean = x_new
            p_c = (1 - c1) * p_c + hsig * np.sqrt(c1 * (2 - c1) * mu_eff) * y_w_diff
            C_update = np.outer(p_c, p_c) - (1 - c1) * C
            C_update += (1 - c1) * (1 + 2 * cmu / mu_eff - c1) * np.eye(n)
            for i in range(mu):
                y_i = (B @ (D * (B.T @ (sorted_offspring[i] - x_new)))) / sigma
                C_update += w[i] * np.outer(y_i, y_i)
            C = C + C_update

            # Ensure C is positive definite
            C = (C + C.T) / 2
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))

            # If sigma is very small, CMA-ES has converged
            if sigma < 0.01:
                break

        # Phase 2: Coordinate descent polish from the best point found by CMA-ES
        # This exploits the anisotropic ridge geometry for final convergence
        x = best_x.copy()
        f = best_f
        
        # Adaptive step sizes per dimension
        steps = np.ones(dim) * 0.5
        min_step = 1e-8
        max_step = 5.0

        last_improvement_call = calls[0]

        # Coordinate descent loop
        while calls[0] < budget and time.time() < deadline:
            if time.time() > deadline - 5:
                break

            # Check for stagnation
            if calls[0] - last_improvement_call > 30:
                # Try a small random jump
                x_jump = x + rng.normal(0, 0.1, dim)
                x_jump = np.clip(x_jump, -5.0, 5.0)
                f_jump = safe_call(x_jump)
                if f_jump < f:
                    x = x_jump.copy()
                    f = f_jump
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        last_improvement_call = calls[0]
                    # Reset steps
                    steps = np.ones(dim) * 0.3
                else:
                    # Reduce all steps
                    steps = np.maximum(min_step, steps * 0.5)
                    if np.all(steps < min_step * 10):
                        break

            # Coordinate descent: sweep through all dimensions
            improved_this_sweep = False
            for d in range(dim):
                if calls[0] >= budget or time.time() > deadline - 5:
                    break

                # Try positive and negative steps
                for direction in [1, -1]:
                    if calls[0] >= budget:
                        break
                    x_trial = x.copy()
                    x_trial[d] += direction * steps[d]
                    x_trial = np.clip(x_trial, -5.0, 5.0)

                    f_trial = safe_call(x_trial)
                    if f_trial < f:
                        # Accept the move
                        x[d] = x_trial[d]
                        f = f_trial
                        improved_this_sweep = True
                        # Increase step size slightly
                        steps[d] = min(max_step, steps[d] * 1.1)

                        # Update best
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            last_improvement_call = calls[0]

                        # Continue in the same direction (line search)
                        for _ in range(2):
                            if calls[0] >= budget or time.time() > deadline - 5:
                                break
                            x_trial2 = x.copy()
                            x_trial2[d] += direction * steps[d]
                            x_trial2 = np.clip(x_trial2, -5.0, 5.0)
                            f_trial2 = safe_call(x_trial2)
                            if f_trial2 < f:
                                x[d] = x_trial2[d]
                                f = f_trial2
                                if f < best_f:
                                    best_f = f
                                    best_x = x.copy()
                                    last_improvement_call = calls[0]
                            else:
                                break
                    else:
                        # Reject, decrease step
                        steps[d] = max(min_step, steps[d] * 0.7)

            # If no improvement in entire sweep, reduce all steps
            if not improved_this_sweep:
                steps = np.maximum(min_step, steps * 0.5)
                if np.all(steps < min_step * 10):
                    break

        return best_x, best_f
