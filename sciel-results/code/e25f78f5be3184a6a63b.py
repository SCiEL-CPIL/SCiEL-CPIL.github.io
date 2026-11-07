"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with local refinement for BBOB f7 (Step Ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        d = self.dim
        budget = self.budget
        bounds = (-5, 5)

        # CMA-ES Parameters
        lam = 10  # Population size
        mu = lam // 2  # Number of best solutions used for recombination

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)

        # Learning rates
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff)))

        # Initialize
        x_mean = np.zeros(d)
        sigma = 2.0  # Initial step size, large enough for exploration
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)

        # PC and PS for path evolution
        pc = np.zeros(d)
        ps = np.zeros(d)

        best_x = None
        best_f = np.inf

        n_evals = 0
        it = 0

        # Phase 1: CMA-ES
        # Use 70% of budget for CMA-ES
        cma_budget = int(0.7 * budget)

        while n_evals + lam <= cma_budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sample
            Z = np.random.randn(lam, d)
            Y = x_mean + sigma * (Z * D) @ B.T
            Y = np.clip(Y, bounds[0], bounds[1])

            # Evaluate
            F = np.array([func(y) for y in Y])
            n_evals += lam

            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = Y[min_idx].copy()

            # Sort
            sorted_idx = np.argsort(F)
            Y_sorted = Y[sorted_idx]

            # Update mean
            y_old = x_mean.copy()
            x_mean = np.dot(w, Y_sorted[:mu])

            # Path evolution sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B.T @ (D * (x_mean - y_old)))

            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (it + 1)))) < (1 + 2 / (d + 1)) * cc
            it += 1

            # Path evolution covariance
            if hsig:
                pc = (1 - cc) * pc + cc / sigma * (B.T @ (D * (x_mean - y_old)))

            # Update covariance
            delta_hebbian = c1 * (np.outer(ps, ps) + (1 - hsig) * (cc * (2 - cc) * np.outer(pc, pc)))

            for i in range(mu):
                delta_hebbian += (cmu * w[i]) * np.outer(Y_sorted[i] - y_old, Y_sorted[i] - y_old) / (sigma * sigma)

            C = (1 - c1 - cmu) * C + delta_hebbian
            C = (C + C.T) / 2

            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D2, 1e-20, None))

            # Step size adaptation
            chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Convergence check
            if sigma < 1e-5:
                break

        # Phase 2: Local Refinement
        if best_x is not None and n_evals < budget:
            x = best_x.copy()
            f_cur = best_f

            # Use a shrinking step size approach
            # Start with a step size based on current sigma, but bounded
            initial_ss = max(sigma, 0.05)

            # Generate a list of step sizes to try
            # We want to cover from coarse to fine
            step_sizes = []
            ss = initial_ss
            while ss > 1e-4:
                step_sizes.append(ss)
                ss *= 0.5

            # For each step size, try small perturbations in all directions
            for ss in step_sizes:
                if n_evals >= budget or time.time() > deadline:
                    break

                # Try perturbations in each dimension
                for i in range(d):
                    for direction in [-1, 1]:
                        if n_evals >= budget or time.time() > deadline:
                            break

                        y = x.copy()
                        y[i] += direction * ss
                        y = np.clip(y, bounds[0], bounds[1])

                        fy = func(y)
                        n_evals += 1

                        if fy < f_cur:
                            x = y
                            f_cur = fy
                            if f_cur < best_f:
                                best_f = f_cur
                                best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
