"""Optimized optimizer for Katsuura (f23) on [-5,5]^5 with 500 budget."""
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with Latin Hypercube initialization and adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _latin_hypercube(self, rng, n_samples, dim):
        """Generate n_samples points via Latin Hypercube Sampling in [-5, 5]^dim."""
        points = np.zeros((n_samples, dim))
        for d in range(dim):
            perm = rng.permutation(n_samples)
            points[:, d] = (perm + 0.5) / n_samples * 10.0 - 5.0
        return points

    def _cma_es(self, func, x0, budget, rng, deadline):
        """Run CMA-ES starting from x0 with given budget."""
        dim = self.dim
        n = dim
        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        # Initial sigma
        sigma = 1.0
        # Covariance matrix
        C = np.eye(n)
        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)
        # Step sizes for paths
        cs = (1.5 + n / 2.0) / (n + 1.5 + 2.0 / n)
        cmu = min(0.3, (2 - (1 + cs) ** 2) / ((n + 2) ** 2 + 1 - cs))
        damps = (1 + 2 * max(0, np.sqrt((n - 1) / (n + 2)) - 1) + cs) / \
                (1 + 2 * max(0, np.sqrt((n - 1) / (n + 2)) - 1 - cs) + cs)
        # Eigen decomposition cache
        D = np.ones(n)
        B = np.eye(n)
        # Best
        best_x = x0.copy()
        best_f = func(x0)
        calls = 1

        # Generate initial population
        for _ in range(budget - 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Update eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            B = eigvecs
            D = np.sqrt(eigvals)

            # Generate offspring
            offspring = []
            fitnesses = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                y = B @ (D * z)
                x = x0 + sigma * y
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                offspring.append(x)
                fitnesses.append(fx)

            if not offspring:
                break

            # Sort by fitness
            indices = np.argsort(fitnesses)
            offspring = np.array(offspring)
            fitnesses = np.array(fitnesses)
            offspring = offspring[indices]
            fitnesses = fitnesses[indices]

            # Update best
            if fitnesses[0] < best_f:
                best_f = fitnesses[0]
                best_x = offspring[0].copy()

            # Compute weighted mean
            mu = lambda_ // 2
            weights = np.zeros(lambda_)
            wsum = np.sum(np.log(np.arange(1, mu + 1)))
            for i in range(mu):
                weights[i] = np.log(mu + 1) - np.log(i + 1)
            weights /= wsum
            mu_eff = 1.0 / np.sum(weights ** 2)

            m = np.sum(weights[:mu] * offspring[:mu], axis=0)

            # Update evolution paths
            delta_x = (m - x0) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * delta_x
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) / np.sqrt(mu_eff)) < \
                   (1.4 + 2 / (n + 1)) * (1 - cs)
            pc = (1 - cmu) * pc + hsig * cmu / cmu * delta_x
            pcnorm = np.linalg.norm(pc)
            hpc = (pcnorm / np.sqrt(1 - (1 - cmu) ** 2) / np.sqrt(mu_eff)) < \
                  (1.4 + 2 / (n + 1)) * cmu
            C = (1 + cmu * (2 * hpc - 1)) * C + \
                cmu * (delta_x[:, None] * delta_x[None, :] - (1 - hpc) * C)

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update mean
            x0 = m

        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # Deadline: 290s to be safe
        start_time = time.time()
        deadline = start_time + 290.0

        # Phase 1: Latin Hypercube sampling for initial diverse points
        # Use about 20% of budget for initialization
        n_init = max(10, budget // 5)
        if n_init > 20:
            n_init = 20

        lhd = self._latin_hypercube(rng, n_init, dim)
        best_x = lhd[0].copy()
        best_f = func(lhd[0])
        calls = 1

        for i in range(1, n_init):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            fi = func(lhd[i])
            calls += 1
            if fi < best_f:
                best_f = fi
                best_x = lhd[i].copy()

        # Phase 2: Multi-start CMA-ES from top diverse points
        # Select top k starting points from LHS
        evals = [func(lhd[i]) for i in range(n_init)] if n_init > 1 else [best_f]
        # Actually we already evaluated them, let's track properly
        # Re-evaluate is wasteful; we have best_f but not all values.
        # Let's just pick a few diverse points.
        
        # Pick top 3-5 points from LHS based on their fitness
        # We need to store fitnesses
        lhs_fitnesses = np.array([func(lhd[i]) for i in range(n_init)])
        # Wait, that's double counting calls. Let's fix.
        
        # Actually, let's restructure: evaluate all LHS points first, track calls
        # Then pick top few for CMA-ES

        # Reset and do it properly
        lhs_fitnesses = np.zeros(n_init)
        lhs_fitnesses[0] = best_f
        for i in range(1, n_init):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            lhs_fitnesses[i] = func(lhd[i])
            calls += 1
            if lhs_fitnesses[i] < best_f:
                best_f = lhs_fitnesses[i]
                best_x = lhd[i].copy()

        # Sort LHS points by fitness
        sorted_idx = np.argsort(lhs_fitnesses)
        
        # Pick top 3 distinct starting points (spread out)
        start_points = []
        used = set()
        for idx in sorted_idx:
            if len(start_points) >= 3:
                break
            pt = lhd[idx]
            # Check if sufficiently different from existing starts
            too_close = False
            for sp in start_points:
                if np.linalg.norm(pt - sp) < 1.0:
                    too_close = True
                    break
            if not too_close:
                start_points.append(pt)
        
        # If we don't have 3, just take the best ones
        while len(start_points) < min(3, n_init):
            for idx in sorted_idx:
                if lhd[idx] not in start_points:
                    start_points.append(lhd[idx])
                    break

        # Phase 2: Run CMA-ES from each start point with remaining budget
        remaining = budget - calls
        if remaining > 0 and start_points:
            n_starts = len(start_points)
            budget_per_start = remaining // n_starts
            # Give a bit more to the best start
            if n_starts > 1:
                budgets = [budget_per_start] * n_starts
                budgets[0] += remaining - budget_per_start * n_starts
            else:
                budgets = [remaining]

            for i, sp in enumerate(start_points):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                if budgets[i] < 5:
                    continue

                # For the best start, use full CMA-ES
                # For others, use a simpler local search
                if i == 0:
                    bx, bf, used_calls = self._cma_es(func, sp, budgets[i], rng, deadline)
                    calls += used_calls - 1  # -1 because we already counted the initial eval
                    if bf < best_f:
                        best_f = bf
                        best_x = bx.copy()
                else:
                    # Simpler (1+1)-ES for secondary starts
                    x = sp.copy()
                    fx = func(x)
                    calls += 1
                    sigma = 0.5
                    for _ in range(budgets[i] - 1):
                        if time.time() > deadline:
                            break
                        if calls >= budget:
                            break
                        y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                        fy = func(y)
                        calls += 1
                        if fy < fx:
                            x, fx = y, fy
                            sigma *= 1.3
                        else:
                            sigma *= 0.85
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        return best_x, best_f
