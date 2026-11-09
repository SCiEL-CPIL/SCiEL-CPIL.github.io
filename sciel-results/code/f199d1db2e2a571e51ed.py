"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start and Rastrigin-aware local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = np.zeros(n)
        best_f = float('inf')
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Multi-start exploration to find good basins
        # Use ~30% of budget for diverse initial sampling
        explore_budget = int(budget * 0.30)
        init_points = []
        for i in range(explore_budget):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            if fx is None:
                break
            init_points.append((fx, x))

        # Sort by fitness
        init_points.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES from the best starting point
        if init_points:
            start_x = init_points[0][1].copy()
            start_f = init_points[0][0]
        else:
            start_x = np.zeros(n)
            start_f = best_f

        # CMA-ES parameters (from Inspiration 1, tuned for dim=5)
        lam = 10  # population size
        mu = lam // 2  # number of parents

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        # Initial state
        mean = start_x.copy()
        sigma = 1.5  # moderate initial step size
        C = np.eye(n)

        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)

        # Learning rates
        cs = (mueff + 2) / (n + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = 2 * (mueff - 2 + np.sum(w[:mu] ** 2) / np.sum(w) ** 2) / ((n + 2) ** 2 + mueff)
        cmu = min(cmu, 2 * (1 - c1))

        # CMA-ES loop
        while calls_used < budget:
            if time.time() > deadline:
                break

            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                L = np.eye(n)
                ps = np.zeros(n)
                pc = np.zeros(n)

            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = mean + sigma * L @ z
                offspring.append(y)

            # Evaluate offspring
            offspring_f = []
            for y in offspring:
                fy = evaluate(y)
                if fy is None:
                    break
                offspring_f.append((fy, y))

            if len(offspring_f) < mu:
                break

            # Sort by fitness
            offspring_f.sort(key=lambda x: x[0])

            # Select mu best
            selected = [x[1] for x in offspring_f[:mu]]

            # Update mean
            m_old = mean.copy()
            mean_new = np.sum([w[i] * selected[i] for i in range(mu)], axis=0)

            # Update ps
            C_inv_sqrt = np.linalg.inv(L)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (mean_new - m_old) / sigma

            # Check hsig
            # Approximate the time-dependent term
            gen = calls_used - explore_budget
            if gen > 0:
                hsig_threshold = (1.4 + 2 / (n + 1)) * np.sqrt(n + 1)
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen / (n * damps))) < hsig_threshold
            else:
                hsig = True

            # Update pc
            if hsig:
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * (mean_new - m_old) / sigma

            # Update C
            delta_sig = 1.0 if hsig else 0.0
            C = (1 - c1 - cmu + c1 * delta_sig) * C
            C += c1 * (1 - delta_sig) * np.outer(ps, ps)
            C += c1 * delta_sig * (np.outer(ps, ps) + (1 - (1 - cs) ** 2) * np.outer(pc, pc))
            C += cmu * np.sum([w[i] * np.outer(selected[i] - m_old, selected[i] - m_old) for i in range(mu)], axis=0)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += 1e-12 * np.eye(n)

            mean = mean_new

            # Adapt sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
            sigma = max(1e-10, min(sigma, 10.0))

            # Check convergence
            if np.linalg.norm(mean_new - m_old) < 1e-8:
                break

        # Phase 3: Rastrigin-aware local refinement
        # Rastrigin has periodic structure with period 2*pi in each dimension.
        # The global minimum (before shift/rotation) is at integer multiples of 2*pi.
        # After shift/rotation, the minimum is at some point in [-4,4]^5.
        # We try to refine by snapping coordinates near "nice" values.
        # Strategy: try small perturbations around best_x, focusing on integer-like values.
        
        remaining_budget = budget - calls_used
        if remaining_budget > 10 and time.time() < deadline - 5:
            # Try a local refinement with decreasing steps
            # Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
            # Minimum at x_i = k (integer) for each i, before shift/rotation.
            # After rotation, the "integer lattice" is rotated, so we can't simply snap to integers.
            # Instead, we do a fine-grained local search around best_x.
            
            # Use a simple Nelder-Mead-like refinement or coordinate descent
            # with very small steps to converge to the local minimum.
            
            x_curr = best_x.copy()
            f_curr = best_f
            step = 0.1
            
            for _ in range(remaining_budget // 5):
                if calls_used >= budget or time.time() > deadline - 2:
                    break
                
                improved = False
                for d in range(n):
                    for direction in [-1, 1]:
                        x_try = x_curr.copy()
                        x_try[d] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = evaluate(x_try)
                        if f_try is None:
                            break
                        if f_try < f_curr - 1e-10:
                            x_curr = x_try.copy()
                            f_curr = f_try
                            improved = True
                            break
                    if calls_used >= budget or time.time() > deadline - 2:
                        break
                    if improved:
                        break
                
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
