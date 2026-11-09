"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer with Rastrigin-aware initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            val = func(x)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def cma_es(x0, sigma0, max_evals=200):
            """Simple CMA-ES implementation."""
            nonlocal call_count, best_x, best_f
            n = dim
            lam = 4 + int(3 * np.log(n))  # population size
            cc = (4 - lam / (n + 4)) / ((n + 1.3) * (lam + 7))  # path length evolution path
            cs = (lam / 2 + 1) / (n + lam / 2 + 3)  # sigma adaptation
            damps = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 1)) - 1) + cs
            c1 = 2 / ((n + 1.3) ** 2 + lam)
            cmu = min(1 - c1, 2 * ((lam / 2 - 2) / (lam + 2)) / ((n + 2) ** 2 + lam))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            alpha_sigma = (cs + 2) / (cs + damps)

            # Initialize
            mean = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            # Initial population
            pop = []
            fvals = []
            for i in range(lam):
                if call_count >= budget or time.time() > deadline:
                    return
                z = rng.standard_normal(n)
                x = mean + sigma * (B @ (D * z))
                x = np.clip(x, -5, 5)
                fv = call_func(x)
                if fv is None:
                    return
                pop.append(x)
                fvals.append(fv)

            evals = lam

            while evals < max_evals and call_count < budget and time.time() < deadline:
                # Sort by fitness
                order = np.argsort(fvals)
                pop = [pop[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update mean
                mu = lam // 2
                weights = np.ones(mu)
                # Use linear weights for better performance
                for i in range(mu):
                    weights[i] = np.log(mu + 0.5) - np.log(i + 1)
                weights /= weights.sum()
                mu_eff = 1 / np.sum(weights ** 2)

                mean_old = mean.copy()
                for i in range(mu):
                    mean += weights[i] * pop[i]

                # Update step size control variables
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / chi_n * (B @ (D * (mean - mean_old) / sigma))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals / lam))) < (1.4 + 2 / (n + 1)) * chi_n:
                    pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (mean - mean_old) / sigma

                # Update covariance matrix
                delta_d = (1 + 2 * max(0, np.sqrt(cc * (2 - cc) * mu_eff) - 1)) * cc
                outer = np.outer(pc, pc)
                C = (1 - c1 - cmu + delta_d) * C + c1 * (outer + (1 - cc) * (1 + delta_d - cc))
                for i in range(mu):
                    y = (mean - mean_old) / sigma - B @ (D * (pop[i] - mean_old) / sigma)
                    C += cmu * weights[i] * np.outer(y, y)

                # Eigen decomposition for B and D
                D2, B2 = np.linalg.eigh(C)
                D2 = np.sqrt(np.maximum(D2, 1e-12))
                B = B2
                D = D2

                # Sigma adaptation
                sigma *= np.exp(alpha_sigma / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Generate new population
                new_pop = []
                new_fvals = []
                for i in range(lam):
                    if call_count >= budget or time.time() > deadline:
                        return
                    z = rng.standard_normal(n)
                    x = mean + sigma * (B @ (D * z))
                    x = np.clip(x, -5, 5)
                    fv = call_func(x)
                    if fv is None:
                        return
                    new_pop.append(x)
                    new_fvals.append(fv)
                    evals += 1

                pop = new_pop
                fvals = new_fvals

        # Phase 1: Rastrigin-aware initialization - sample integer-aligned points
        # The global minimum of Rastrigin is at integer coordinates
        candidates = [np.zeros(dim)]
        
        # Add points with one coordinate at ±1, ±2
        for d in range(dim):
            for val in [1, -1, 2, -2]:
                pt = np.zeros(dim)
                pt[d] = val
                candidates.append(pt)
        
        # Add some mixed integer points
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for v1 in [1, -1]:
                    for v2 in [1, -1]:
                        pt = np.zeros(dim)
                        pt[d1] = v1
                        pt[d2] = v2
                        candidates.append(pt)

        # Evaluate initial candidates
        for pt in candidates:
            if call_count >= budget or time.time() > deadline:
                break
            pt_clipped = np.clip(pt, -5, 5)
            fx = call_func(pt_clipped)
            if fx is None:
                break

        # Phase 2: Run CMA-ES from the best point found
        if call_count < budget and time.time() < deadline:
            x0 = best_x.copy()
            sigma0 = 1.0  # Start with unit step size
            max_evals = min(300, budget - call_count)
            cma_es(x0, sigma0, max_evals=max_evals)

        # Phase 3: If CMA-ES didn't find a very good solution, try additional restarts
        if call_count < budget and time.time() < deadline and best_f > 5.0:
            # Try a few more random restarts with CMA-ES
            n_restarts = min(3, (budget - call_count) // 100)
            for _ in range(n_restarts):
                if call_count >= budget or time.time() > deadline:
                    break
                
                # Random restart near integer grid
                pt = np.array([rng.choice(range(-4, 5)) for _ in range(dim)], dtype=float)
                pt += rng.uniform(-0.5, 0.5, dim)
                pt = np.clip(pt, -5, 5)
                
                fx = call_func(pt)
                if fx is None:
                    break
                
                if fx < best_f:
                    # Run CMA-ES from this point with smaller budget
                    max_evals = min(100, (budget - call_count) // 2)
                    if max_evals > 20:
                        cma_es(pt.copy(), 0.5, max_evals=max_evals)

        # Phase 4: Fine-tuning with coordinate descent if we have budget left
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            # Try to snap to nearest integer in each dimension first
            for d in range(dim):
                if call_count >= budget or time.time() > deadline:
                    break
                
                best_coord_val = fx
                best_coord = x[d]
                
                # Search integer coordinates near current position
                for ic in range(int(x[d]) - 1, int(x[d]) + 2):
                    if call_count >= budget or time.time() > deadline:
                        break
                    pt = x.copy()
                    pt[d] = float(ic)
                    pt = np.clip(pt, -5, 5)
                    fx_test = call_func(pt)
                    if fx_test is None:
                        break
                    if fx_test < best_coord_val - 1e-10:
                        best_coord_val = fx_test
                        best_coord = float(ic)
                
                if best_coord_val < fx:
                    x[d] = best_coord
                    fx = best_coord_val
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

            # Now do fine-tuning around the integer-aligned point
            for step in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
                if call_count >= budget or time.time() > deadline:
                    break
                
                improved = True
                while improved and call_count < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if call_count >= budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_new = x.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5, 5)
                            fx_new = call_func(x_new)
                            if fx_new is None:
                                break
                            if fx_new < fx - 1e-12:
                                x = x_new
                                fx = fx_new
                                improved = True
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
