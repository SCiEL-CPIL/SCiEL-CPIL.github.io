"""Optimizer for BBOB f19 (Griewank-Rosenbrock): CMA-ES with adaptive restarts and Nelder-Mead polish."""
import numpy as np
import time
import math

class Optimizer:
    """CMA-ES based optimizer with multi-start, adaptive restarts, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if time.time() >= deadline:
                return None
            try:
                fx = func(np.clip(x, -5, 5))
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                calls_made += 1
                return None

        rng = np.random.default_rng(42)

        # Nelder-Mead implementation for local polish
        def nelder_mead(x0, max_calls):
            nonlocal calls_made, best_x, best_f, budget

            if calls_made >= budget or time.time() >= deadline:
                return

            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(dim):
                x_temp = x0.copy()
                if x_temp[i] == 0:
                    x_temp[i] = 0.05
                else:
                    x_temp[i] += 0.05
                simplex.append(x_temp)
            
            f_simplex = []
            for i in range(dim + 1):
                if calls_made >= budget or time.time() >= deadline:
                    return
                fx = safe_eval(simplex[i])
                if fx is None:
                    return
                f_simplex.append(fx)
            
            # Main loop
            max_iter = max_calls // (dim + 1) + 5
            for k in range(max_iter):
                if calls_made >= budget or time.time() >= deadline:
                    break
                
                # Sort simplex by function value
                indices = np.argsort(f_simplex)
                simplex = [simplex[i] for i in indices]
                f_simplex = [f_simplex[i] for i in indices]
                
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                x_best = simplex[0]
                f_best = f_simplex[0]
                
                # Check for convergence
                if f_worst - f_best < 1e-12:
                    break
                
                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + 1.0 * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_eval(x_r)
                if f_r is None:
                    break
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + 2.0 * (centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_eval(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_eval(x_c)
                        if f_c is None:
                            break
                        if f_c <= min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_s = safe_eval(simplex[i])
                                if f_s is None:
                                    return
                                f_simplex[i] = f_s
                            break
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (centroid - x_worst)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_eval(x_c)
                        if f_c is None:
                            break
                        if f_c <= min(f_r, f_worst):
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_s = safe_eval(simplex[i])
                                if f_s is None:
                                    return
                                f_simplex[i] = f_s
                            break

            return

        # CMA-ES implementation
        def cma_es(x0, sigma0, max_calls):
            nonlocal calls_made, best_x, best_f, budget

            if calls_made >= budget or time.time() >= deadline:
                return

            # Parameters
            mu = max(1, int(np.floor(3 * dim / 2)))  # Number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # State variables
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            # Cholesky factor of C
            try:
                B = np.linalg.cholesky(C)
            except:
                B = np.eye(dim)
            D = np.sqrt(np.diag(B))
            B_inv = np.linalg.inv(B)

            # Initial population
            pop = []
            pop_f = []
            for i in range(mu * 2):
                if calls_made >= budget or time.time() >= deadline:
                    return
                z = rng.standard_normal(dim)
                x = mean + sigma * (B @ z)
                fx = safe_eval(x)
                if fx is None:
                    return
                pop.append(x)
                pop_f.append(fx)

            # Sort
            indices = np.argsort(pop_f)
            pop = [pop[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]

            # Main loop
            stagnation_count = 0
            best_f_local = pop_f[0]

            while calls_made < budget and time.time() < deadline:
                # Update mean
                new_mean = np.sum([weights[i] * pop[i] for i in range(mu)], axis=0)
                
                # Update step size
                old_mean = mean
                mean = new_mean
                
                # Update ps
                ps = (1 - c_sigma) * ps + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B_inv @ (mean - old_mean))
                if np.linalg.norm(ps) / math.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // (2 * mu)))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                    sigma *= math.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / math.sqrt(dim) - 1))

                # Update pc
                pc = (1 - c_c) * pc + math.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

                # Update covariance matrix
                delta_sigma = 1 if np.linalg.norm(ps) / math.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // (2 * mu)))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim) else 0
                C = (1 + c_1 * delta_sigma - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * np.sum([weights[i] * np.outer(pop[i] - old_mean, pop[i] - old_mean) for i in range(mu)], axis=0)
                
                # Symmetrize
                C = (C + C.T) / 2

                # Update B and D
                try:
                    B = np.linalg.cholesky(C)
                    D = np.sqrt(np.diag(B))
                    B_inv = np.linalg.inv(B)
                except:
                    # Fallback if C becomes non-positive definite
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    B_inv = np.eye(dim)

                # Generate new population
                pop = []
                pop_f = []
                for i in range(mu * 2):
                    if calls_made >= budget or time.time() >= deadline:
                        return
                    z = rng.standard_normal(dim)
                    x = mean + sigma * (B @ z)
                    fx = safe_eval(x)
                    if fx is None:
                        return
                    pop.append(x)
                    pop_f.append(fx)

                # Sort
                indices = np.argsort(pop_f)
                pop = [pop[i] for i in indices]
                pop_f = [pop_f[i] for i in indices]

                # Check for stagnation
                if pop_f[0] < best_f_local - 1e-10:
                    best_f_local = pop_f[0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1

                # Check for convergence
                if np.linalg.norm(np.array(pop) - mean, axis=1).max() < 1e-10 * sigma or sigma < 1e-10:
                    break

                # Adaptive restart logic: if stagnated for too long, break out to restart
                if stagnation_count > 20:
                    break

            return

        # Phase 1: Initial Exploration
        # Start with a few random points to find a good starting point for CMA-ES
        n_init = 10
        init_points = [np.zeros(dim)]
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4, 4, dim))
        
        for x0 in init_points:
            if calls_made >= budget or time.time() >= deadline:
                break
            fx = safe_eval(x0)
            if fx is None:
                break

        # Phase 2: CMA-ES with Restarts
        # Use most of the budget for CMA-ES
        cma_budget = int(budget * 0.7)
        
        # Run CMA-ES
        cma_es(best_x, 2.0, cma_budget)

        # If we still have budget, try another restart from the best point found
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 10:
                # Slightly perturb the best point
                x0 = best_x + rng.standard_normal(dim) * 0.1
                x0 = np.clip(x0, -5, 5)
                cma_es(x0, 1.0, remaining // 2)

        # Phase 3: Nelder-Mead Polish
        # Use remaining budget for local polish
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 10:
                nm_budget = min(remaining, max(20, int(remaining * 0.5)))
                nelder_mead(best_x, nm_budget)
            
            # If budget still remains, do another polish from a perturbed best
            if calls_made < budget and time.time() < deadline:
                x0 = best_x + rng.standard_normal(dim) * 0.1
                x0 = np.clip(x0, -5, 5)
                remaining = budget - calls_made
                if remaining > 10:
                    nm_budget = min(remaining, 20)
                    nelder_mead(x0, nm_budget)

        return best_x, best_f
