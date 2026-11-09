"""Differential Evolution with adaptive parameters and CMA-ES refinement for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                f_val = float(func(x_clipped))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val, x_clipped
            except Exception:
                return None, None

        rng = np.random.default_rng(42)

        # Phase 1: Initial sampling
        n_init = 15
        for _ in range(n_init):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            call_func(x)

        if best_x is None:
            x = np.zeros(dim)
            call_func(x)
            if best_x is None:
                return None

        # Phase 2: Differential Evolution (jDE)
        # Large population for global diversity
        lam = 20
        pop = np.zeros((lam, dim))
        fvals = np.zeros(lam)
        
        # Initialize population: best_x + random
        pop[0] = best_x
        fvals[0] = best_f
        for i in range(1, lam):
            x = rng.uniform(-5, 5, dim)
            f_val, x_eval = call_func(x)
            if f_val is not None:
                pop[i] = x_eval
                fvals[i] = f_val
            else:
                pop[i] = best_x
                fvals[i] = best_f

        # jDE parameters
        F = 0.5
        CR = 0.9
        F_min, F_max = 0.1, 0.9
        CR_min, CR_max = 0.1, 1.0
        
        generation = 0
        max_generations = 50
        stagnation_count = 0
        best_f_prev = best_f

        while n_calls < budget and not check_time() and generation < max_generations:
            generation += 1
            best_f_gen_start = best_f

            # Sort population
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            new_pop = pop.copy()
            new_fvals = fvals.copy()

            for i in range(lam):
                if check_time() or n_calls >= budget:
                    break
                
                # Adaptive parameters (jDE)
                F_i = F + rng.normal(0, 0.1)
                F_i = np.clip(F_i, F_min, F_max)
                CR_i = CR + rng.normal(0, 0.1)
                CR_i = np.clip(CR_i, CR_min, CR_max)

                # DE/rand/1 mutation
                idxs = rng.choice(lam, 3, replace=False)
                while i in idxs:
                    idxs = rng.choice(lam, 3, replace=False)
                a, b, c = idxs
                v = pop[a] + F_i * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Binomial crossover
                r = rng.random()
                j_rand = rng.integers(0, dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR_i or j == j_rand:
                        trial[j] = v[j]
                
                f_trial, x_trial = call_func(trial)
                if f_trial is None:
                    break

                # Selection
                if f_trial <= fvals[i]:
                    new_pop[i] = trial
                    new_fvals[i] = f_trial
                    # Update F and CR based on success
                    F = 0.5 * F + 0.5 * F_i
                    CR = 0.5 * CR + 0.5 * CR_i

            pop = new_pop
            fvals = new_fvals

            if best_f < best_f_prev:
                best_f_prev = best_f
                stagnation_count = 0
            else:
                stagnation_count += 1
                # If stagnating, try to diversify
                if stagnation_count > 10:
                    # Replace worst individuals with random ones
                    worst_idx = np.argsort(fvals)[-lam//4:]
                    for idx in worst_idx:
                        x = rng.uniform(-5, 5, dim)
                        f_val, x_eval = call_func(x)
                        if f_val is not None:
                            pop[idx] = x_eval
                            fvals[idx] = f_val
                    stagnation_count = 0
                    F = 0.8
                    CR = 0.5

        # Phase 3: CMA-ES Refinement
        if n_calls < budget and not check_time():
            n = dim
            lambda_pop = max(4, int(4 + 3 * np.log(n)))
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)

            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            x_mean = best_x.copy()
            sigma = 0.1  # Start with a smaller sigma for refinement
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            best_f_segment = best_f
            stagnation_count = 0

            while n_calls < budget and not check_time():
                # Generate population
                pop_cma = np.zeros((lambda_pop, n))
                fvals_cma = np.zeros(lambda_pop)
                valid_idx = 0
                
                while valid_idx < lambda_pop and n_calls < budget and not check_time():
                    z = rng.standard_normal(n)
                    try:
                        L = np.linalg.cholesky(C)
                        x_trial = x_mean + sigma * (L @ z)
                    except np.linalg.LinAlgError:
                        x_trial = x_mean + sigma * z
                    
                    f_val, x_eval = call_func(x_trial)
                    if f_val is None:
                        break
                    pop_cma[valid_idx] = x_eval
                    fvals_cma[valid_idx] = f_val
                    valid_idx += 1
                
                if valid_idx < lambda_pop:
                    break

                order = np.argsort(fvals_cma[:valid_idx])
                pop_cma = pop_cma[order]
                fvals_cma = fvals_cma[order]
                
                if fvals_cma[0] < best_f_segment:
                    best_f_segment = fvals_cma[0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    if stagnation_count > 15:
                        break # Stop refinement if stuck

                # Update CMA-ES state
                x_mean_new = np.zeros(n)
                for i in range(mu):
                    x_mean_new += weights[i] * pop_cma[i]
                
                try:
                    L = np.linalg.cholesky(C)
                    C_inv_sqrt = np.linalg.inv(L).T
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)

                y_mean = (x_mean_new - x_mean) / sigma
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (C_inv_sqrt @ y_mean)

                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (valid_idx + 1)))) < (1.4 + 2.0 / (n + 1)) * chi_n

                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_mean

                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                y_i = np.zeros((mu, n))
                for i in range(mu):
                    y_i[i] = (pop_cma[i] - x_mean) / sigma
                C += cmu * (weights @ (y_i @ y_i.T))
                C = (C + C.T) / 2.0

                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-10, 1.0)
                x_mean = x_mean_new

        return best_x, best_f
# EVOLVE-BLOCK-END
