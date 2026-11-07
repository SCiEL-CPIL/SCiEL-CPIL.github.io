"""
Integer-grid initialized CMA-ES for BBOB f3 (Rastrigin) on [-5, 5]^5.
Exploits Rastrigin's known periodic structure via integer-grid sampling,
then refines with CMA-ES to handle hidden rotations and shifts.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin

        n = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Integer-grid initialization exploiting Rastrigin structure
        # Rastrigin has local minima near integers; sample a structured grid
        # plus some random points to handle the hidden rotation/shift.
        
        init_points = []
        
        # Origin
        init_points.append(np.zeros(n))
        
        # Integer lattice points in [-4, 4]^n (sparse subset for budget)
        # Use a structured subset: corners and axis-aligned points
        for coord in range(-4, 5):
            for axis in range(n):
                if check_deadline():
                    break
                x = np.zeros(n)
                x[axis] = coord
                init_points.append(np.clip(x, -5, 5))
        
        # Random points to handle rotation
        n_random = 10
        for _ in range(n_random):
            if check_deadline():
                break
            x = rng.uniform(-4.5, 4.5, n)
            init_points.append(x)
        
        # Evaluate all initial points
        for x in init_points:
            if check_deadline():
                break
            x = np.clip(x, -5, 5)
            f = func(x)
            update_best(x, f)

        if best_x is None:
            x = np.zeros(n)
            f = func(x)
            update_best(x, f)
            return best_x, best_f

        # Phase 2: CMA-ES refinement
        def run_cmaes(x_start, f_start, max_calls, pop_size=None, sigma_init=None):
            nonlocal calls_made, best_x, best_f

            if max_calls < 10:
                return

            if pop_size is None:
                pop_size = 4 + int(3 * np.log(n))
                pop_size = max(4, min(pop_size, max_calls // 3))
            
            if pop_size < 2:
                return

            mu = max(1, pop_size // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)

            if sigma_init is None:
                sigma_init = 0.5
            sigma = sigma_init
            mean = x_start.copy().astype(float)

            C = np.eye(n)
            C_sqrt = np.eye(n)

            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (weights[1] - 1) / ((n + 2)**2 + mu_eff))

            ps = np.zeros(n)
            pc = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

            no_improve = 0
            last_best = f_start
            generation = 0

            while not check_deadline() and calls_made < budget:
                if calls_made + pop_size > budget:
                    remaining = budget - calls_made
                    if remaining < mu:
                        break
                    pop_size = remaining
                    mu = max(1, pop_size // 2)
                    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    weights /= np.sum(weights)
                    mu_eff = 1.0 / np.sum(weights**2)

                offspring = []
                for i in range(pop_size):
                    z = rng.standard_normal(n)
                    y = mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    offspring.append((fy, y))
                    if check_deadline():
                        break

                if len(offspring) < mu:
                    break

                offspring.sort(key=lambda t: t[0])
                gen_best = offspring[0][0]

                if gen_best < last_best - 1e-10:
                    last_best = gen_best
                    no_improve = 0
                else:
                    no_improve += 1

                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * offspring[i][1]

                old_mean = mean.copy()
                mean = y_w

                ys = (mean - old_mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * ys
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chi_n
                        < (1.4 + 2 / (n + 1)) * cc)

                if hsig:
                    pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * ys
                else:
                    pc = (1 - cs) * pc

                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    zi = (offspring[i][1] - old_mean) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)

                C = ((1 - c1 - cmu) * C
                     + c1 * (np.outer(pc, pc) + (1 - hsig) * C)
                     + cmu * rank_mu)

                C = (C + C.T) / 2.0

                try:
                    C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(n))
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_sqrt = np.eye(n)

                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-15)

                generation += 1

                if sigma < 1e-12:
                    break
                if no_improve > 25 and sigma < 0.01:
                    break
                if generation > 150:
                    break

            return best_x, best_f

        # Run CMA-ES from best initial point with moderate sigma
        run_cmaes(best_x, best_f, budget - calls_made, sigma_init=0.5)

        # Phase 3: Targeted restarts from integer-grid neighbors of best
        remaining = budget - calls_made
        restart_attempts = 0
        max_restarts = 8

        # Generate restart candidates from integer grid near current best
        restart_candidates = []
        best_rounded = np.round(best_x).astype(int)
        for delta in range(-1, 2):
            for axis in range(n):
                if check_deadline() or remaining < 30:
                    break
                x_cand = best_rounded.copy()
                x_cand[axis] += delta
                if np.all(x_cand >= -4) and np.all(x_cand <= 4):
                    restart_candidates.append(x_cand.astype(float))
        
        # Add some random restarts
        for _ in range(5):
            if check_deadline() or remaining < 30:
                break
            restart_candidates.append(rng.uniform(-4.5, 4.5, n))
        
        for x_restart in restart_candidates:
            if remaining < 30 or check_deadline():
                break
            x_restart = np.clip(x_restart, -5, 5)
            f_restart = func(x_restart)
            calls_made += 1
            remaining = budget - calls_made

            if f_restart < best_f:
                best_f = f_restart
                best_x = x_restart.copy()
            
            if remaining > 25:
                run_cmaes(best_x, best_f, remaining, sigma_init=0.3)
                remaining = budget - calls_made

            restart_attempts += 1
            if restart_attempts >= max_restarts:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
