# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Global Exploration ---
        # Mix of random and structured points to find a good basin quickly.
        # Structured points help if the shift is small or aligned with axes.
        n_global = 60
        
        # Generate a set of diverse initial points
        init_points = []
        # 1. Center
        init_points.append(np.zeros(d))
        # 2. Random points
        for _ in range(n_global - 1):
            init_points.append(rng.uniform(-5, 5, d))
        
        # Evaluate initial points
        for x in init_points:
            if check_time() or used_calls >= budget:
                break
            x = np.clip(x, -5, 5)
            f = func(x)
            used_calls += 1
            try_update(x, f)

        if best_x is None:
            return np.zeros(d), 0.0

        # --- Phase 2: CMA-ES with Adaptive Restarts ---
        # We run CMA-ES multiple times. If it stagnates, we restart from the best point
        # with a fresh covariance and moderate sigma.
        
        lambda_ = 8  # Population size
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # CMA-ES parameters for d=5
        cc = (4 + d / 4) / (d + 4 + 2 * d / 4)
        cs = cc
        c1 = 1 / (d + 4)
        cmu = 1 / (d + 8)
        damps = 1 + 2 * max(0, np.sqrt((d - 1) / (d + 3)) - 1) + c1
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))

        def run_cma_es(x0, sigma0, max_generations):
            nonlocal used_calls, best_f, best_x
            
            x = x0.copy()
            sigma = sigma0
            C = np.eye(d)
            C_sqrt = np.eye(d)
            pc = np.zeros(d)
            
            no_improve = 0
            stagnation_threshold = 15
            
            for gen in range(max_generations):
                if check_time() or used_calls >= budget:
                    break
                
                # Generate population
                pop = []
                for i in range(lambda_):
                    if used_calls >= budget or check_time():
                        break
                    z = rng.standard_normal(d)
                    y = C_sqrt @ z
                    xi = x + sigma * y
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    used_calls += 1
                    pop.append((xi, fi))

                if len(pop) < lambda_:
                    break

                # Sort
                pop.sort(key=lambda t: t[1])
                
                # Check improvement
                prev_best = best_f
                if pop[0][1] < best_f:
                    best_f = pop[0][1]
                    best_x = pop[0][0].copy()
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Check for stagnation
                if no_improve >= stagnation_threshold:
                    break
                
                # Compute mean of top mu
                x_new = np.zeros(d)
                for i in range(mu):
                    x_new += weights[i] * pop[i][0]

                # Update step size
                y_mean = (x_new - x) / sigma if sigma > 0 else np.zeros(d)
                sigma *= np.exp(cs / damps * (np.linalg.norm(y_mean) / chiN - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

                # Update path
                pc = (1 - cs) * pc + cs * y_mean
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs)**2) / chiN < 1.4 + 2 / (d + 1))

                # Update covariance
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cs * (1 - cs) * C)
                for i in range(mu):
                    yi = (pop[i][0] - x) / sigma
                    C += cmu * weights[i] * np.outer(yi, yi)

                x = x_new

                # Recompute C_sqrt
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                except:
                    C_sqrt = np.eye(d)
                    C = np.eye(d)

                # Convergence check
                if sigma < 1e-8:
                    break
                if np.all(np.abs(x_new - x) < 1e-8):
                    break

        # Run CMA-ES multiple times if budget allows
        # Each run starts from the current best, with a potentially larger sigma if previous one collapsed
        sigma_init = 1.0
        while used_calls < budget - 50 and not check_time():
            # Estimate max generations allowed
            remaining = budget - used_calls
            max_gens = min(50, remaining // lambda_)
            
            if max_gens < 5:
                break
                
            run_cma_es(best_x, sigma_init, max_gens)
            
            # If we stopped due to stagnation or convergence, try a restart with a slightly larger sigma
            # to potentially escape local minima
            sigma_init = max(0.5, sigma_init * 0.8)
            if sigma_init < 0.1:
                sigma_init = 0.1

        # --- Phase 3: Local Polish with (1+1)-ES ---
        if best_x is not None and used_calls < budget and not check_time():
            x = best_x.copy()
            f = best_f
            sigma = 0.01
            for _ in range(50):
                if check_time() or used_calls >= budget:
                    break
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                used_calls += 1
                if fy < f:
                    x = y
                    f = fy
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                sigma = float(np.clip(sigma, 1e-6, 5.0))
                if sigma < 1e-5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
