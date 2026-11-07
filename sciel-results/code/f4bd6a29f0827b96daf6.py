import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_f = float('inf')

        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        # 1. Random Initialization
        n_init = 20
        init_points = []
        for _ in range(n_init):
            if n_calls >= budget:
                break
            x = np.random.uniform(-5, 5, dim)
            f = timed_call(x)
            init_points.append((f, x.copy()))

        if n_calls >= budget:
            return best_x, best_f

        # Sort by fitness
        init_points.sort(key=lambda x: x[0])
        
        # 2. CMA-ES with restarts
        # We will run a sequence of CMA-ES runs.
        # Each run starts from the best point found so far (or a random point if it's the first run after init).
        # We use a stagnation-based restart: if no improvement for a certain number of iterations, we restart.
        
        def run_cma_es(x_start, sigma_start, max_iter):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or time.time() > deadline:
                return x_start, best_f

            # CMA-ES parameters
            N = 4 + int(np.floor(3 * np.log(dim)))  # population size
            c1 = (4 + mu / (dim + 1.7)) / ((dim + 2) ** 2 + mu)
            cmu = min(1 - c1, 2 * (mu - 2 + 1. / mu) / ((dim + 2) ** 2 + mu))
            c_sigma = (mu + 2) / (dim + mu + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
            c_cs = (4 + dim / c_sigma) / (dim + 4 + 2 * c_sigma / c_sigma)
            c_h = (1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1)) / (dim + 3)
            h_sig = 1.4 + 2 / (dim + 1)
            h = h_sig * chi_n

            mu = N // 2  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1. / np.sum(weights ** 2)

            # State
            x_mean = x_start.copy()
            sigma = sigma_start
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            stepsize_history = []

            best_f_local = best_f
            best_x_local = x_start.copy()
            no_improvement = 0

            for it in range(max_iter):
                if n_calls >= budget or time.time() > deadline:
                    break

                # Generate population
                y = np.zeros((N, dim))
                for i in range(N):
                    z = np.random.standard_normal(dim)
                    y[i] = B @ (D * z)

                # Evaluate population
                x_pop = x_mean + sigma * y
                f_pop = []
                for i in range(N):
                    if n_calls >= budget or time.time() > deadline:
                        break
                    f_pop.append(timed_call(x_pop[i]))
                
                if len(f_pop) < N:
                    break

                # Sort by fitness
                order = np.argsort(f_pop)
                x_pop = x_pop[order]
                f_pop = np.array(f_pop)[order]

                # Update best
                if f_pop[0] < best_f_local:
                    best_f_local = f_pop[0]
                    best_x_local = x_pop[0].copy()
                    no_improvement = 0
                else:
                    no_improvement += 1

                # Restart condition: if no improvement for 15 iterations, break
                if no_improvement > 15:
                    break

                # Update mean
                x_mean_new = x_mean + sigma * (weights[:mu] @ x_pop[:mu])
                x_mean = x_mean_new

                # Update covariance matrix
                y_old = y[order]
                y_w = y_old[:mu]
                delta_x = (x_mean_new - x_mean) / sigma
                pc = (1 - c1) * pc + c1 * delta_x
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c1) ** (it + 1)) < (1 + 2 / (dim + 1)) * h)
                ps = (1 - c_sigma) * ps + float(hsig) * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_x
                C = (1 + c1 * cmu * hsig * (hsig - 1) - c1) * C + cmu * np.outer(y_w.T @ weights[:mu], y_w.T @ weights[:mu])
                
                # Update eigen decomposition
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.clip(D, 1e-12, None))

                # Update step size
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

            return best_x_local, best_f_local

        # Execution Plan:
        # 1. 20 Random Calls.
        # 2. Run CMA-ES from best point.
        # 3. If budget remains, run another CMA-ES from the new best point with a larger sigma (restart).
        
        x_start = init_points[0][1].copy()
        
        # First CMA-ES run
        x_best, f_best = run_cma_es(x_start, 1.0, 200)
        
        # Second CMA-ES run (restart) if budget allows
        if n_calls < budget and time.time() < deadline:
            # Use a larger sigma to explore more broadly
            x_best, f_best = run_cma_es(x_best, 2.0, 200)
            
        # Final polish with a small sigma
        if n_calls < budget and time.time() < deadline:
            x_best, f_best = run_cma_es(x_best, 0.1, 50)

        return best_x, best_f
