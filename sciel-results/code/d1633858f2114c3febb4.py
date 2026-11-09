import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            if calls >= budget:
                return np.inf
            calls += 1
            try:
                f_val = float(func(x))
            except:
                f_val = np.inf
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Latin Hypercube global search (60 evals)
        # LHS gives better uniform coverage than a 3^5 lattice
        n_lhs = 60
        rng = np.random.RandomState(42)
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Divide [−5, 5] into n_lhs intervals, pick one random point per interval
            intervals = np.linspace(-5.0, 5.0, n_lhs + 1)
            for i in range(n_lhs):
                lhs_points[i, d] = rng.uniform(intervals[i], intervals[i + 1])
        # Add origin and a few structured points
        extra = [np.zeros(dim), np.ones(dim)*3.0, -np.ones(dim)*3.0,
                 np.array([5.0]*dim), np.array([-5.0]*dim)]
        all_global = np.vstack([lhs_points] + [e.reshape(1,-1) for e in extra])
        
        for x in all_global:
            if calls >= budget or time.time() > deadline:
                break
            evaluate(x)

        if best_f == np.inf:
            evaluate(np.zeros(dim))

        # Phase 2: CMA-ES with restarts on remaining budget
        remaining = budget - calls
        if remaining < 10:
            return best_x, best_f

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # 11 for dim=5
        lam = min(lam, max(4, remaining // 3))
        mu = lam // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
        c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
        c_1 = 2.0 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0)**2 + mu_eff))

        chi_n = np.sqrt(dim) * (1.0 + 1.0/(4.0*dim) - 1.0/(20.0*dim*dim))

        # CMA-ES loop with stagnation restart
        x_mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        gen = 0
        stagnation_count = 0
        best_in_cma = best_f
        max_stagnation = 15  # generations without improvement before restart

        while calls < budget and time.time() < deadline:
            # Check if we have enough budget for a full generation
            if budget - calls < lam:
                # Do a partial generation or single-point refinement
                # Simple (1+1) step for remaining budget
                while calls < budget and time.time() < deadline:
                    step = sigma * np.random.randn(dim)
                    y = x_mean + step
                    f_y = evaluate(y)
                    if np.isfinite(f_y) and f_y < best_f:
                        x_mean = y.copy()
                        best_in_cma = f_y
                        stagnation_count = 0
                    else:
                        # Try a smaller step
                        step2 = sigma * 0.5 * np.random.randn(dim)
                        y2 = x_mean + step2
                        f_y2 = evaluate(y2)
                        if np.isfinite(f_y2) and f_y2 < best_f:
                            x_mean = y2.copy()
                            best_in_cma = f_y2
                            stagnation_count = 0
                            sigma *= 0.7
                        else:
                            sigma *= 0.7
                    if sigma < 1e-8:
                        break
                break

            # Generate population
            population = []
            for i in range(lam):
                z = np.random.randn(dim)
                y = x_mean + sigma * (B @ (D * z))
                population.append(y)

            # Evaluate
            f_values = np.array([evaluate(y) for y in population])
            
            # Handle any inf values
            finite_mask = np.isfinite(f_values)
            if not np.any(finite_mask):
                break

            # Sort by fitness (inf goes to end)
            order = np.argsort(f_values)
            selected_idx = order[:mu]
            selected = [population[i] for i in selected_idx]
            selected_f = f_values[selected_idx]

            # Update mean
            x_mean_new = np.sum(np.array(selected) * weights[:, None], axis=0)

            # Update evolution path for covariance matrix
            y_sigma = (x_mean_new - x_mean) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma

            # Update evolution path for step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ y_sigma)

            # Update step size
            hsig = (1.4 + 2.0 / (dim + 1.0)) * np.sqrt(
                (dim + 1.0) / (dim - 1.0) * (1.0 - (1.0 - c_sigma)**2 * (budget - calls))
            )
            if np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - c_sigma)**(2.0 * (budget - calls))) < hsig:
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / chi_n - 1.0))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance matrix
            delta_sig = (1.0 + c_sigma * (np.linalg.norm(ps)**2 / chi_n**2 - 1.0)) if (np.linalg.norm(ps) / chi_n > hsig) else 1.0
            rank_one = np.outer(y_sigma, y_sigma)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                vi = (selected[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(vi, vi)

            C = (1.0 + c_1 * c_mu * delta_sig - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2.0 + 1e-12 * np.eye(dim)

            x_mean = x_mean_new
            gen += 1

            # Check for improvement
            if best_f < best_in_cma - 1e-10:
                best_in_cma = best_f
                stagnation_count = 0
            else:
                stagnation_count += 1

            # Restart if stagnating
            if stagnation_count >= max_stagnation and budget - calls > 2 * lam:
                # Restart from best with fresh sigma
                x_mean = best_x.copy()
                sigma = 1.0
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                stagnation_count = 0
                best_in_cma = best_f
                gen = 0
                # Small perturbation to escape local minimum
                x_mean = x_mean + 0.5 * np.random.randn(dim)
                x_mean = np.clip(x_mean, -5.0, 5.0)
                continue

            # Eigen decomposition (every 5 generations for speed, but dim=5 is cheap)
            if gen % 3 == 0:
                D, B = np.linalg.eigh(C)
                D = np.maximum(D, 1e-10)

        return best_x, best_f
