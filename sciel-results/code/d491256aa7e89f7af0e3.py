"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with paraboloid basin-jumping for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def fit_paraboloid(center, points, values):
            """Fit a paraboloid and return its minimum."""
            # points: (2n+1, n) array, values: (2n+1,) array
            # Fit f(x) = c + b^T(x-center) + (x-center)^T A (x-center)
            # Using 2n+1 points: center, center +/- h*e_i
            X = points - center  # (2n+1, n)
            # Linear terms: b_i = (f(center + h e_i) - f(center - h e_i)) / (2h)
            # Quadratic terms: A_ii = (f(center + h e_i) + f(center - h e_i) - 2*f(center)) / h^2
            h = 0.1
            f0 = values[0]
            b = np.zeros(n)
            A_diag = np.zeros(n)
            for i in range(n):
                b[i] = (values[1 + i] - values[1 + n + i]) / (2 * h)
                A_diag[i] = (values[1 + i] + values[1 + n + i] - 2 * f0) / (h * h)
            
            # Minimum of f(x) = c + b^T(x-c) + sum_i A_ii (x_i - c_i)^2
            # df/dx_i = b_i + 2*A_ii*(x_i - c_i) = 0
            # x_i - c_i = -b_i / (2*A_ii)
            x_min = center.copy()
            for i in range(n):
                if abs(A_diag[i]) > 1e-12:
                    x_min[i] = center[i] - b[i] / (2 * A_diag[i])
                else:
                    x_min[i] = center[i]
            
            # Clip to bounds
            x_min = np.clip(x_min, -5, 5)
            return x_min

        def do_cma_es(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with initial sigma0, using up to max_evals evaluations."""
            nonlocal calls_made, best_f, best_x
            
            if max_evals <= 0 or calls_made >= total_budget or time.time() >= deadline:
                return x0, best_f
            
            # CMA-ES parameters
            pop_size = 4 + int(np.floor(3 * np.log(n)))
            pop_size = min(pop_size, 12)
            lambda_ = pop_size
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1) - 1)) + c_sigma
            
            mean = np.array(x0, dtype=np.float64)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            
            # Generate population
            evals_used = 0
            max_iters = max_evals // lambda_
            
            for it in range(max_iters):
                if calls_made >= total_budget or time.time() >= deadline:
                    break
                
                # Generate population
                pop = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    x_i = mean + sigma * z @ np.linalg.cholesky(C).T
                    x_i = np.clip(x_i, -5, 5)
                    pop.append(x_i)
                
                # Evaluate population
                pop_f = np.array([evaluate(x) for x in pop])
                evals_used += lambda_
                
                # Sort by fitness
                order = np.argsort(pop_f)
                pop = [pop[i] for i in order]
                pop_f = pop_f[order]
                
                # Update mean
                new_mean = np.zeros(n)
                for i in range(mu):
                    new_mean += weights[i] * pop[i]
                
                # Update evolution paths
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
                sigma_old = sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (new_mean - mean) / sigma
                mean = new_mean
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
                
                # Update covariance
                delta = 1 if np.linalg.norm(ps) ** 2 / (1 - (1 - c_sigma) ** (2 * (it + 1))) < (2 + 2 / c_c) * chi_n ** 2 else 0
                C = (1 - c_1 - c_mu + c_1 * delta * (1 / c_c - 1) + c_mu) * C
                C += c_1 * (pc @ pc.T - (1 - c_c) * (1 - 2 * c_c) * np.eye(n))
                
                diff = [pop[i] - mean for i in range(mu)]
                C += c_mu * np.sum([np.outer(w, w) for w in diff], axis=0) / sigma ** 2
                # Simplified: use outer products
                C += c_mu * sum(weights[i] * np.outer(diff[i], diff[i]) for i in range(mu))
                
                # Symmetrize
                C = (C + C.T) / 2
                
                # Ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Adaptive sigma
                if it > 0 and sigma < 1e-10:
                    break
            
            return mean, best_f

        def do_paraboloid_jump(center):
            """Try to jump to a better basin using paraboloid fitting."""
            nonlocal calls_made, best_f, best_x
            
            if calls_made >= total_budget or time.time() >= deadline:
                return center, best_f
            
            h = 0.1
            # Evaluate center and 2n neighbors
            points = [center]
            values = [evaluate(center)]
            
            for i in range(n):
                for sign in [1, -1]:
                    pt = center.copy()
                    pt[i] += sign * h
                    pt = np.clip(pt, -5, 5)
                    values.append(evaluate(pt))
                    points.append(pt)
            
            if calls_made >= total_budget or time.time() >= deadline:
                return center, best_f
            
            points = np.array(points)
            values = np.array(values)
            
            # Check if we have a valid local minimum (center is lower than neighbors)
            if values[0] < np.min(values[1:]):
                # Local minimum, try to jump
                x_min = fit_paraboloid(center, points, values)
                if not np.allclose(x_min, center, atol=0.5):
                    # Evaluate the predicted minimum
                    f_min = evaluate(x_min)
                    if f_min < best_f:
                        best_f = f_min
                        best_x = x_min.copy()
                        return x_min, best_f
            else:
                # Not a local minimum, move toward lower neighbor
                idx = np.argmin(values[1:]) + 1
                return points[idx], best_f
            
            return center, best_f

        # Phase 1: Multi-start CMA-ES
        # Try several starting points
        start_points = []
        
        # Origin
        start_points.append(np.zeros(n))
        
        # Random starts
        for _ in range(3):
            start_points.append(rng.uniform(-3, 3, n))
        
        # Structured starts for Lunacek
        start_points.append(np.ones(n) * 2.0)
        start_points.append(np.ones(n) * -2.0)
        
        # Evaluate all starts
        for sp in start_points:
            if calls_made >= total_budget or time.time() >= deadline:
                break
            evaluate(np.clip(sp, -5, 5))
        
        # Sort starts by fitness and run CMA-ES from best ones
        # We'll run CMA-ES from the current best with adaptive sigma
        current_best_x = best_x.copy() if best_x is not None else np.zeros(n)
        current_best_f = best_f
        
        # Phase 2: CMA-ES with periodic paraboloid jumps
        sigma = 1.0
        cma_budget = 300  # evaluations for CMA-ES
        jump_interval = 40  # try paraboloid jump every 40 evals
        
        cma_start = calls_made
        while calls_made < total_budget and time.time() < deadline:
            # Run a chunk of CMA-ES
            chunk_evals = min(20, total_budget - calls_made)
            if chunk_evals <= 0:
                break
            
            # Run CMA-ES for this chunk
            x_new, f_new = do_cma_es(current_best_x, sigma, chunk_evals)
            
            if f_new < current_best_f:
                current_best_f = f_new
                current_best_x = x_new.copy()
                # Reduce sigma on improvement
                sigma *= 0.7
            else:
                # No improvement, try paraboloid jump
                x_jump, f_jump = do_paraboloid_jump(current_best_x)
                if f_jump < current_best_f:
                    current_best_f = f_jump
                    current_best_x = x_jump.copy()
                    sigma = 0.5
                else:
                    # Increase sigma to explore more
                    sigma *= 1.3
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-4, 3.0)
        
        # Phase 3: Final local refinement with adaptive step
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.05
            
            for _ in range(50):
                if calls_made >= total_budget or time.time() >= deadline:
                    break
                z = rng.standard_normal(n)
                y = np.clip(x + sigma_local * z, -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x = y
                    fx = fy
                    sigma_local *= 1.05
                else:
                    sigma_local *= 0.75
                sigma_local = max(sigma_local, 1e-7)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
