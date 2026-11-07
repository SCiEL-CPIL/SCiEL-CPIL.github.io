import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        time_limit = 295.0

        best_x = None
        best_f = float('inf')
        calls = 0

        def time_ok():
            return time.time() - start_time < time_limit and calls < budget

        # Phase 1: Coarse random search to find a good starting point
        # Use 15 calls for initial exploration
        coarse_calls = 15
        np.random.seed(42)
        for _ in range(coarse_calls):
            if not time_ok():
                break
            x = np.random.uniform(-5, 5, n)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1

        # Phase 2: Deterministic multi-start CMA-ES with progressively smaller sigma
        # For Sphere function, we can use a more aggressive approach
        # Since Sphere is convex and has a single minimum, we can use
        # a simpler optimizer that converges quickly
        
        # Use Nelder-Mead or a simple gradient-free method
        # For 5D Sphere, a simple approach works well
        
        # Let's use a combination: start with CMA-ES but with better initialization
        # and restart strategy
        
        # Multi-start with different initial sigmas
        sigmas = [3.0, 1.5, 0.5, 0.1]
        
        for sigma_init in sigmas:
            if not time_ok():
                break
            
            # Run CMA-ES with this initial sigma
            lambda_ = 10
            mu = 5
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_cov = (2 + 2 * mu_eff / n) / (n + 2 + 2 * mu_eff / n)
            
            mean = best_x.copy()
            sigma = sigma_init
            C = np.eye(n)
            C_sqrt = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            chi_n = np.sqrt(n) * (1 + 1 / (4 * n) + 1 / (20 * n ** 2))
            
            # Track stagnation
            prev_best = best_f
            stagnation_count = 0
            max_stagnation = 10  # generations without improvement
            
            while time_ok():
                # Generate population
                pop = np.array([mean + sigma * (C_sqrt @ np.random.randn(n)) for _ in range(lambda_)])
                pop = np.clip(pop, -5, 5)
                
                # Evaluate population
                f_pop = []
                for x in pop:
                    if not time_ok():
                        break
                    fx = func(x)
                    calls += 1
                    f_pop.append(fx)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                
                if not time_ok() or len(f_pop) < lambda_:
                    break
                
                f_pop = np.array(f_pop)
                order = np.argsort(f_pop)
                pop = pop[order]
                f_pop = f_pop[order]
                
                # Check for improvement
                if f_pop[0] < prev_best - 1e-10:
                    prev_best = f_pop[0]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                if stagnation_count >= max_stagnation:
                    break
                
                # Compute new mean
                new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)
                diff = new_mean - mean
                
                if np.linalg.norm(diff) / sigma < 1e-12:
                    break
                
                # Update p_sigma
                norm_C_sqrt = np.sqrt(np.trace(C))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ diff) / (sigma * norm_C_sqrt)
                
                h_sigma = 1 if np.linalg.norm(p_sigma) / chi_n < (1.4 + 1 / n) else 0
                
                # Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (1 - c_c)
                C = (1 + c_cov * (mu_eff - 2 + delta_h) - c_c) * C
                
                try:
                    C_inv_sqrt = np.linalg.inv(C_sqrt)
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)
                
                for i in range(mu):
                    y = C_inv_sqrt @ (pop[i] - mean) / sigma
                    C += c_cov * np.outer(y, y)
                
                C += c_cov * delta_h * np.outer(p_c, p_c)
                
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                # Compute C_sqrt via eigendecomposition
                C_sqrt = eigvecs @ (np.sqrt(eigvals)[:, None] * eigvecs.T)
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 5.0)
                
                # Update mean
                mean = np.clip(new_mean, -5, 5)
                
                if sigma < 1e-8:
                    break
        
        # Phase 3: Final local refinement with a simple coordinate search
        # Use remaining budget for fine-tuning
        if time_ok():
            remaining = budget - calls
            if remaining > 5:
                # Simple coordinate descent
                for _ in range(min(remaining, 20)):
                    if not time_ok():
                        break
                    improved = False
                    for i in range(n):
                        # Try small perturbations in each dimension
                        for delta in [0.01, -0.01, 0.001, -0.001]:
                            if not time_ok():
                                break
                            x_test = best_x.copy()
                            x_test[i] += delta
                            x_test = np.clip(x_test, -5, 5)
                            fx = func(x_test)
                            calls += 1
                            if fx < best_f - 1e-10:
                                best_f = fx
                                best_x = x_test.copy()
                                improved = True
                    if not improved:
                        break

        return best_x, best_f
