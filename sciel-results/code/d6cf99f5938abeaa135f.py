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
            if n_calls >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        # CMA-ES Parameters
        lambda_pop = 4 + int(np.floor(3 * np.log(dim)))
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1. / np.sum(weights ** 2)
        
        c1 = (4 + mu / (dim + 1.7)) / ((dim + 2) ** 2 + mu)
        cmu = min(1 - c1, 2 * (mu - 2 + 1. / mu) / ((dim + 2) ** 2 + mu))
        c_sigma = (mu + 2) / (dim + mu + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
        
        # Initialize
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.clip(D, 1e-12, None))
        
        # Run CMA-ES
        # Reserve some budget for final polish, e.g., 20 calls
        max_iter = (budget - 20) // lambda_pop
        
        for it in range(max_iter):
            if n_calls >= budget or time.time() > deadline:
                break

            # Sample population
            y = np.random.standard_normal((lambda_pop, dim))
            y = y @ B.T * D
            x_pop = x_mean + sigma * y
            
            # Evaluate
            f_pop = []
            for i in range(lambda_pop):
                f_pop.append(timed_call(x_pop[i]))
                
            if len(f_pop) < lambda_pop:
                break
                
            f_pop = np.array(f_pop)
            order = np.argsort(f_pop)
            x_pop = x_pop[order]
            f_pop = f_pop[order]
            y = y[order]
            
            # Update best
            if f_pop[0] < best_f:
                best_f = f_pop[0]
                best_x = x_pop[0].copy()
                
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = x_mean + sigma * (weights[:mu] @ x_pop[:mu])
            
            # Update covariance matrix
            delta_x = (x_mean - x_mean_old) / sigma
            pc = (1 - c1) * pc + c1 * delta_x
            
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c1) ** (it + 1)) < (1 + 2 / (dim + 1)) * chi_n)
            
            ps = (1 - c_sigma) * ps + float(hsig) * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_x
            
            C = (1 + c1 * cmu * hsig * (hsig - 1) - c1) * C + cmu * np.outer(y[:mu].T @ weights[:mu], y[:mu].T @ weights[:mu])
            
            # Ensure positive definiteness
            C = (C + C.T) / 2
            
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D, 1e-12, None))
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Stopping criteria
            if sigma < 1e-10:
                break

        # Final Polish with Nelder-Mead
        if n_calls < budget and best_x is not None and time.time() < deadline:
            def nm_func(x):
                return timed_call(x)
            
            # Use a simple Nelder-Mead implementation or scipy
            # Since scipy is available, let's use it for the polish
            # But we must be careful with the budget.
            # Nelder-Mead with 50 iterations in 5D costs ~ 50 * (dim+1) = 300 calls.
            # We have ~20 calls left. So we can only do a few iterations.
            # Let's do a manual coordinate descent or just a few NM steps.
            
            # Simple coordinate descent polish
            x_curr = best_x.copy()
            f_curr = best_f
            step_size = 0.5
            for _ in range(10): # Max 10 * 5 * 2 = 100 calls, but we have less
                if n_calls >= budget - 5 or time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if n_calls >= budget - 5 or time.time() > deadline:
                        break
                    for direction in [-1, 1]:
                        x_try = x_curr.copy()
                        x_try[i] += direction * step_size
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = timed_call(x_try)
                        if f_try < f_curr:
                            x_curr = x_try
                            f_curr = f_try
                            improved = True
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-4:
                        break

            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr

        return best_x, best_f
