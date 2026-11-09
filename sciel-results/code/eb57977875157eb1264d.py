import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_y = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def call_func(x):
            nonlocal calls_made, best_x, best_y
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5, 5)
            y = func(x)
            calls_made += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Phase 1: Initial Sampling
        n_init = 5
        for _ in range(n_init):
            if not time_ok() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, d)
            call_func(x)
        
        if calls_made >= budget or not time_ok():
            return best_x, best_y

        # CMA-ES Implementation
        def cma_es(start_x, max_iter=200):
            nonlocal best_x, best_y, calls_made
            
            if best_x is None:
                return

            lambda_ = 4 + int(3 * np.log(d))
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            cs = (1 + 2 * min(1, (mu_eff - 1) / (d + 1)) + c_sigma) / (d + 1 + 3 * c_sigma)
            ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
            c1 = 2 / ((d + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * ((mu_eff + 2) / (d + 2) - 2))

            x_mean = start_x.copy()
            sigma = 1.0
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            chi_n = np.sqrt(d) * (1 + 1/(4*d) - 1/(21*d**2))
            
            # Initial Population
            pop = []
            fvals = []
            for i in range(lambda_):
                z = rng.normal(0, 1, d)
                x = x_mean + sigma * (C @ z)
                x = np.clip(x, -5, 5)
                y = call_func(x)
                if y is None:
                    return
                pop.append(x)
                fvals.append(y)
            
            pop = np.array(pop)
            fvals = np.array(fvals)
            
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            if fvals[0] < best_y:
                best_y = fvals[0]
                best_x = pop[0].copy()

            for _ in range(max_iter):
                if not time_ok() or calls_made >= budget:
                    break
                
                # Update Mean
                x_mean_old = x_mean
                x_mean = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
                
                # Update Step Size
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / np.linalg.norm(np.sqrt(C)) * (C @ (pop[0] - x_mean_old) / sigma)
                if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (d + 1)) * np.sqrt((1 - (1 - c_sigma)**2) / (1 - (1 - c_sigma)**(2 * (max_iter + 1)))):
                    sigma *= np.exp(cs * (np.linalg.norm(ps) / chi_n - 1) / ds)
                
                # Update Covariance
                pc = (1 - c_c) * pc + c_c * (x_mean - x_mean_old) / sigma
                delta_heav = 1 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**2) / chi_n < (1.4 + 2 / (d + 1)) else 0
                C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - (1 - c_c)**2) * delta_heav * np.eye(d)) + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(pop[:mu] - x_mean_old, pop[:mu] - x_mean_old) / (sigma**2), axis=0)
                
                # Generate New Population
                new_pop = []
                new_fvals = []
                for i in range(lambda_):
                    z = rng.normal(0, 1, d)
                    x = x_mean + sigma * (C @ z)
                    x = np.clip(x, -5, 5)
                    y = call_func(x)
                    if y is None:
                        return
                    new_pop.append(x)
                    new_fvals.append(y)
                
                pop = np.array(new_pop)
                fvals = np.array(new_fvals)
                
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]
                
                if fvals[0] < best_y:
                    best_y = fvals[0]
                    best_x = pop[0].copy()
                    
                # Stagnation check
                if sigma < 1e-8:
                    break

        # Phase 2: Main Loop with CMA-ES
        if best_x is not None and time_ok() and calls_made < budget:
            cma_es(best_x)
            
        # Restart Strategy
        while time_ok() and calls_made < budget:
            if best_x is not None:
                noise_scale = 0.5
                start_x = best_x + rng.normal(0, noise_scale, d)
                start_x = np.clip(start_x, -5, 5)
                cma_es(start_x)
            else:
                start_x = rng.uniform(-5, 5, d)
                cma_es(start_x)
                
        return best_x, best_y
