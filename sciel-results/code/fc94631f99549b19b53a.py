import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        d = self.dim
        total_budget = self.budget
        
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        n_calls = 0

        def run_cmaes(center, sigma_init, budget_limit):
            nonlocal n_calls, best_f, best_x
            
            lam = max(4, int(4 + 3 * np.log(d)))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w**2)

            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c_1 = (2 + 2 / (d + 1)) / ((d + 2)**2 + 2)
            c_mu = min(1 - c_1, (2 + (mu_eff - 2) / (d + 1)) / ((d + 2)**2 + mu_eff + 2))

            x = np.array(center, dtype=float)
            f_x = func(x)
            n_calls += 1
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()

            sigma = sigma_init
            C = np.eye(d)
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)
            E_norm = np.sqrt(d) * (1 - 1/(4*d) + 1/(32*d**2))
            
            stagnation = 0
            prev_best = best_f

            while n_calls < budget_limit:
                if time.time() > deadline:
                    return

                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                offspring = []
                for i in range(lam):
                    if n_calls >= budget_limit or time.time() > deadline:
                        break
                    z = rng.standard_normal(d)
                    y = x + sigma * (C_inv_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    f_y = func(y)
                    n_calls += 1
                    offspring.append((f_y, y, z))
                
                if len(offspring) < mu:
                    return

                offspring.sort(key=lambda t: t[0])
                
                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()
                    stagnation = 0
                else:
                    stagnation += 1
                    
                if stagnation > 30:
                    return

                z_w = np.zeros(d)
                for i in range(mu):
                    z_w += w[i] * offspring[i][2]
                
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ z_w)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / E_norm - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 1.0/d) * c_sigma * E_norm else 0.0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ z_w)
                
                C_temp = np.zeros((d, d))
                for i in range(mu):
                    z_i = offspring[i][2]
                    C_temp += w[i] * np.outer(z_i, z_i)
                
                C = ((1 - c_1 - c_mu + mu_eff * c_sigma * (1 - c_sigma)) * C +
                     c_1 * (np.outer(p_c, p_c) + (1 - c_sigma * (1 - c_sigma)) * np.eye(d)) +
                     c_mu * C_temp)
                C = (C + C.T) / 2
                
                x = x + sigma * (C_inv_sqrt @ z_w)
                x = np.clip(x, -5, 5)

        # Initial random search
        init_budget = min(30, total_budget)
        for _ in range(init_budget):
            if n_calls >= total_budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, d)
            f = func(x)
            n_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            n_calls += 1

        # Main CMA-ES with restarts
        remaining = total_budget - n_calls
        if remaining <= 0:
            return best_x, best_f

        # Run CMA-ES
        run_cmaes(best_x, 2.0, total_budget)
        
        # If we have budget left and haven't hit deadline, restart with larger sigma
        if n_calls < total_budget and time.time() < deadline:
            # Restart from best with larger sigma to escape local minima
            run_cmaes(best_x, 3.0, total_budget)
            
        if n_calls < total_budget and time.time() < deadline:
            # Another restart with even larger sigma or random center
            center = rng.uniform(-4, 4, d)
            run_cmaes(center, 4.0, total_budget)

        return best_x, best_f
