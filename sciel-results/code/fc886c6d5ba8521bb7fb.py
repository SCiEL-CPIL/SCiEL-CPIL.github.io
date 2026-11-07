import numpy as np
import time

class Optimizer:
    """Hybrid DE-CMA-ES optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(12345)
        d = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls = 0

        def time_ok():
            return time.time() < deadline

        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Differential Evolution for Global Search
        # DE is robust for multimodal functions and doesn't require covariance estimation
        # which is slow with small populations in early stages.
        lambda_de = 10
        F = 0.8
        CR = 0.9
        budget_de = int(0.4 * budget)
        
        # Initialize DE population
        pop = [rng.uniform(-4.5, 4.5, d) for _ in range(lambda_de)]
        fvals = [0.0] * lambda_de
        
        for i in range(lambda_de):
            if not time_ok() or calls >= budget_de:
                break
            fvals[i] = float(func(pop[i]))
            calls += 1
            record(pop[i], fvals[i])

        # DE Main Loop
        while time_ok() and calls < budget_de:
            # Sort population by fitness
            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check if we can afford a full generation
            if calls + lambda_de > budget_de:
                break

            new_pop = [None] * lambda_de
            new_fvals = [0.0] * lambda_de
            
            for i in range(lambda_de):
                if not time_ok() or calls >= budget_de:
                    break
                
                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(lambda_de, 3, replace=False)
                
                # DE/Rand/1 mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Binary Crossover
                j_rand = rng.integers(0, d)
                u = np.empty(d)
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                # Clip to box
                u = np.clip(u, -5.0, 5.0)
                
                # Evaluate
                fu = float(func(u))
                calls += 1
                record(u, fu)
                
                # Selection
                if fu <= fvals[i]:
                    new_pop[i] = u
                    new_fvals[i] = fu
                else:
                    new_pop[i] = pop[i]
                    new_fvals[i] = fvals[i]

            if any(p is None for p in new_pop):
                break

            pop = new_pop
            fvals = new_fvals

        # Phase 2: CMA-ES for Local Refinement
        # CMA-ES adapts the covariance matrix to the local landscape,
        # providing fast convergence to high precision.
        
        if best_x is None:
            best_x = np.zeros(d)
            best_fx = float(func(best_x))
            calls += 1
            record(best_x, best_fx)

        # Initialize CMA-ES
        mean = best_x.copy()
        sigma = 0.5  # Start with a moderate step size for refinement
        C = np.eye(d)
        C_sqrt_inv = np.eye(d)
        ps = np.zeros(d)
        pc = np.zeros(d)
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        lambda_cma = max(4, int(4 + 3 * np.log(d)))
        mu = lambda_cma // 2
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        # Generate initial population for CMA-ES
        pops = []
        fvals = []
        for i in range(lambda_cma):
            if not time_ok() or calls >= budget:
                break
            z = rng.standard_normal(d)
            x_i = np.clip(mean + sigma * (C_sqrt_inv @ z), -5.0, 5.0)
            pops.append(x_i)
            fx_i = float(func(x_i))
            calls += 1
            record(x_i, fx_i)
            fvals.append(fx_i)

        if len(fvals) < lambda_cma:
            return best_x, best_fx

        order = np.argsort(fvals)
        sorted_pops = [pops[i] for i in order]
        sorted_fvals = [fvals[i] for i in order]

        # CMA-ES Main Loop
        while time_ok() and calls < budget:
            # Update mean
            new_mean = sum(w[i] * sorted_pops[i] for i in range(mu))
            
            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt_inv.T @ (new_mean - mean))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) / chi_d) < 1.4 + 2 / (d + 1)
            
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
            
            # Update covariance matrix
            delta_c = 0
            for i in range(mu):
                y_i = (sorted_pops[i] - mean) / sigma
                delta_c += w[i] * np.outer(y_i, y_i)
            
            C = (1 + c_1 * (hsig - 1) + c_mu) * C + c_1 * np.outer(pc, pc) - c_1 * C + c_mu * (delta_c - C)
            C = (C + C.T) / 2
            C += 1e-6 * np.eye(d)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_d - 1))
            sigma = float(np.clip(sigma, 1e-7, 5.0))
            
            # Update mean
            mean = new_mean
            
            # Generate next population
            pops = []
            fvals = []
            for i in range(lambda_cma):
                if not time_ok() or calls >= budget:
                    break
                z = rng.standard_normal(d)
                try:
                    L = np.linalg.cholesky(C)
                    x_i = np.clip(mean + sigma * (L @ z), -5.0, 5.0)
                except np.linalg.LinAlgError:
                    x_i = np.clip(mean + sigma * z, -5.0, 5.0)
                pops.append(x_i)
                fx_i = float(func(x_i))
                calls += 1
                record(x_i, fx_i)
                fvals.append(fx_i)

            if len(fvals) < lambda_cma:
                break

            order = np.argsort(fvals)
            sorted_pops = [pops[i] for i in order]
            sorted_fvals = [fvals[i] for i in order]

        return best_x, best_fx
