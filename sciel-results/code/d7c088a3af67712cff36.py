# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= self.budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        dim = self.dim
        budget = self.budget

        # Phase 1: Random Initialization
        # Use 50 random points to find a good starting basin
        n_init = 50
        init_points = rng.uniform(-4.0, 4.0, size=(n_init, dim))
        
        best_init_x = None
        best_init_f = np.inf
        
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            fx = evaluate(init_points[i])
            if fx is None:
                break
            if fx < best_init_f:
                best_init_f = fx
                best_init_x = init_points[i].copy()

        if best_init_x is None:
            return None, np.inf

        best_x = best_init_x.copy()
        best_f = best_init_f

        # Phase 2: CMA-ES with Restarts
        lambda_ = 10
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        chi_N = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        max_restarts = 10
        current_restart = 0
        stall_threshold = 10 # generations without improvement
        
        while calls < budget and time.time() < deadline:
            # Initialize CMA-ES state
            if current_restart == 0:
                x = best_init_x.copy()
                sigma = 1.0
            else:
                # Restart strategy: random point or perturbation of best
                if current_restart % 2 == 0:
                    x = best_x + rng.normal(0, 2.0, dim)
                else:
                    x = rng.uniform(-4.0, 4.0, dim)
                x = np.clip(x, -5.0, 5.0)
                sigma = 1.0
            
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            stall_count = 0
            best_f_cma = np.inf
            
            # Run CMA-ES until budget/deadline/stall
            while calls < budget and time.time() < deadline:
                # Generate population
                offspring = []
                fitnesses = []
                
                for i in range(lambda_):
                    if calls >= budget or time.time() > deadline:
                        break
                    y = rng.standard_normal(dim)
                    z = C_sqrt @ y
                    xi = x + sigma * z
                    xi = np.clip(xi, -5.0, 5.0)
                    fx = evaluate(xi)
                    if fx is None:
                        break
                    offspring.append(xi)
                    fitnesses.append(fx)
                    
                    if fx < best_f_cma:
                        best_f_cma = fx
                    if fx < best_f:
                        best_f = fx
                        best_x = xi.copy()
                
                if len(offspring) < lambda_:
                    break
                
                # Sort
                indices = np.argsort(fitnesses)
                sorted_offspring = [offspring[i] for i in indices]
                sorted_fitnesses = [fitnesses[i] for i in indices]
                
                # Stagnation check
                if sorted_fitnesses[0] < best_f_cma - 1e-10:
                    stall_count = 0
                else:
                    stall_count += 1
                
                # Update mean
                x_new = np.sum([weights[i] * sorted_offspring[i] for i in range(mu)], axis=0)
                
                # Update sigma
                y_c = (x_new - x) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ y_c
                h_sigma = True
                if h_sigma:
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
                
                # Update Covariance
                if h_sigma:
                    delta_h = (1 - h_sigma) * (1 - c_c) - 1
                    rank_1 = np.outer(p_c, p_c)
                    rank_mu = np.sum([weights[i] * np.outer(sorted_offspring[i] - x, sorted_offspring[i] - x) for i in range(mu)], axis=0)
                    C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
                else:
                    rank_mu = np.sum([weights[i] * np.outer(sorted_offspring[i] - x, sorted_offspring[i] - x) for i in range(mu)], axis=0)
                    C = (1 - c_1 - c_mu) * C + c_mu * rank_mu
                
                # Update square roots
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-12, None)
                C_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues))
                C_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues))
                
                x = x_new
                
                # Termination: sigma too small or stalled
                if sigma < 1e-6 or stall_count >= stall_threshold:
                    break

            current_restart += 1
            if current_restart > max_restarts:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
