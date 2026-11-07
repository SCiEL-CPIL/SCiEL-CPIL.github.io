import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global exploration with structured + random sampling
        # Büche-Rastrigin has many local minima; we need broad coverage
        
        n_global = 50  # Use 50 calls for global phase
        
        # Add origin
        f0 = call_func(np.zeros(dim))
        if f0 is None:
            return best_x, best_f

        # Add axis-aligned points
        for i in range(dim):
            for sign in [-1, 1]:
                x = np.zeros(dim)
                x[i] = sign * 3.0
                f = call_func(x)
                if f is None:
                    break
            if f is None:
                break
        if calls_made >= budget or not time_ok():
            return best_x, best_f

        # Random sampling in [-4, 4]^5
        for _ in range(n_global - 1 - 2 * dim):
            if calls_made >= budget or not time_ok():
                break
            x = rng.uniform(-4.0, 4.0, dim)
            f = call_func(x)
            if f is None:
                break

        if calls_made >= budget or not time_ok():
            return best_x, best_f

        # Phase 2: CMA-ES with adaptive restarts
        # CMA-ES is a powerful derivative-free optimizer that adapts the
        # covariance matrix to the problem's geometry, which is crucial
        # for rotated/shifted functions like Büche-Rastrigin.
        
        remaining = budget - calls_made
        n_restarts = 8
        budget_per_restart = max(20, remaining // n_restarts)
        
        for restart in range(n_restarts):
            if calls_made >= budget or not time_ok():
                break
            
            # Initialize this restart
            if restart == 0:
                # Start from best found so far
                x_mean = best_x.copy()
                lambda_ = 10  # population size
                sigma = 1.0
            else:
                # Mix: sometimes perturb best, sometimes sample fresh
                if rng.random() < 0.3:
                    x_mean = best_x + rng.normal(0, 1.0, dim)
                else:
                    x_mean = rng.uniform(-4.0, 4.0, dim)
                x_mean = np.clip(x_mean, -5.0, 5.0)
                lambda_ = 10
                sigma = 0.5
            
            # CMA-ES parameters
            cc = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
            cs = (lambda_ + 2 + 2 / (dim + lambda_)) / (dim + lambda_ + 5 + 2 * lambda_ / dim)
            cmu = min(0.5 - cc, 2 * ((lambda_ - 2 + 1. / lambda_) / (dim + 2.)) ** 2 - 1)
            damps = 1 + 2 * max(0, np.sqrt((lambda_ - 1) / (dim + 1)) - 1) + cs
            
            # State variables
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            C_sqrt = np.eye(dim)  # Cholesky factor
            inv_C_sqrt = np.eye(dim)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            local_best_f = float('inf')
            local_best_x = None
            stagnation_count = 0
            stagnation_limit = 30
            
            # Main CMA-ES loop
            while calls_made < budget and time_ok():
                # Generate population
                offspring = []
                offspring_f = []
                
                for i in range(lambda_):
                    z = rng.standard_normal(dim)
                    x_off = x_mean + sigma * C_sqrt @ z
                    x_off = np.clip(x_off, -5.0, 5.0)
                    f_off = call_func(x_off)
                    if f_off is None:
                        break
                    offspring.append(x_off)
                    offspring_f.append(f_off)
                
                if len(offspring) == 0:
                    break
                
                # Sort by fitness
                sorted_idx = np.argsort(offspring_f)
                offspring = [offspring[i] for i in sorted_idx]
                offspring_f = [offspring_f[i] for i in sorted_idx]
                
                # Update local best
                if offspring_f[0] < local_best_f:
                    local_best_f = offspring_f[0]
                    local_best_x = offspring[0].copy()
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                # Update mean using best mu individuals
                mu = lambda_ // 2
                w = np.log(mu + 1) - np.log(1 + np.arange(mu))
                w = w / np.sum(w)
                mu_eff = 1 / np.sum(w**2)
                
                x_new_mean = np.zeros(dim)
                for i in range(mu):
                    x_new_mean += w[i] * offspring[i]
                x_mean = x_new_mean
                
                # Update step size
                z_mean = np.zeros(dim)
                for i in range(mu):
                    z_mean += w[i] * (offspring[i] - x_mean) / sigma
                # Note: z_mean should be computed from the original z, but we approximate
                
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * (z_mean / sigma)
                
                # Hessian-free approximation for C update
                # In practice, we need the actual z values; for simplicity, we'll use a simplified update
                # This is a simplified CMA-ES; a full implementation would track z values
                
                # Update C matrix (simplified)
                for i in range(mu):
                    y_i = (offspring[i] - x_mean) / sigma
                    C = (1 + cmu * (2 / (mu_eff * dim) - 1) - cs * (2 - cs)) * C + \
                        cmu * np.outer(y_i, y_i) * w[i]
                
                # Ensure C is positive definite
                try:
                    C_sqrt, _ = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                    inv_C_sqrt = np.linalg.inv(C_sqrt)
                except np.linalg.LinAlgError:
                    # Fallback: reset C
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    inv_C_sqrt = np.eye(dim)
                
                # Update p_c
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mu_eff) * inv_C_sqrt @ z_mean
                
                # Step size adaptation
                sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Stagnation check
                if stagnation_count >= stagnation_limit:
                    break
                
                # Minimum sigma guard
                if sigma < 0.001:
                    break

        return best_x, best_f
