"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with deadline guard for f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_left():
            return deadline - time.time()

        calls = 0
        best_x = None
        best_f = np.inf

        # 1. Initialization: Sample diverse points to find a good starting basin
        # Use 10 points to get a rough idea of the landscape
        n_init = 10
        init_pts = rng.uniform(-5, 5, (n_init, dim))
        
        for pt in init_pts:
            if time_left() < 1.0 or calls >= budget:
                break
            f_val = func(pt)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # 2. Multi-restart CMA-ES
        # Budget remaining
        remaining = budget - calls
        
        if remaining <= 0:
            return best_x, best_f

        # CMA-ES Parameters
        # Population size lambda. For dim=5, lambda=4 or 5 is good.
        # Let's use lambda=5 to get better covariance estimation with fewer evaluations.
        lambda_pop = 5
        # Number of restarts we can afford.
        # Each generation costs lambda_pop calls.
        # Let's aim for restarts that consume roughly equal budget chunks.
        # If we use 10 restarts, each gets remaining/10 calls.
        # Min calls per restart to be useful: maybe 20-30?
        # Let's calculate max restarts.
        max_restarts = max(1, remaining // (lambda_pop * 10)) # At least 10 generations per restart
        # Cap restarts to avoid too many tiny runs
        max_restarts = min(max_restarts, 10)
        
        # Start points for CMA-ES:
        # 1. The current global best
        # 2. Other good points from init
        # 3. Random points if we have budget for more restarts
        
        # Sort init points by fitness to pick good starting locations
        # We didn't store all init fitnesses in a list, so let's just use best_x and randoms.
        # Better: We can re-evaluate or just use randoms for subsequent restarts.
        # To be more strategic, we can use the top 2-3 from init if we kept them. 
        # For simplicity and robustness, we use best_x for the first restart, 
        # and random points for others, but biased towards the center or previous bests.
        
        start_points = [best_x.copy()]
        for _ in range(max_restarts - 1):
            # Sample new start points, maybe slightly clustered around best_x or random
            # Random is safer to escape local optima
            start_points.append(rng.uniform(-5, 5, dim))

        for sp in start_points:
            if time_left() < 5.0 or calls >= budget:
                break
            
            # Run CMA-ES from sp
            x_mean = sp.copy()
            sigma = 1.0
            C = np.eye(dim)
            C_inv = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Adaptation parameters
            mu = lambda_pop // 2  # Number of selected individuals
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # Pre-compute square root of C (Cholesky)
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(dim)
            
            stagnation_counter = 0
            best_f_restart = best_f
            
            while calls < budget and time_left() > 5.0:
                # Sample lambda points
                ys = []
                xs = []
                for _ in range(lambda_pop):
                    z = rng.standard_normal(dim)
                    y = C_sqrt @ z
                    x = x_mean + sigma * y
                    # Clip to box
                    x = np.clip(x, -5, 5)
                    ys.append(y)
                    xs.append(x)
                
                # Evaluate
                fs = []
                for x in xs:
                    if calls >= budget or time_left() < 1.0:
                        break
                    f_val = func(x)
                    calls += 1
                    fs.append(f_val)
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x.copy()
                
                if len(fs) < lambda_pop:
                    break # Out of budget or time
                
                # Sort by fitness
                idx = np.argsort(fs)
                xs_sorted = [xs[i] for i in idx[:mu]]
                ys_sorted = [ys[i] for i in idx[:mu]]
                fs_sorted = [fs[i] for i in idx[:mu]]
                
                # Update mean
                x_new_mean = np.sum(np.array(xs_sorted) * weights[:, np.newaxis], axis=0)
                
                # Update step size (sigma)
                y_sigma = (x_new_mean - x_mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                # Update covariance matrix
                y_c = (x_new_mean - x_mean) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
                
                h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2)) < (1.4 + 2 / (dim + 1)) * np.sqrt(mu_eff) else 0
                
                C = (1 + c_1 * h_sigma * (2 - h_sigma) - c_1 - c_mu) * C \
                    + c_1 * (p_c @ p_c - C) \
                    + c_mu * np.sum(weights[:, np.newaxis] * np.array(ys_sorted) @ np.array(ys_sorted).T, axis=0)
                    
                # Ensure positive definiteness
                try:
                    C_inv = np.linalg.inv(C)
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_inv = np.eye(dim)
                    C_sqrt = np.eye(dim)
                
                x_mean = x_new_mean
                
                # Check for stagnation
                if best_f_restart - best_f < 1e-8:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                    best_f_restart = best_f
                    
                if stagnation_counter > 10: # 10 generations without improvement
                    break
                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
