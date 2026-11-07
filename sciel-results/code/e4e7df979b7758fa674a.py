"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Allocate budget to 5 restarts
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        # Ensure we don't exceed budget due to integer division
        # We will manage a global call counter to be safe.
        
        calls_made = 0
        best_x = None
        best_f = float('inf')

        # Helper to check time and budget
        def should_stop():
            nonlocal calls_made
            if calls_made >= budget:
                return True
            if time.time() > deadline:
                return True
            return False

        # CMA-ES implementation
        def run_cmaes(x0, sigma0, max_calls):
            nonlocal calls_made, best_x, best_f
            
            # CMA-ES parameters
            n = dim
            lambda_ = int(4 + 3 * np.log(n)) # Population size
            mu = lambda_ // 2 # Number of weights
            # Compute weights
            w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mueff = 1 / np.sum(w**2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n**2))
            
            # State
            x_mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Initial population
            # We must respect the global call budget and time limit
            # We'll evaluate points one by one to strictly control calls and check time.
            
            # For simplicity and strict budget control, we will generate the population, 
            # but evaluate and process them in a way that allows early termination.
            
            # Actually, standard CMA-ES evaluates the whole population at once.
            # Given the tight budget (500 total, 100 per restart), and n=5, lambda=10.
            # 100 calls / 10 = 10 generations per restart. This is very few.
            # CMA-ES might not converge well in 10 generations.
            
            # Alternative: Use a simpler, more robust method for small budgets?
            # The seed (1+1) got 1.0. 
            # Let's stick to CMA-ES but maybe increase lambda or reduce n_restarts?
            # If we do 1 restart with 500 calls, lambda=10 -> 50 generations. That's better.
            # But multi-start is better for AOCC (early progress).
            
            # Let's try 3 restarts with ~166 calls each.
            # Or, dynamically adjust.
            
            # Let's refine the strategy: 
            # Use a smaller population for faster iterations? 
            # Or just accept that CMA-ES with small budget is tricky.
            
            # Let's try a different approach: 
            # 1. Initial random search to get a decent starting point (exploits early AOCC).
            # 2. Then CMA-ES from the best point.
            
            # Revised Strategy inside __call__:
            # Phase 1: Random search for 50 calls to find a good basin.
            # Phase 2: CMA-ES with remaining 450 calls.
            
            # This is getting complex to implement inside a nested function with nonlocal state.
            # Let's restructure __call__ to be cleaner.
            
            return x_mean, best_f # Placeholder, we won't use this nested structure directly like this.

        # --- Restructured __call__ Logic ---
        
        # Phase 1: Random Search for early AOCC boost
        n_random = 50
        for i in range(n_random):
            if should_stop():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES from the best random point
        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)
            calls_made += 1

        x_mean = best_x.copy()
        sigma = 2.0 # Initial step size
        n = dim
        # Larger population for better sampling in remaining budget?
        # Remaining budget ~ 450. 
        # If lambda=10, 45 generations. 
        # If lambda=20, 22 generations.
        # Let's use lambda = 10 for n=5.
        lambda_ = 10
        mu = lambda_ // 2
        
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w**2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n**2))
        
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # CMA-ES Main Loop
        while not should_stop():
            # Generate population
            # We need to generate lambda points. 
            # We check budget before generating each point to avoid overshooting.
            # However, CMA-ES needs the whole population to compute the mean update.
            # If we run out of budget mid-population, we can't finish the generation.
            # Strategy: If we don't have enough calls for a full population, break.
            if calls_made + lambda_ > budget:
                # Try to use remaining calls for a smaller population? 
                # Or just stop. Stopping is safer for validity.
                break
                
            # Generate lambda samples
            # Use Cholesky of C
            try:
                B = np.linalg.cholesky(C)
            except:
                # If C is not positive definite, reset to identity
                C = np.eye(n)
                B = np.eye(n)
                
            ys = []
            for i in range(lambda_):
                y = rng.standard_normal(n)
                ys.append(y)
            ys = np.array(ys)
            
            # Transform to x
            X = np.array([x_mean + sigma * (B @ y.T).T for y in ys])
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lambda_
            
            # Sort by fitness
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Update mean
            x_new_mean = np.sum(w[:, np.newaxis] * X[:mu], axis=0)
            
            # Update covariance matrix
            y_invC = np.linalg.solve(B, (X[:mu] - x_mean).T / sigma).T
            # Note: y_invC shape is (mu, n). We need (mu, n) for outer product?
            # Actually, standard formula: 
            # C = (1 - c_1 - c_mu) C + c_1 (2 - mu_1) p_c p_c^T + c_mu / sigma^2 sum w_i (x_i - x_mean)(x_i - x_mean)^T
            # Using transformed variables:
            # C = (1 - c_1 - c_mu) C + c_1 (2 - mu_1) p_c p_c^T + c_mu / sigma^2 sum w_i (B y_i)(B y_i)^T
            # This is computationally heavier. Let's use the direct formula with x.
            
            # Direct formula update
            delta_mean = x_new_mean - x_mean
            p_c = (1 - c_c) * p_c + c_c * delta_mean / sigma
            
            # Weighted sum of outer products
            # We need to be careful with the term c_mu / sigma^2 * sum w_i (x_i - x_mean)(x_i - x_mean)^T
            # Let's compute it
            diff = X[:mu] - x_mean
            # Weighted outer product sum
            # This can be done as: sum w_i diff_i diff_i^T
            # Using matrix multiplication: (W * diff) @ diff.T where W is diag(w)
            # Or: (diff * w[:, np.newaxis]) @ diff.T
            weighted_outer = (diff * w[:, np.newaxis]) @ diff.T
            
            C = (1 - c_1 - c_mu) * C + c_1 * (2 - mueff) * np.outer(p_c, p_c) + (c_mu / (sigma * sigma)) * weighted_outer
            
            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * delta_mean / sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Ensure sigma doesn't explode or vanish
            sigma = np.clip(sigma, 1e-10, 100)
            
            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5, 5)
            
            # Check convergence
            if sigma < 1e-10:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
