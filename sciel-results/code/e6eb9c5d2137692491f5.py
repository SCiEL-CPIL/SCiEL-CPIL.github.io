import numpy as np
import time


class Optimizer:
    """Robust optimizer for BBOB f9 (rotated Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        def time_left():
            return deadline - time.time()

        # State
        n_calls = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            n_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Exploration
        # Evaluate origin and several random points to find a good basin
        # Rosenbrock is often tricky; random points help escape bad local minima or find the global one
        # We use a small fraction of the budget for this.
        n_global = min(50, budget // 5)
        
        # Always try origin
        evaluate(np.zeros(dim))
        
        # Try random points
        np.random.seed(42) # Reproducibility for the search itself is not required, but good for debugging
        # Actually, let's use a fixed seed for the optimizer's internal random choices to ensure consistency if needed, 
        # but the task implies stochastic search is fine. Let's just use np.random.
        
        for _ in range(n_global - 1):
            if n_calls >= budget or time_left() < 10.0:
                break
            x_try = np.random.uniform(-5.0, 5.0, size=dim)
            evaluate(x_try)

        # If we ran out of budget or time, return best
        if n_calls >= budget or time_left() < 5.0:
            return best_x, best_f

        # Phase 2: Local Refinement with CMA-ES
        # Start CMA-ES from the best point found
        x_mean = best_x.copy()
        sigma = 1.0  # Start with a moderate sigma
        
        # CMA-ES Parameters for dim=5
        lam = 6 + int(3 * np.log(dim))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        c_mu = (weights[0] ** 2) / (np.sum(weights ** 2))
        c_sigma = (1 + 2 * max(0, np.sqrt((c_mu - 1) / (dim + 1)) - 1)) / (dim + 2)
        c_c = (4 + c_mu) / (dim + 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + c_mu)
        c_mu_ = 2 * ((c_mu - 2) / (c_mu + 1.3) - 1)
        
        x_mean = best_x.copy()
        sigma = 1.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = 1.0 / (c_c * (dim + 1))
        ds = 1.0 + 2 * max(0, np.sqrt((c_mu - 1) / (dim + 1)) - 1) + c_sigma
        C = np.eye(dim)

        while n_calls < budget and time_left() > 1.0:
            # Generate population
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            N = np.random.randn(lam, dim)
            y = N @ L.T
            x_pop = x_mean + sigma * y

            # Evaluate population
            f_pop = np.array([evaluate(xi) for xi in x_pop])
            
            if n_calls >= budget or time_left() < 1.0:
                break

            # Sort
            order = np.argsort(f_pop)
            y_sorted = y[order]
            f_sorted = f_pop[order]

            # Update mean
            x_mean_new = x_mean + sigma * np.sum(weights[:mu, None] * y_sorted[:mu], axis=0)

            # Update sigma
            y_sigma = y_sorted[:mu]
            norm_y_sigma = np.sqrt(np.sum(weights[:mu] * np.sum(y_sigma**2, axis=1)))
            sigma *= np.exp(c_sigma / ds * (norm_y_sigma / np.sqrt(c_mu) - 1))

            # Update evolution paths
            pc = (1 - c_c) * pc + c_c / np.sqrt(c_mu) * (x_mean_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * c_mu) / sigma * (x_mean_new - x_mean)

            # Update covariance
            # h_sigma is approx 1 if ps is small enough
            h_sigma = 1.0 if np.dot(ps, ps) / dim < (1 + (1 - c_sigma) ** 2) * (2 - c_sigma) * c_mu else 0.0
            # More standard check:
            # h_sigma = 1 if ||ps||^2/dim < (1 + (1-c_sigma)^2) * (2-c_sigma) * c_mu
            
            # Correction for h_sigma calculation based on standard CMA-ES
            # The condition is typically: ||ps||^2 / dim < (1 + (1-c_sigma)^2) * (2-c_sigma) * c_mu
            # Let's use a simpler, robust approximation or the standard one.
            # Standard: h_sigma = 1 if ||ps||^2 < (1 + (1-c_sigma)^2) * (2-c_sigma) * c_mu * dim
            # Actually, the standard condition is:
            # h_sigma = 1 if ||ps||^2 / dim < (1 + (1-c_sigma)^2) * (2-c_sigma) * c_mu
            # Let's stick to the standard formula from Hansen's docs.
            # However, for robustness and speed, a simplified rank update is often sufficient for small budgets.
            # Let's use the standard rank-mu update with a simplified delta_hs.
            
            # A common stable implementation uses:
            # delta_hs = 1 if h_sigma else 0
            # But let's just use the standard update with h_sigma = 1.0 for simplicity in this small dim, 
            # or calculate it properly.
            
            # Proper h_sigma:
            # h_sigma = 1 if np.dot(ps, ps) / dim < (1 + (1 - c_sigma)**2) * (2 - c_sigma) * c_mu else 0
            
            # Let's just set h_sigma = 1.0 for simplicity and stability in 5D, 
            # as the covariance adaptation is the main driver.
            h_sigma = 1.0
            
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:mu, None, None] * y_sorted[:mu][:, :, None] * y_sorted[:mu][:, None, :], axis=0)
            
            # Update C
            C = (1 + c_1 * h_sigma - c_1 - c_mu_) * C + c_1 * (rank_one + (h_sigma - 1) * C) + c_mu_ * rank_mu
            
            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2.0
            
            x_mean = x_mean_new

        return best_x, best_f
