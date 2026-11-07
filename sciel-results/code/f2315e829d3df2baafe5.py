"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """IPOP-CMA-ES for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _ipop_cmaes(self, func, x0, sigma0, budget, rng):
        n = self.dim
        # Initial population size
        # For n=5, lambda=9 or 10 is often good. Let's use 10 for slightly better exploration early on.
        lambda_ = 10
        mu = lambda_ // 2
        
        # CMA-ES parameters
        c_sigma = (mu + 0.5) / (n + mu + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)
        c_1 = (2 / ((n + 1.3) ** 2 + mu)) * (2 - c_c + c_sigma)
        c_mu = (2 / ((n + 1.3) ** 2 + mu)) * min(1, c_1 / 2)
        
        # Eigenvalues for update
        alpha_mu = 2 * (c_mu * (n + 2) + c_1 * (n - 2)) / (c_mu + c_1)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Initialize state
        x = x0.copy()
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Pre-compute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Initial population
        pops = []
        fitvals = []
        
        for i in range(lambda_):
            z = rng.standard_normal(n)
            y = x + sigma * (B @ (D * z))
            y = np.clip(y, -5, 5)
            fy = func(y)
            pops.append(y)
            fitvals.append(fy)
            
        # Track global best
        best_x = x.copy()
        best_f = min(fitvals)
        
        # IPOP parameters
        # Restart if no improvement for a certain number of generations or if sigma is too small
        # Standard IPOP restarts when sigma < 1e-11 or if no improvement for 100 generations
        # We'll use a dynamic threshold.
        
        # We will run until budget is exhausted or time limit
        # But we need to return the best found so far.
        
        # To handle the "return best" requirement and IPOP restarts within a single __call__,
        # we can just run the loop and if a restart condition is met, we re-initialize the state
        # but keep the global best.
        
        # However, IPOP typically restarts with a smaller sigma.
        # Let's implement a simple version: if stuck, shrink sigma and recenter around best.
        
        # Actually, for a single run of 500 budget, a full IPOP might be overkill or complex to manage
        # with the strict budget count. 
        # Let's stick to a robust single CMA-ES run with good initialization.
        # The previous program failed likely due to poor initialization or too many restarts eating budget.
        # A single well-tuned CMA-ES is very strong on BBOB.
        
        # Let's use a slightly larger lambda to improve early exploration.
        # lambda=10, mu=5.
        
        it = 0
        no_improve_count = 0
        last_best_f = best_f
        
        # Main loop
        while len([1 for _ in pops]) <= budget - lambda_: # Rough check, we track calls
             # Actually, we need to track calls precisely.
             # Let's refactor to count calls.
             pass
             
        # Refactored loop for precise budget tracking
        calls_made = lambda_ # Initial population
        while calls_made < budget:
            # Sort by fitness
            indices = np.argsort(fitvals)
            pops = [pops[i] for i in indices]
            fitvals = [fitvals[i] for i in indices]
            
            # Update best
            if fitvals[0] < best_f:
                best_f = fitvals[0]
                best_x = pops[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # IPOP Restart Condition
            # If no improvement for 50 generations and sigma is small, restart
            if no_improve_count > 50 and sigma < 1.0:
                # Restart: recenter around best, shrink sigma
                x = best_x.copy()
                sigma = sigma * 0.5
                B = np.eye(n)
                D = np.ones(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                no_improve_count = 0
                # Generate new population
                pops = []
                fitvals = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    pops.append(y)
                    fitvals.append(fy)
                calls_made += lambda_
                continue
            
            # Update mean
            y_w = np.sum(weights[:, None] * np.array(pops[:mu]), axis=0)
            x_new = y_w
            
            # Update step size
            y_sigma = (B @ (D * np.array(pops[:mu] - x) / sigma))
            norm_y_sigma = np.linalg.norm(y_sigma)
            hsig = (np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1) / mu_eff)) / chi_n)
            ps = (1 - c_sigma) * ps + hsig * np.sqrt(mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e6)
            
            # Update evolution paths
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ (D * (x_new - x) / sigma))
            
            # Update covariance matrix
            C = (B @ (D ** 2) @ B.T)
            delta_h = (1 - c_c) ** (-0.5) + 2 * (c_c - c_1) / (c_1 + c_mu)
            hpc = np.linalg.norm(pc) < (1.4 + 1 / n) * chi_n * np.sqrt(1 - (1 - c_c) ** (it + 1))
            
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(weights[:, None] * (np.array(pops[:mu] - x) / sigma)[:, None] * (np.array(pops[:mu] - x) / sigma)[None, :], axis=0)
            
            C = (1 + c_1 * (hpc * delta_h * rank_one - rank_one) + c_mu * rank_mu) * C
            
            # Regularize C
            C = 0.5 * (C + C.T)
            
            # Eigen decomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D, 1e-12, None))
            
            x = x_new
            it += 1
            
            # Generate new population
            new_pops = []
            new_fitvals = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                new_pops.append(y)
                new_fitvals.append(fy)
                calls_made += 1
                
            pops = new_pops
            fitvals = new_fitvals
            
        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        rng = np.random.default_rng(42)
        
        # Single run of IPOP-CMA-ES
        # Initial point: Center is a good bet for BBOB, but random is safer for hidden shifts.
        # Let's try a mix: start at 0, but CMA-ES will move.
        # Actually, for BBOB, the optimum is shifted. A random start in [-4,4] is standard.
        # Let's start at 0 to leverage any symmetry if the shift is small, but 500 budget is small.
        # Random start is generally better for robustness against unknown shifts.
        x0 = rng.uniform(-4, 4, self.dim)
        sigma0 = 2.0 # Standard deviation of initial distribution
        
        # Check time
        if time.time() > deadline - 10:
            return x0, func(x0)
            
        best_x, best_f = self._ipop_cmaes(func, x0, sigma0, self.budget, rng)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
