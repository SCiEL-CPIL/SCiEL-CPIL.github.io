# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-Start CMA-ES Optimizer for BBOB-style functions.
    
    Strategy:
    1. Brief random initialization to identify a promising region.
    2. Run CMA-ES (Covariance Matrix Adaptation Evolution Strategy) starting from that region.
       CMA-ES is superior to (1+1)-ES for functions with sharp ridges or ill-conditioned landscapes
       because it adapts the shape of the search distribution (covariance matrix) to the local geometry.
    3. If time allows, perform a second CMA-ES restart from a different random point to mitigate
       the risk of getting stuck in a local optimum, though for 5D BBOB functions, a single
       well-tuned CMA-ES run usually dominates.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def timed_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= self.budget:
                return None
            if time.time() > deadline - 2:
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initialization
        # Sample a few points to find a good starting center for CMA-ES.
        # CMA-ES is local, so a good start is crucial.
        n_init = 20
        init_xs = []
        init_fs = []
        
        for _ in range(n_init):
            if calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = timed_call(x)
            if fx is not None:
                init_xs.append(x)
                init_fs.append(fx)
                
        if not init_xs:
            # Fallback if we ran out of budget immediately
            x = np.zeros(self.dim)
            fx = func(x)
            calls_made += 1
            return x, fx

        # Find the best initialization point
        best_init_idx = np.argmin(init_fs)
        x_mean = init_xs[best_init_idx].copy()
        f_mean = init_fs[best_init_idx]

        # Phase 2: CMA-ES Implementation
        # We implement a lightweight CMA-ES here.
        # Parameters:
        # lambda: population size (default 4 + floor(3*ln(n)))
        # mu: number of selected parents (lambda // 2)
        # sigma: step size
        # C: covariance matrix
        # C1, Cm: adaptation rates
        
        n = self.dim
        lam = 4 + int(3 * np.log(n))
        if lam < 2:
            lam = 2
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, sqrt(mueff / n - 1) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initialization
        sigma = 1.0
        C = np.eye(n)
        Csqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Main Loop
        while calls_made < self.budget:
            if time.time() > deadline - 2:
                break
            
            # Generate population
            offspring = []
            offspring_vals = []
            
            for i in range(lam):
                if calls_made >= self.budget or time.time() > deadline - 2:
                    break
                
                # Sample z
                z = rng.standard_normal(n)
                # Transform using Cholesky decomposition of C
                # Note: Csqrt is updated incrementally in full CMA-ES, 
                # but for simplicity and speed in 5D, we can recompute 
                # or use a simpler approximation. 
                # For robustness, let's use the standard update.
                # However, maintaining Csqrt correctly requires careful bookkeeping.
                # A simpler alternative for small n: Use the eigenvalue decomposition 
                # or just update C and use a generic transformation.
                # Actually, for n=5, computing Cholesky every step is cheap.
                # But we need to update C first? No, C is updated after selection.
                # So we use the current Csqrt.
                
                y = x_mean + sigma * (Csqrt @ z)
                # Clip to bounds
                y = np.clip(y, -5, 5)
                
                fy = timed_call(y)
                if fy is not None:
                    offspring.append(y)
                    offspring_vals.append(fy)
                    
            if len(offspring) == 0:
                break
                
            # Sort offspring by fitness
            sorted_idx = np.argsort(offspring_vals)
            selected_offspring = [offspring[i] for i in sorted_idx[:mu]]
            selected_vals = [offspring_vals[i] for i in sorted_idx[:mu]]
            
            # Update mean
            new_x_mean = np.zeros(n)
            for i in range(mu):
                new_x_mean += weights[i] * selected_offspring[i]
                
            # Update step size (sigma)
            # Calculate difference of mean
            diff_mean = new_x_mean - x_mean
            norm_diff = np.linalg.norm(diff_mean)
            
            if norm_diff > 1e-16:
                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (Csqrt.T @ (diff_mean / (sigma * norm_diff)))
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update p_c
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lam + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                # Simplified h_sigma check
                h_sigma = np.linalg.norm(p_sigma) / chi_n < 1.4 + 2 / (n + 1)
                
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (diff_mean / sigma)
                
                # Update covariance C
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(p_c, p_c)
                
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    y_i = (selected_offspring[i] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(y_i, y_i)
                    
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                
                # Update Csqrt (Cholesky)
                # To ensure positive definiteness, we might need to fix eigenvalues
                # For n=5, we can just recompute Cholesky
                try:
                    Csqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback: reset C to identity or add small regularization
                    C = C + 1e-6 * np.eye(n)
                    try:
                        Csqrt = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(n)
                        Csqrt = np.eye(n)
            
            # Update best if new mean is better? No, CMA-ES keeps the weighted mean.
            # We track global best in timed_call.
            
            x_mean = new_x_mean
            f_mean = np.min(selected_vals) # Approximate
            
            # Update global best if any offspring was better (handled in timed_call)
            
            # Optional: If sigma becomes very small, we might be converged.
            if sigma < 1e-8:
                break

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1
            
        return best_x, best_f

# EVOLVE-BLOCK-END
