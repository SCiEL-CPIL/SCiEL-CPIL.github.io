import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES implementation
        # 1. Initialization
        # Start at origin, which is often close to the shifted minimum in BBOB
        x = np.zeros(dim)
        
        # Initial sigma. 
        # For Rosenbrock in [-5,5]^5, a sigma of 1.0 is a reasonable scale.
        # The shift is in [-4,4], so 1.0 covers a good portion of the space initially.
        sigma = 1.0
        
        # Covariance matrix C (initially identity)
        C = np.eye(dim)
        
        # Inverse square root of C for sampling
        # We maintain C and its inverse square root C^{-1/2}
        # For efficiency, we can use eigendecomposition updates or just recompute if dim is small (dim=5)
        # Recomputing is cheap for dim=5.
        
        # Weights for selection
        # Standard CMA-ES uses mu = lambda/2
        # We will adapt lambda based on budget to balance exploration/exploitation
        
        # Call counter
        calls_made = 0
        
        # Best value tracking
        best_x = x.copy()
        best_f = func(x)
        calls_made += 1
        
        # History of best values for potential stagnation detection (simple version)
        best_history = [best_f]
        
        # Main loop
        while calls_made < budget and time.time() < deadline - 5:
            # Adaptive population size
            # Start with larger population for exploration, shrink for exploitation
            # A common heuristic: lambda = max(4, 4 + int(3 * (dim + 1))) 
            # But for budget constraints, we can be more aggressive.
            # Let's use a schedule:
            # Phase 1 (first 50% budget): lambda = 20
            # Phase 2 (next 30%): lambda = 10
            # Phase 3 (last 20%): lambda = 5
            
            progress = calls_made / budget
            if progress < 0.4:
                lambda_pop = 20
            elif progress < 0.8:
                lambda_pop = 10
            else:
                lambda_pop = 5
            
            mu = max(1, lambda_pop // 2)
            
            # Weights (standard CMA-ES weights)
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 1) - np.log(i + 1)
            weights /= np.sum(weights)
            
            # mu_eff
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Step size control parameters
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # State variables
            if not hasattr(self, '_cma_state') or self._cma_state is None:
                self._cma_state = {
                    'p_sigma': np.zeros(dim),
                    'p_c': np.zeros(dim),
                    'sigma': sigma,
                    'C': C,
                    'x': x
                }
            
            # Restore state (in case we restarted or just first time)
            # Actually, since we are in a single __call__, we can keep local variables
            # But wait, the class instance is reused? No, "evaluator creates Optimizer... and calls it once".
            # So local variables are fine.
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            # We need to persist C, sigma, p_sigma, p_c across iterations within this call.
            # I initialized them outside the loop? No, I initialized C and sigma.
            # I need to initialize p_sigma and p_c outside the loop too.
            
            # Let's refactor: Move state init outside the while loop.
            
            break # Break to fix structure
            
        # RE-IMPLEMENTATION WITH CORRECT STATE SCOPE
        
        # State
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial evaluation
        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls_made = 1
        
        # Parameters
        c_sigma = (1 + 2 / (dim + 1)) / (dim + 1) # Approx for mu_eff=1
        # Better to calculate mu_eff properly each time or use standard approximations
        # For simplicity and robustness in this short budget, we use fixed standard params
        # tuned for dim=5.
        
        c_sigma = 0.5
        d_sigma = 1.0
        c_c = 0.5
        c_1 = 0.1
        c_mu = 0.1
        
        # Main Loop
        while calls_made < budget and time.time() < deadline - 10:
            # Adaptive population size
            progress = calls_made / budget
            if progress < 0.3:
                lambda_pop = 20
            elif progress < 0.7:
                lambda_pop = 10
            else:
                lambda_pop = 5
                
            mu = max(1, lambda_pop // 2)
            
            # Weights
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 1) - np.log(i + 1)
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            # Update c_sigma, etc. based on mu_eff
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            # Sample lambda points
            # Need C^{-1/2}
            # Compute eigendecomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Generate random normals
            Z = np.random.randn(lambda_pop, dim)
            
            # Transform
            Y = Z @ C_inv_sqrt.T
            X = x_mean + sigma * Y
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lambda_pop
            
            # Check if we exceeded budget (shouldn't happen if we check before, but safe)
            if calls_made > budget:
                break
                
            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
            
            # Selection
            X_selected = X_sorted[:mu]
            w = weights
            
            # New mean
            x_new_mean = np.sum(w[:, None] * X_selected, axis=0)
            
            # Update step size
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu_eff) * C_inv_sqrt @ (x_new_mean - x_mean) / sigma
            delta_x = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C_inv_sqrt @ delta_x)
            
            # Check h_sigma condition for p_c update
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lambda_pop + 1))) <
                       (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)))
            h_sigma = 1.0 if h_sigma else 0.0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * delta_x
            
            # Update covariance
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c + c_c * p_c) @ (p_c + c_c * p_c).T + c_mu * sum(w_i * y_i @ y_i.T)
            # where y_i = (x_i - x_mean) / sigma
            
            Y_selected = (X_selected - x_mean) / sigma
            rank_mu = np.sum(w[:, None] * Y_selected * Y_selected.T, axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c + c_c * p_c, p_c + c_c * p_c) + c_mu * rank_mu
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update mean
            x_mean = x_new_mean
            
            # Check convergence
            # If sigma is very small or step size is negligible
            if sigma < 1e-10:
                break
                
            # Optional: If stagnation, restart?
            # For now, rely on adaptive lambda and sigma.
            # If we get stuck, the small lambda in late phase will exploit.
            # If we are far from optimum, the initial large lambda explores.
            
            # Safety check for time
            if time.time() > deadline - 20:
                break

        return best_x, best_f
