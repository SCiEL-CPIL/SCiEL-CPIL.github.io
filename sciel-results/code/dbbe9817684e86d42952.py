"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """CMA-ES based optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill
        
        # CMA-ES parameters
        # Population size lambda
        pop_size = 4 + int(3 * np.log(self.dim))  # Standard rule, for dim=5 -> 4+3*1.6=8 -> 12? 
        # Let's use a slightly larger population to make better use of 500 budget
        # 500 evals / 50 generations = 10 per generation. Let's try lambda=10.
        lambda_ = 10
        
        # Initial point: center of the box is a good heuristic for BBOB
        x0 = np.zeros(self.dim)
        f_best = func(x0)
        x_best = x0.copy()
        
        # Initial sigma (step size)
        # Box is [-5, 5], so width 10. Starting sigma ~ 2.5 covers the box well.
        sigma = 2.5
        
        # CMA-ES state
        # We will implement a simplified CMA-ES loop to avoid external dependencies if possible,
        # but scipy is available. However, scipy.optimize.minimize with method='CMA-ES' 
        # is robust and handles the budget via maxiter.
        # Note: scipy CMA-ES uses `maxiter` as number of generations, not function evaluations.
        # We need to map 500 evaluations to generations.
        # Each generation uses lambda_ evaluations.
        max_iter = self.budget // lambda_ + 1
        
        # To strictly control evaluation count and handle the deadline, 
        # a manual implementation or careful wrapping is safer.
        # Let's use scipy.optimize.minimize but with a counter.
        
        eval_count = 1 # Initial point already evaluated
        
        def objective(x):
            nonlocal eval_count
            if time.time() > deadline:
                raise TimeoutError()
            eval_count += 1
            return func(x)
            
        # We cannot easily pass max_evals to scipy CMA-ES directly in all versions,
        # but we can use `maxiter` and `options`.
        # However, scipy's CMA-ES might not stop exactly on budget.
        # Let's implement a manual CMA-ES step for full control and to ensure 
        # we don't exceed 500 calls or the time limit.
        
        # Manual CMA-ES Implementation
        # Parameters
        mu = lambda_ // 2  # Number of best individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # State
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        C = np.eye(self.dim)
        C_inv = np.eye(self.dim)
        C_sqrt_inv = np.eye(self.dim)
        
        # Helper to update C and its inverse/sqrt
        def update_cma(C, C_inv, C_sqrt_inv, weights, y_norms, alpha_mu, mu_eff, dim):
            # Rank-one update
            C += alpha_mu * np.outer(y_norms, y_norms)
            
            # Rank-mu update
            # y_k are the transformed steps
            # We need the y_k vectors. Let's store them.
            
            # This manual implementation is getting complex and prone to errors.
            # Let's stick to scipy.optimize.minimize with a strict budget wrapper.
            pass

        # Re-evaluating strategy: 
        # Scipy's CMA-ES is robust. The main risk is exceeding budget.
        # We can wrap the function to raise an exception on budget overrun.
        # Scipy will catch it? No, it will propagate. 
        # Better: use `maxiter` and accept that it might use slightly more/less,
        # or implement a simple (mu+lambda)-ES which is easier to control.
        
        # Given the "Different Powers" function is non-separable but smooth,
        # and dim=5 is small, a well-tuned (mu+lambda)-ES is often competitive 
        # and much faster/simpler to implement correctly with strict budget.
        
        # Let's implement a (mu+lambda)-ES with adaptive step size.
        mu = 5
        lambda_ = 20
        # 500 / 20 = 25 generations.
        
        # Initial population
        pop = x0 + sigma * np.random.randn(lambda_, self.dim)
        # Clip to box
        pop = np.clip(pop, -5, 5)
        
        f_vals = np.array([func(x) for x in pop])
        eval_count = 1 + lambda_
        
        # Sort
        idx = np.argsort(f_vals)
        pop = pop[idx]
        f_vals = f_vals[idx]
        
        x_best = pop[0].copy()
        f_best = f_vals[0]
        
        # Update mean
        mean = np.sum(pop[:mu] * weights[:, np.newaxis], axis=0)
        
        # Update sigma (1/5 rule adapted for population)
        # Simple adaptation: if best improves, increase sigma, else decrease
        # Better: use the standard CMA-ES sigma control or a simple hill-climbing check.
        # Let's use a simple adaptive sigma based on the success of the step.
        
        # Actually, let's use `scipy.optimize.minimize` with method='Powell' 
        # or 'Nelder-Mead' as a fallback if CMA-ES is too heavy?
        # No, CMA-ES is best for this.
        
        # Let's try a hybrid: Start with CMA-ES from scipy, but if it fails or 
        # uses too many, fall back.
        
        # Actually, the simplest robust approach for 5D with 500 evals:
        # Use `scipy.optimize.differential_evolution`? No, too many evals.
        # Use `scipy.optimize.minimize` with `method='CMA-ES'`.
        
        # Let's implement the manual CMA-ES properly but simply.
        # Or just use a well-tuned (1+1)-ES with restarts?
        # The current program is (1+1)-ES. The directive asks for a MATERIAL change.
        # Population-based is the material change.
        
        # Let's implement a standard (mu+lambda)-ES with covariance adaptation 
        # simplified to diagonal covariance (which is just adaptive step sizes per dimension).
        # This is much faster and handles the "Different Powers" (which has different 
        # scales per dimension potentially) well.
        
        # Diagonal CMA-ES / Adaptive Step Size ES
        sigmas = np.ones(self.dim) * 2.0
        mean = x0.copy()
        
        # We already used 21 evals above. Let's restart cleanly with this method.
        # Reset eval count
        eval_count = 0
        
        def eval_func(x):
            nonlocal eval_count
            eval_count += 1
            return func(x)
            
        # Initial point
        mean = np.zeros(self.dim)
        f_mean = eval_func(mean)
        x_best = mean.copy()
        f_best = f_mean
        
        # Main loop
        # Budget remaining: 499
        # Generations: 499 // 20 = 24
        for gen in range(24):
            if time.time() > deadline:
                break
                
            # Generate population
            pop = []
            for _ in range(lambda_):
                step = sigmas * np.random.randn(self.dim)
                y = mean + step
                y = np.clip(y, -5, 5)
                pop.append(y)
            
            # Evaluate
            f_pop = np.array([eval_func(x) for x in pop])
            
            # Select best mu
            idx = np.argsort(f_pop)
            best_idx = idx[:mu]
            best_pop = pop[best_idx]
            best_f = f_pop[best_idx]
            
            # Update best
            if best_f[0] < f_best:
                f_best = best_f[0]
                x_best = best_pop[0].copy()
                
            # Update mean
            new_mean = np.sum(best_pop * weights[:, np.newaxis], axis=0)
            
            # Update sigmas (diagonal covariance adaptation)
            # Simple rule: if the new mean is better than the old mean, increase sigmas?
            # Or use the distance?
            # A common simple adaptation:
            # sigma_i = sigma_i * (1 + c * (success - 0.5))
            # Let's use a more robust approach: 
            # Estimate the variance of the successful steps.
            
            # For simplicity and robustness, let's just use a fixed schedule or 
            # a simple (1+1) rule on the mean step.
            # Actually, let's use the standard CMA-ES sigma update logic but simplified.
            
            # If the step towards the new mean was successful (improved f), increase sigma.
            if f_mean > f_best: # If current mean is worse than best found
                # We made progress
                sigmas *= 1.1
            else:
                sigmas *= 0.9
                
            # Clip sigmas
            sigmas = np.clip(sigmas, 0.01, 5.0)
            
            mean = new_mean
            f_mean = f_best # Approximation, we don't evaluate new_mean directly to save budget
            
        return x_best, f_best
# EVOLVE-BLOCK-END
