"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Population-based CMA-ES Optimizer for BBOB f15 (Rastrigin rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # State
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Helper to safely call func and update state
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            try:
                f_val = func(x)
                calls_made += 1
                if np.isnan(f_val) or np.isinf(f_val):
                    return np.inf
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return np.inf

        # CMA-ES Parameters
        # For Rastrigin, we need a good initial step size.
        # The domain is [-5, 5]^5. The minimizer is in [-4, 4]^5.
        # A standard CMA-ES starts with sigma = 0.5 * range or similar.
        # Let's use a fixed seed for reproducibility of the random number generator.
        rng = np.random.default_rng(42)
        
        # Initial mean: 0 is a reasonable start, but maybe slightly randomized?
        # Given rotation, 0 is safe.
        mean = np.zeros(n)
        
        # Initial step size. 
        # Rastrigin has local minima at integer coordinates (before rotation/shift).
        # Distance between minima is 2. 
        # sigma = 1.0 is a good initial guess to jump between basins.
        sigma = 1.0
        
        # Population size
        # Standard CMA-ES: lambda = 4 + floor(3 * log(n))
        # For n=5, lambda = 4 + floor(3 * 1.6) = 4 + 4 = 8.
        # Let's use a slightly larger population for better exploration in early stages.
        # lambda = 10 is a good balance for dim=5.
        lambda_pop = 10
        
        # Weights for recombination
        # Standard: mu = lambda / 2
        mu = lambda_pop // 2
        # Weights are proportional to rank, normalized
        weights = np.arange(mu, 0, -1, dtype=float)
        weights /= np.sum(weights)
        
        # Variance-effectiveness
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        # Standard CMA-ES adaptation rules
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Covariance matrix
        C = np.eye(n)
        
        # Evolution path for sigma
        p_sigma = np.zeros(n)
        # Evolution path for covariance
        p_c = np.zeros(n)
        
        # Chi distribution expectation for N(0, I)
        chi_n = np.sqrt(n * (1 - 1/(4*n) + 1/(20*n**2)))
        
        # Damping factor
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        
        # Main Loop
        # We will run until budget is exhausted or deadline is reached.
        # Each generation uses 'lambda_pop' function calls.
        
        while calls_made < budget and time.time() < deadline:
            # 1. Generate population
            # Sample from N(mean, sigma^2 * C)
            # We need to sample from the multivariate normal.
            # x = mean + sigma * (L @ z), where L is Cholesky of C, z ~ N(0, I)
            
            # Optimization: We can use scipy.stats.multivariate_normal or manual sampling.
            # Manual sampling with Cholesky is faster and avoids scipy overhead if possible, 
            # but numpy.linalg.cholesky is sufficient.
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset it
                C = np.eye(n)
                L = np.eye(n)
                sigma *= 0.5 # Shrink step size to recover
            
            population = np.zeros((lambda_pop, n))
            fitnesses = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                z = rng.standard_normal(n)
                x = mean + sigma * (L @ z)
                population[i] = x
                fitnesses[i] = evaluate(x)
                
                # Check if we should break early if budget is hit
                if calls_made >= budget or time.time() > deadline:
                    break
            
            if calls_made >= budget or time.time() > deadline:
                break
                
            # 2. Sort population by fitness
            # np.inf for failed evaluations
            order = np.argsort(fitnesses)
            # Handle case where all are inf? Unlikely, but safe.
            if np.all(np.isinf(fitnesses)):
                # Random restart
                mean = rng.uniform(-4, 4, n)
                sigma = 1.0
                C = np.eye(n)
                continue
                
            sorted_population = population[order]
            sorted_fitnesses = fitnesses[order]
            
            # 3. Update mean
            # Weighted recombination of the best 'mu' individuals
            new_mean = np.sum(weights[:, np.newaxis] * sorted_population[:mu], axis=0)
            
            # 4. Update covariance matrix
            # Diff
            diff = (sorted_population[:mu] - mean) / sigma
            # Rank-mu update
            C += c_mu * np.sum(weights[:, np.newaxis] * (diff * diff.T), axis=0)
            # Rank-1 update
            C += (2 - c_1 - c_mu) * (p_c * p_c.T)
            
            # 5. Update evolution paths
            # p_sigma update
            # First, normalize p_sigma by the norm of the step
            # The standard update is:
            # p_sigma = (1 - c_sigma) * p_sigma + h_sigma * sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (new_mean - mean)
            
            # Calculate h_sigma (heaviside function for adaptation)
            # This requires the norm of the step vector in the C-space.
            # norm_step = || new_mean - mean ||_C
            # We approximate using the current C.
            # Actually, the standard formula uses the inverse of C.
            # Let's use a simplified but robust version often used in implementations:
            # If we don't have inv(C) easily, we can use the norm of the difference divided by sigma.
            # A more accurate h_sigma check:
            # E[||z||] = chi_n
            # norm_diff = np.linalg.norm(new_mean - mean)
            # This is in Euclidean space. We need to transform to C-space.
            # x_C = L^{-1} (x - mean)
            # Let's just use the standard heuristic if exact h_sigma is too complex.
            # For simplicity and speed, we can use a fixed h_sigma = 1 if sigma is not shrinking too fast, 
            # or implement the standard check.
            
            # Let's implement the standard check:
            # norm_C = np.linalg.norm(np.linalg.solve(L, new_mean - mean))
            # h_sigma = 1 if norm_C < (1.4 + 2/(n+1)) * chi_n else 0
            
            try:
                norm_C = np.linalg.norm(np.linalg.solve(L, new_mean - mean))
            except np.linalg.LinAlgError:
                norm_C = np.inf
                
            h_sigma = 1.0 if norm_C < (1.4 + 2.0/(n+1.0)) * chi_n else 0.0
            
            p_sigma = (1.0 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2.0 - c_sigma) * mueff) * (new_mean - mean) / sigma
            
            # p_c update
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mueff) * (new_mean - mean) / sigma
            p_c = (1.0 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2.0 - c_c) * mueff) * (new_mean - mean) / sigma
            
            # 6. Update sigma
            # sigma = sigma * exp(c_sigma / damps * (||p_sigma|| / chi_n - 1))
            p_sigma_norm = np.linalg.norm(p_sigma)
            sigma = sigma * np.exp(c_sigma / damps * (p_sigma_norm / chi_n - 1.0))
            
            # Prevent sigma from becoming too small or too large
            if sigma < 1e-10:
                sigma = 1e-10
            if sigma > 10.0:
                sigma = 10.0
                
            # 7. Update mean
            mean = new_mean
            
            # 8. Check for stagnation or convergence
            # If the best fitness hasn't improved in many generations, we might restart or adjust.
            # CMA-ES handles this with sigma adaptation, but if sigma gets too small, it stagnates.
            # If sigma is very small and we haven't found a good solution, we can try a random jump.
            # However, with a budget of 500 and lambda=10, we have ~50 generations.
            # This is enough for CMA-ES to converge if initialized well.
            
            # Optional: If best_f is still very high (e.g., > 100), and we have used half the budget,
            # we might consider a restart from a different random point.
            # But CMA-ES is generally robust. Let's stick to the main loop.
            
            # To ensure we don't miss the global minimum due to poor initialization,
            # we can add a small probability of injecting a random point into the population
            # or restarting the mean if progress is slow.
            # Given the budget is small (500), restarting is risky. 
            # Better to rely on the global exploration of the initial large sigma.
            
            # One improvement: If the best point found so far is significantly better than the current mean's neighborhood,
            # we can pull the mean towards it.
            # CMA-ES does this naturally via recombination.
            
            # Safety: If we are stuck (sigma very small, no improvement), break and use best_x.
            if sigma < 1e-6:
                # Try to escape local minimum by randomizing direction
                # This is a heuristic.
                # Let's just break and return best found.
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
