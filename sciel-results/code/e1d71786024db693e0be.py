"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. 
        # We use 300s as our hard stop to ensure we return within limits.
        deadline = start_time + 300
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Initial population size lambda
        # For dim=5, a standard lambda is around 4 + floor(3 * log(5)) = 4 + 5 = 9
        # Let's use a slightly larger population for robustness: 12
        lam = 12
        
        # Initial step size sigma
        # Box is [-5, 5], width 10. 
        # If we start at 0, we want to cover the space reasonably.
        # A common initialization is sigma = 1.0 or related to box size.
        # Given the shift is in [-4, 4], starting at 0 with sigma=2.0 allows 
        # good coverage of the likely region.
        sigma = 2.0
        
        # Mean of the distribution
        # Start at origin, as it's the center of the box and often close to 
        # the shifted minimum in BBOB-style setups.
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Weights for recombination
        # Standard CMA-ES uses lognormal weights
        mu = lam // 2
        weights = np.log(np.arange(lam, 0, -1))
        weights /= weights.sum()
        # Effective population size
        mueff = 1.0 / np.sum(weights[:mu]**2)
        
        # Step size adaptation parameters
        # 1/5 rule is standard for (1+1), but for CMA-ES we use CSA (CMA-ES)
        # or simple 1/5 rule on the mean step.
        # Let's implement a simplified but effective CMA-ES variant.
        
        # Evaluation counter
        eval_count = 0
        
        # Best tracking
        best_x = mean.copy()
        best_f = float('inf')
        
        # Function wrapper to enforce budget and time
        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise TimeoutError
            
            x = np.asarray(x, dtype=float)
            # The problem states func clips to [-5, 5], but we should 
            # ensure our inputs are sane.
            x = np.clip(x, -5, 5)
            
            fx = func(x)
            eval_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        # Main CMA-ES Loop
        try:
            while eval_count < budget and time.time() < deadline:
                # 1. Sample population
                # Cholesky decomposition for sampling from N(mean, sigma^2 * C)
                # We update C periodically, but for simplicity and speed in this 
                # constrained budget, we can use a simplified update or 
                # rely on the step size adaptation primarily.
                # A full CMA-ES updates C every generation.
                
                # For dim=5 and budget=500, we can afford ~40 generations.
                # Full CMA-ES is feasible.
                
                # Compute Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset to identity
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                # Sample
                population = np.empty((lam, dim))
                for i in range(lam):
                    z = np.random.randn(dim)
                    population[i] = mean + sigma * (L @ z)
                    
                    # Clip to bounds? 
                    # The problem says func clips. It is often better to 
                    # let the optimizer work in the unbounded space if 
                    # the function handles clipping, but here the domain 
                    # is strictly [-5, 5]. 
                    # Standard practice: clip the samples before evaluation 
                    # to respect the box constraint directly, or use 
                    # a transformation. Clipping is simpler and robust 
                    # for box constraints.
                    population[i] = np.clip(population[i], -5, 5)
                
                # 2. Evaluate population
                fvals = np.empty(lam)
                for i in range(lam):
                    try:
                        fvals[i] = wrapped_func(population[i])
                    except TimeoutError:
                        break
                
                # If we broke out of the loop due to timeout/budget, 
                # we might have partial fvals. We need to handle this.
                # However, the while loop condition checks budget/time.
                # If we hit the limit inside the inner loop, we break.
                
                if eval_count >= budget or time.time() > deadline:
                    break

                # 3. Sort by fitness
                indices = np.argsort(fvals)
                sorted_pop = population[indices]
                sorted_fvals = fvals[indices]
                
                # 4. Update mean (Weighted recombination)
                # Use top mu individuals
                new_mean = np.sum(weights[:mu, np.newaxis] * sorted_pop[:mu], axis=0)
                
                # 5. Update Covariance Matrix C
                # Rank-mu update
                # d = new_mean - mean
                d = new_mean - mean
                H = np.sum(weights[:mu, np.newaxis] * (sorted_pop[:mu] - mean)[:, np.newaxis] * (sorted_pop[:mu] - mean)[np.newaxis, :], axis=0)
                # Rank-1 update is also standard, but Rank-mu is better.
                # C = (1 - c1 - cmu) * C + c1 * (d d^T) + cmu * H
                # c1 = 2 / (dim + 2)^1.5
                # cmu = min(c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)^2 + mueff))
                
                c1 = 2.0 / (dim + 2)**1.5
                cmu = min(c1, 2.0 * (mueff - 2 + 1.0/mueff) / ((dim + 2)**2 + mueff))
                
                C = (1 - c1 - cmu) * C + c1 * np.outer(d, d) + cmu * H
                
                # Ensure symmetry and positive semi-definite
                C = (C + C.T) / 2.0
                # Add small regularization if needed
                C += 1e-6 * np.eye(dim)
                
                # 6. Update Step Size sigma
                # Use the 1/5 success rule on the step size? 
                # Or the standard CMA-ES hsig update.
                # Standard CMA-ES uses a path variable for step size adaptation.
                # For simplicity and robustness in this short budget, 
                # we can use a simpler heuristic: 
                # If the best fitness improved significantly, increase sigma slightly?
                # No, standard 1/5 rule is safer.
                
                # Let's implement the standard hsig mechanism for CMA-ES.
                # This requires maintaining path_sigma and path_c.
                # Since we are implementing from scratch, let's use a simplified 
                # step size adaptation based on the success of the last generation.
                
                # Success rate: how many of the top mu are better than the previous mean?
                # Or simply: if the best of the population is better than the current mean's best, 
                # it's hard to define "success" for sigma in CMA-ES directly without the path.
                
                # Alternative: Use the "1/5 rule" adapted for population.
                # If more than 1/5 of the population leads to improvement (relative to previous best?), 
                # increase sigma.
                
                # Actually, a very robust and simple alternative for short runs is 
                # to just use the standard CMA-ES step size update which relies on 
                # the expected norm of the step.
                
                # Let's stick to a simpler, effective strategy for this specific task:
                # The "1/5 rule" is for (1+1). For CMA-ES, the step size adapts 
                # to the covariance. If we keep C updated, sigma is less critical 
                # for exploration direction, but still scales the step.
                
                # Let's use a fixed sigma that decays very slowly or stays constant?
                # No, we need to converge.
                
                # Let's use the standard CMA-ES step size update.
                # We need to track the path_sigma.
                # This adds complexity. 
                
                # Given the constraints and the fact that Nelder-Mead failed, 
                # and CMA-ES is the gold standard, let's implement the 
                # standard path-based step size adaptation.
                
                # We need to store state from previous iteration.
                # This makes the code stateful.
                
                # To keep it simple and within the function scope without 
                # complex state management, let's use a slightly different 
                # but very effective variant: 
                # **Sep-CMA** or just **fixed sigma with C update**?
                # If sigma is fixed, the exploration radius is fixed. 
                # This might prevent convergence to the minimum.
                
                # Let's use a simple adaptive sigma:
                # If the best f in the population is better than the best f in the 
                # previous population, we might have found the valley.
                
                # Actually, the most robust simple approach is:
                # **CMA-ES with step size adaptation via the 1/5 rule on the mean step length**.
                # But that's complex.
                
                # Let's try a different structural change: 
                # **Powell's Method with a robust scout phase**.
                # The inspiration mentioned Powell + Scout outperformed (1+1)-ES.
                # The current code is Nelder-Mead.
                # Powell is deterministic and often faster for Rosenbrock 
                # once in the valley.
                
                # Strategy: 
                # 1. Scout: Use a small population-based search (like (1+1)-ES or 
                #    a few random restarts) to get into the basin of attraction.
                # 2. Refine: Use Powell's method to converge.
                
                # This is a hybrid approach, which is structurally different 
                # from pure Nelder-Mead.
                
                # However, the directive says "different design family".
                # Powell is a direct search method, like Nelder-Mead.
                # CMA-ES is an Evolution Strategy.
                
                # Let's stick with CMA-ES but implement the step size update 
                # properly using the path variables, storing them in local 
                # variables that persist across the while loop iterations.
                
                # I will rewrite the loop to be a proper CMA-ES implementation.
                
                # Reset for the next iteration's state variables
                # We need to define path_sigma and path_c before the loop?
                # No, they evolve.
                
                # Since I cannot easily insert state variables before the 
                # `while` loop in this draft structure without restructuring, 
                # I will move the state initialization before the loop.
                
                # (Self-Correction: I will restructure the code to initialize 
                # all CMA-ES state variables before the main loop.)
                
                # To avoid re-indenting the whole block in my head, I will 
                # output the full corrected code below.
                
                break # Break out of this draft loop to write the final code

        except Exception:
            pass

        return best_x, best_f

# The above draft is getting messy. I will write the clean version below.
# The key is to implement CMA-ES correctly with state persistence.

class Optimizer:
    """CMA-ES optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 12
        mu = lam // 2
        sigma = 2.0
        mean = np.zeros(dim)
        C = np.eye(dim)
        
        # Weights
        weights = np.log(np.arange(lam, 0, -1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights[:mu]**2)
        
        # Adaptation parameters
        c1 = 2.0 / (dim + 2)**1.5
        cmu = min(c1, 2.0 * (mueff - 2 + 1.0/mueff) / ((dim + 2)**2 + mueff))
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        
        # Path variables
        path_sigma = np.zeros(dim)
        path_c = np.zeros(dim)
        
        # Expected norm of N(0, I)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Evaluation counter
        eval_count = 0
        best_x = mean.copy()
        best_f = float('inf')
        
        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise TimeoutError
            
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            
            fx = func(x)
            eval_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        try:
            while eval_count < budget and time.time() < deadline:
                # 1. Sample
                L = np.linalg.cholesky(C)
                
                population = np.empty((lam, dim))
                for i in range(lam):
                    z = np.random.randn(dim)
                    population[i] = mean + sigma * (L @ z)
                    population[i] = np.clip(population[i], -5, 5)
                
                # 2. Evaluate
                fvals = np.empty(lam)
                for i in range(lam):
                    try:
                        fvals[i] = wrapped_func(population[i])
                    except TimeoutError:
                        break
                
                if eval_count >= budget or time.time() > deadline:
                    break

                # 3. Sort
                indices = np.argsort(fvals)
                sorted_pop = population[indices]
                sorted_fvals = fvals[indices]
                
                # 4. Update Mean
                new_mean = np.sum(weights[:mu, np.newaxis] * sorted_pop[:mu], axis=0)
                
                # 5. Update Step Size
                # Calculate step length
                # y_k = (new_mean - mean) / sigma
                y_k = (new_mean - mean) / sigma
                # Update path_sigma
                path_sigma = (1 - cs) * path_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_k
                # Update path_c
                path_c = (1 - cc) * path_c + np.sqrt(cc * (2 - cc) * mueff) * (L @ y_k)
                
                # Step size update
                # hsig = 1 if ||path_sigma|| / chi_n < (1 + 2/(dim+2)) * (1 - (1-cc)**(1/(lam*2))) else 0
                # Simplified:
                norm_path_sigma = np.linalg.norm(path_sigma)
                # The standard condition
                # We use a simple approximation for the threshold
                threshold = chi_n * (1 + 2.0/(dim+2.0))
                hsig = 1.0 if norm_path_sigma < threshold else 0.0
                
                sigma *= np.exp(cs / (dim + 2.0) * (norm_path_sigma / chi_n - 1.0))
                
                # 6. Update Covariance
                # Rank-mu update
                d = new_mean - mean
                H = np.sum(weights[:mu, np.newaxis] * (sorted_pop[:mu] - mean)[:, np.newaxis] * (sorted_pop[:mu] - mean)[np.newaxis, :], axis=0)
                
                # Correction for hsig
                hsig_delta = (1 - hsig) * cc * (2 - cc)
                
                C = (1 - c1 - cmu) * C + c1 * (np.outer(d, d) + hsig_delta * np.outer(path_c, path_c)) + cmu * H
                
                # Ensure symmetry
                C = (C + C.T) / 2.0
                C += 1e-6 * np.eye(dim)
                
                # Update mean
                mean = new_mean

        except Exception:
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
