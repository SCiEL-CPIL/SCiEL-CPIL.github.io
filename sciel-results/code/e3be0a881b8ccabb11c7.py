import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        time_limit = 290.0

        best_x = None
        best_f = float('inf')
        calls = 0

        # 1. Initial Exploration
        # Use a fixed seed for reproducibility, but mix in some randomness?
        # The task says "hidden seeded shift", implying the function itself is fixed per instance.
        # So deterministic exploration is fine.
        np.random.seed(42)
        init_points = [np.zeros(n)]
        # Add points along axes to help gradient estimation if needed, but CMA-ES handles this.
        for _ in range(10):
            init_points.append(np.random.uniform(-5, 5, n))
        
        for x in init_points:
            if time.time() - start_time > time_limit or calls >= budget:
                break
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1

        # 2. CMA-ES Optimization
        # CMA-ES is robust, derivative-free, and adapts to the landscape.
        # It is particularly effective for non-separable or rotated quadratic functions like the shifted/rotated Sphere.
        
        # Parameters
        # Population size: lambda = 4 + 3*ln(n)
        lambda_ = 4 + int(3 * np.log(n))
        # Number of selected points
        mu = lambda_ // 2
        
        # Weights for the selected points
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        
        # Effective population size
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0) - 1.0)) + cs
        
        # Initial mean and step size
        # Start from the best point found in exploration
        mean = best_x.copy()
        # Initial step size: large enough to explore, but not too large to miss the basin.
        # Since minimizer is in [-4,4]^5 and box is [-5,5]^5, sigma=1.0 or 2.0 is reasonable.
        # Let's start with 1.0 to be safe, CMA-ES will adapt.
        sigma = 1.0
        
        # Covariance matrix (initially identity)
        C = np.eye(n)
        
        # Eigen decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.sqrt(np.maximum(eigvals, 1e-16))
        
        # Cumulative step size
        ps = np.zeros(n)
        # Covariance adaptation
        pc = np.zeros(n)
        
        # Best so far is already set
        
        # Precompute weight square root for sampling
        sqrt_weights = np.sqrt(weights)
        
        # Generation counter for time-varying parameters if needed, but standard CMA-ES uses fixed rates.
        gen = 0
        
        while calls < budget:
            if time.time() - start_time > time_limit:
                break
            
            # Generate population
            population = np.zeros((lambda_, n))
            f_values = np.zeros(lambda_)
            
            for i in range(lambda_):
                # Sample from N(0, C)
                z = np.random.randn(n)
                y = eigvecs @ (eigvals * z)
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                population[i] = x
                f_values[i] = func(x)
                calls += 1
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
            
            # Sort by function value
            order = np.argsort(f_values)
            population = population[order]
            f_values = f_values[order]
            
            # Update best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = population[0].copy()
            
            # Select top mu points
            selected = population[:mu]
            
            # New mean
            new_mean = np.sum(weights[:, np.newaxis] * selected, axis=0)
            
            # Step vector
            y = (new_mean - mean) / sigma
            
            # Cumulative step size adaptation
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * y
            
            # hsig: indicator if ps is within expected range
            # The term 1 - (1-cs)^(2/gen) approximates the expected norm.
            # For early generations, gen is small. Let's use a stable version.
            # Standard formula:
            # hsig = norm(ps) / sqrt(1 - (1-cs)^(2/gen)) < (1.3 + 2/(n+1)) * E||N(0,I)||
            # E||N(0,I)|| ~ sqrt(n) * (1 - 1/(4n) + ...)
            
            # To avoid division by zero or instability in early generations:
            # We can simply check if the step is reasonable.
            # A common implementation uses:
            # hsig = np.linalg.norm(ps) < (1.3 + 2.0 / (n + 1.0)) * np.sqrt(1.0 - (1.0 - cs) ** (2.0 / max(1, gen))) * np.sqrt(n)
            
            # Let's use a simpler heuristic for hsig to avoid numerical issues:
            # If the step is not too large compared to sigma, it's likely a good direction.
            # Actually, the standard CMA-ES implementation in Python (cma library) is robust.
            # Let's stick to the standard formula but guard against gen=0.
            
            if gen == 0:
                hsig = True
            else:
                expected_norm = np.sqrt(1.0 - (1.0 - cs) ** (2.0 / gen)) * np.sqrt(n)
                hsig = np.linalg.norm(ps) < (1.3 + 2.0 / (n + 1.0)) * expected_norm
            
            # Covariance matrix adaptation
            if hsig:
                pc = (1.0 - cs) * pc + cs * y
            
            # Rank-one and rank-mu updates
            delta_hs = (1.0 if hsig else 0.0) * cs * (2.0 - cs)
            C = (1.0 - cs * (c1 + cmu) - delta_hs) * C
            C += (1.0 + c1 * (1.0 / c1 - 1.0) - cs) * np.outer(pc, pc)
            
            # Rank-mu update
            for i in range(mu):
                zi = (selected[i] - mean) / sigma
                C += cmu * weights[i] * np.outer(zi, zi)
            
            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-16)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            eigvals = np.sqrt(eigvals)
            
            # Step size adaptation
            if gen == 0:
                sigma_change = 1.0
            else:
                expected_norm = np.sqrt(1.0 - (1.0 - cs) ** (2.0 / gen)) * np.sqrt(n)
                sigma_change = np.exp(cs / damps * (np.linalg.norm(ps) / expected_norm - 1.0))
            
            sigma *= sigma_change
            
            # Update mean
            mean = new_mean
            mean = np.clip(mean, -5, 5)
            
            # Convergence check
            if sigma < 1e-10:
                break
            
            gen += 1

        return best_x, best_f
