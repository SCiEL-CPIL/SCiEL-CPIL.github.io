import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        N = 15  # Population size
        sigma = 2.0  # Initial step size
        c1 = (1 + 2 / (self.dim + 1)) / (self.dim + 4 + 2 / (self.dim + 1))
        cmu = (2 + 2 / (self.dim + 1)) / (self.dim + 4 + 2 / (self.dim + 1))
        cc = (4 + self.dim / 4) / (self.dim + 4 + 4 / self.dim)
        cs = (1 + 2 / (self.dim + 1) + cmu) / (self.dim + cc + 5)
        damp = 0.5 + 2 / (self.dim + 1)
        
        # Initialize
        x_mean = np.zeros(self.dim)
        B = np.eye(self.dim)
        D = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chi_n = np.sqrt(self.dim) * (1 + 1 / (2 * self.dim) - 1 / (8 * self.dim ** 2))
        
        # Evaluate initial population
        def evaluate_population(x_mean, sigma, B, D):
            z = rng.standard_normal((N, self.dim))
            population = x_mean + sigma * (z @ (D @ B).T)
            fitnesses = np.array([func(x) for x in population])
            return population, fitnesses
        
        population, fitnesses = evaluate_population(x_mean, sigma, B, D)
        evals = N
        
        # Sort population by fitness
        order = np.argsort(fitnesses)
        population = population[order]
        fitnesses = fitnesses[order]
        
        # Track best
        global_best = population[0].copy()
        global_best_fitness = fitnesses[0]
        
        while evals < self.budget:
            # Update mean
            weights = np.zeros(N)
            weights[:N // 2 + 1] = np.log(N / 2 + 1) - np.log(np.arange(1, N // 2 + 2))
            weights /= weights.sum()
            x_mean = np.sum(weights[:, None] * population, axis=0)
            
            # Update evolution path for covariance matrix adaptation
            old_B = B
            old_D = D
            C = B @ (D ** 2) @ B.T
            C_sqrt = B @ D @ B.T
            
            # Update path for covariance
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * C_sqrt_inv @ (x_mean - old_mean)
            
            # Update path for step-size control
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_sqrt_inv @ (x_mean - old_mean)
            
            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            delta_sigma = (1 - cc) * (np.linalg.norm(pc) ** 2 / cc - 1)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (delta_sigma - 1) * C)
            
            # Add rank-mu update
            y = np.linalg.solve(C_sqrt, x_mean - old_mean)
            for i in range(N // 2 + 1):
                C += cmu * weights[i] * np.outer(y, y)
            
            # Update B and D via eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            B = eigenvectors
            D = np.diag(np.sqrt(eigenvalues))
            
            # Evaluate new population
            old_mean = x_mean.copy()
            population, fitnesses = evaluate_population(x_mean, sigma, B, D)
            evals += N
            
            # Sort population by fitness
            order = np.argsort(fitnesses)
            population = population[order]
            fitnesses = fitnesses[order]
            
            # Update best
            if fitnesses[0] < global_best_fitness:
                global_best_fitness = fitnesses[0]
                global_best = population[0].copy()
        
        return global_best, global_best_fitness
