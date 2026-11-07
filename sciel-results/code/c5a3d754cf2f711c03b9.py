import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290.0

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290.0
        
        def wrapped_func(x):
            self.call_count += 1
            if time.time() > self.deadline:
                raise TimeoutError
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            # Clip to bounds
            x = np.clip(x, -self.bound, self.bound)
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Phase 1: Global Search with Random Sampling
        # Use more points to increase probability of finding the global basin
        # 100 points is a reasonable trade-off for 5D
        n_global_samples = 100
        for _ in range(n_global_samples):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            p = np.random.uniform(-self.bound, self.bound, self.dim)
            wrapped_func(p)
            
        if self.best_x is None:
             # Fallback if no calls made? Should not happen
             return np.zeros(self.dim), 0.0

        # Phase 2: CMA-ES for rotation-aware local optimization
        # CMA-ES adapts its covariance matrix to the local landscape geometry,
        # which handles the hidden rotation in f21 better than Powell or Nelder-Mead
        try:
            self._run_cmaes(wrapped_func, self.best_x.copy())
        except TimeoutError:
            pass
        except RuntimeError:
            pass
        except Exception:
            pass

        # Phase 3: If budget remains, do a second CMA-ES restart from best point
        # with smaller sigma for fine-tuning
        if self.call_count < self.budget - 20 and time.time() < self.deadline - 10:
            try:
                self._run_cmaes_polish(wrapped_func, self.best_x.copy())
            except TimeoutError:
                pass
            except RuntimeError:
                pass
            except Exception:
                pass

        return self.best_x, self.best_f

    def _run_cmaes(self, func, x0):
        """Run CMA-ES with adaptive strategy."""
        dim = self.dim
        # Initial sigma: start with moderate step size
        sigma = 1.0
        # Population size: lambda = 4 + floor(3 * ln(dim))
        lam = max(4 + int(3 * np.log(dim)), 6)
        
        # CMA-ES state
        mean = x0.copy()
        sigma = 1.0
        # Covariance matrix C (identity initially)
        C = np.eye(dim)
        # Step sizes for axes
        damps = 1.0 / (dim + 3)  # damping for evolution paths
        cs = (dim + 1.7) / (dim + dim + 2.5)  # step-size adaptation
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, (2 * (mu_eff - 1) / ((dim + 2) ** 2 + 2)))
        # mu_eff for ranking
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Evolution paths
        ps = np.zeros(dim)
        ph = np.zeros(dim)
        
        # Cholesky factor of C
        Csqrt = np.linalg.cholesky(C)
        
        while self.call_count < self.budget and time.time() < self.deadline:
            # Generate population
            population = []
            fitnesses = []
            for i in range(lam):
                if self.call_count >= self.budget or time.time() > self.deadline:
                    break
                z = np.random.randn(dim)
                x = mean + sigma * (Csqrt @ z)
                fx = func(x)
                population.append(x)
                fitnesses.append(fx)
            
            if len(population) < mu:
                break
            
            # Sort by fitness
            order = np.argsort(fitnesses)
            best_indices = order[:mu]
            best_x = population[best_indices[0]]
            best_f = fitnesses[best_indices[0]]
            
            # Update mean (weighted)
            new_mean = np.zeros(dim)
            for i, idx in enumerate(best_indices):
                new_mean += weights[i] * population[idx]
            mean = new_mean
            
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (Csqrt.T @ (mean - x0) / sigma)
            ph_norm = np.linalg.norm(ps / np.sqrt(1 - (1 - cs) ** 2))
            if ph_norm < (1.4 + 1 / dim) * np.sqrt((dim + 1) * (1 - (1 - cs) ** 2)):
                C = (1 - c1 - cmu) * C + c1 * (ps @ ps + (1 if ph_norm > (1.4 + 1/dim) * np.sqrt((dim+1)*(1-(1-cs)**2)) else 0) * np.eye(dim))
                for i in range(mu):
                    y = (population[best_indices[i]] - x0) / sigma
                    C += cmu * weights[i] * (y @ y)
            
            # Update sigma
            sigma *= np.exp(cs / (dim + 3) * (ph_norm / np.sqrt((dim + 1) * (1 - (1 - cs) ** 2)) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update Cholesky
            try:
                Csqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-10 * np.eye(dim)
                Csqrt = np.linalg.cholesky(C)
            
            # Check for convergence
            if sigma < 1e-10 or np.linalg.norm(mean - x0) < 1e-12:
                break
        
        return mean, sigma

    def _run_cmaes_polish(self, func, x0):
        """Run a shorter CMA-ES polish with smaller initial sigma."""
        dim = self.dim
        lam = max(4 + int(3 * np.log(dim)), 6)
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        
        mean = x0.copy()
        sigma = 0.1  # Smaller for polish
        C = np.eye(dim)
        Csqrt = np.linalg.cholesky(C)
        ps = np.zeros(dim)
        cs = (dim + 1.7) / (dim + dim + 2.5)
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, (2 * (mu_eff - 1) / ((dim + 2) ** 2 + 2)))
        
        max_iter = 50
        for _ in range(max_iter):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            
            population = []
            fitnesses = []
            for i in range(lam):
                if self.call_count >= self.budget or time.time() > self.deadline:
                    break
                z = np.random.randn(dim)
                x = mean + sigma * (Csqrt @ z)
                fx = func(x)
                population.append(x)
                fitnesses.append(fx)
            
            if len(population) < mu:
                break
            
            order = np.argsort(fitnesses)
            best_indices = order[:mu]
            
            new_mean = np.zeros(dim)
            for i, idx in enumerate(best_indices):
                new_mean += weights[i] * population[idx]
            mean = new_mean
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (Csqrt.T @ (mean - x0) / sigma)
            ph_norm = np.linalg.norm(ps / np.sqrt(1 - (1 - cs) ** 2))
            if ph_norm < (1.4 + 1 / dim) * np.sqrt((dim + 1) * (1 - (1 - cs) ** 2)):
                C = (1 - c1 - cmu) * C + c1 * (ps @ ps)
                for i in range(mu):
                    y = (population[best_indices[i]] - x0) / sigma
                    C += cmu * weights[i] * (y @ y)
            
            sigma *= np.exp(cs / (dim + 3) * (ph_norm / np.sqrt((dim + 1) * (1 - (1 - cs) ** 2)) - 1))
            sigma = np.clip(sigma, 1e-10, 1.0)
            
            try:
                Csqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-10 * np.eye(dim)
                Csqrt = np.linalg.cholesky(C)
            
            if sigma < 1e-12:
                break
        
        return mean, sigma
