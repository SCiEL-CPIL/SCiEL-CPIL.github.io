import numpy as np
import time
from scipy.optimize import minimize

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
        n_global_samples = 60
        for _ in range(n_global_samples):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            p = np.random.uniform(-self.bound, self.bound, self.dim)
            wrapped_func(p)
            
        if self.best_x is None:
             return np.zeros(self.dim), 0.0

        # Phase 2: Run CMA-ES from best point
        try:
            self._run_cmaes(wrapped_func, self.best_x.copy(), sigma_init=1.0)
        except (TimeoutError, RuntimeError):
            pass
        except Exception:
            pass

        # Phase 3: Final polish with CMA-ES
        if self.call_count < self.budget - 15 and time.time() < self.deadline - 5:
            try:
                self._run_cmaes_polish(wrapped_func, self.best_x.copy())
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return self.best_x, self.best_f

    def _run_cmaes(self, func, x0, sigma_init=1.0):
        """Run CMA-ES with correct standard updates."""
        dim = self.dim
        lam = max(4 + int(3 * np.log(dim)), 6)
        mu = lam // 2
        
        # Compute weights and mu_eff FIRST
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # CMA-ES parameters
        cs = (dim + 1.7) / (dim + dim + 2.5)
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, (2 * (mu_eff - 1) / ((dim + 2) ** 2 + 2)))
        
        # State
        mean = x0.copy()
        sigma = sigma_init
        C = np.eye(dim)
        Csqrt = np.linalg.cholesky(C)
        ps = np.zeros(dim)
        
        while self.call_count < self.budget and time.time() < self.deadline:
            old_mean = mean.copy()
            
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
            
            # Update mean (weighted recombination)
            new_mean = np.zeros(dim)
            for i, idx in enumerate(best_indices):
                new_mean += weights[i] * population[idx]
            
            # Update evolution path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (Csqrt.T @ (new_mean - old_mean) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) < (1.4 + 1.0 / dim) * np.sqrt(dim + 1 - 1))
            
            # Update covariance
            C = (1 - c1 - cmu) * C
            if hsig:
                C += c1 * (ps @ ps + np.eye(dim))
            else:
                C += c1 * (ps @ ps)
            for i in range(mu):
                y = (population[best_indices[i]] - old_mean) / sigma
                C += cmu * weights[i] * (y @ y)
            
            mean = new_mean
            
            # Update sigma
            chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
            sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)
            
            # Update Cholesky with regularization
            try:
                Csqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = C + 1e-10 * np.eye(dim)
                Csqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            
            # Convergence check
            if sigma < 1e-10:
                break
            if np.all(np.abs(mean - old_mean) < 1e-12):
                break
        
        return mean, sigma

    def _run_cmaes_polish(self, func, x0):
        """Run a shorter CMA-ES polish with smaller initial sigma."""
        dim = self.dim
        lam = max(4 + int(3 * np.log(dim)), 6)
        mu = lam // 2
        
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        cs = (dim + 1.7) / (dim + dim + 2.5)
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, (2 * (mu_eff - 1) / ((dim + 2) ** 2 + 2)))
        
        mean = x0.copy()
        sigma = 0.1
        C = np.eye(dim)
        Csqrt = np.linalg.cholesky(C)
        ps = np.zeros(dim)
        
        max_iter = 40
        for _ in range(max_iter):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            
            old_mean = mean.copy()
            
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
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (Csqrt.T @ (new_mean - old_mean) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) < (1.4 + 1.0 / dim) * np.sqrt(dim + 1 - 1))
            
            C = (1 - c1 - cmu) * C
            if hsig:
                C += c1 * (ps @ ps + np.eye(dim))
            else:
                C += c1 * (ps @ ps)
            for i in range(mu):
                y = (population[best_indices[i]] - old_mean) / sigma
                C += cmu * weights[i] * (y @ y)
            
            mean = new_mean
            
            chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
            sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-14, 1.0)
            
            try:
                Csqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = C + 1e-10 * np.eye(dim)
                Csqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            
            if sigma < 1e-14:
                break
            if np.all(np.abs(mean - old_mean) < 1e-14):
                break
        
        return mean, sigma
