import numpy as np
import time

class Optimizer:
    """CMA-ES based Optimizer for Rosenbrock with Latin Hypercube initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 330s limit, use 320s
        self.deadline = self.start_time + 320.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # Phase 1: Latin Hypercube Sampling for global exploration
        # Use LHS to spread initial points across the search space more evenly than pure random
        n_lhs = 30
        rng = np.random.default_rng(42)
        
        # Generate LHS points in [-4, 5]^5
        # For each dimension, divide into n_lhs intervals and pick one random point per interval
        lhs_points = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            intervals = np.linspace(-4, 5, n_lhs + 1)
            for i in range(n_lhs):
                lhs_points[i, d] = rng.uniform(intervals[i], intervals[i+1])
        
        # Evaluate LHS points
        for x in lhs_points:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            try:
                f = wrapped_func(x)
            except StopIteration:
                break
            except Exception:
                continue
        
        # Phase 2: CMA-ES from the best point found
        # Implement a simple CMA-ES variant
        if self.best_x is None:
            x0 = np.ones(self.dim)
            try:
                self.best_f = wrapped_func(x0)
                self.best_x = x0.copy()
            except:
                pass
        
        if self.best_x is not None and self.calls_made < self.budget:
            # Initialize CMA-ES
            x_mean = self.best_x.copy()
            sigma = 1.0  # Initial step size
            C = np.eye(self.dim)  # Covariance matrix
            C_inv = np.eye(self.dim)  # Inverse covariance matrix
            C_sqrt = np.eye(self.dim)  # Square root of covariance matrix
            
            # CMA-ES parameters
            pop_size = 2 * (4 + int(np.log(self.dim)))  # Standard CMA-ES population size
            if pop_size > self.budget - self.calls_made:
                pop_size = max(5, (self.budget - self.calls_made) // 2)
            
            # Adaptation rates
            c_sigma = (pop_size + 2) / (self.dim + pop_size + 5)
            c_c = (4 + pop_size / self.dim) / (self.dim + 4 + 2 * pop_size / self.dim)
            c_1 = 2 / ((self.dim + 1.3)**2 + pop_size)
            c_mu = min(1 - c_1, 2 * ((pop_size/2 + 1) - self.dim - 2) / ((self.dim + 2)**2 + pop_size))
            d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (self.dim + 1)) - 1) + c_sigma
            
            p_sigma = np.zeros(self.dim)
            p_c = np.zeros(self.dim)
            
            # Eigen decomposition cache
            eigvals, eigvecs = np.linalg.eigh(C)
            
            max_iterations = (self.budget - self.calls_made) // pop_size
            if max_iterations < 1:
                max_iterations = 1
            
            for it in range(max_iterations):
                if self.calls_made >= self.budget:
                    break
                if time.time() > self.deadline:
                    break
                
                # Generate population
                population = []
                for i in range(pop_size):
                    if self.calls_made >= self.budget:
                        break
                    if time.time() > self.deadline:
                        break
                    
                    # Generate random normal
                    z = rng.standard_normal(self.dim)
                    # Transform using Cholesky-like decomposition
                    # For simplicity, use eigendecomposition
                    x_new = x_mean + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                    x_new = np.clip(x_new, -5, 5)
                    population.append(x_new)
                
                if len(population) < 2:
                    break
                
                # Evaluate population
                evals = []
                for x in population:
                    try:
                        f = wrapped_func(x)
                        evals.append(f)
                    except StopIteration:
                        break
                    except Exception:
                        evals.append(np.inf)
                
                if len(evals) < len(population):
                    population = population[:len(evals)]
                    evals = evals[:len(evals)]
                
                if len(population) < 2:
                    break
                
                # Sort by fitness
                sorted_indices = np.argsort(evals)
                population = [population[i] for i in sorted_indices]
                evals = [evals[i] for i in sorted_indices]
                
                # Update best
                if evals[0] < self.best_f:
                    self.best_f = evals[0]
                    self.best_x = population[0].copy()
                
                # Weighted mean (top half)
                mu_eff = int(pop_size / 2)
                weights = np.zeros(mu_eff)
                w_sum = 0
                for i in range(mu_eff):
                    weights[i] = np.log(mu_eff + 1) - np.log(i + 1)
                    w_sum += weights[i]
                weights /= w_sum
                
                x_new_mean = np.zeros(self.dim)
                for i in range(mu_eff):
                    x_new_mean += weights[i] * population[i]
                
                # Update path accumulators
                if eigvals[0] > 1e-16:
                    C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                    p_sigma = (1 - c_sigma) * p_sigma + c_sigma * d_sigma * C_sqrt_inv @ (x_new_mean - x_mean) / sigma
                    h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1))) < (1.4 + 2 / (self.dim + 1)) * np.trace(C_inv) / self.dim else 0.0
                    p_c = (1 - c_c) * p_c + h_sigma * c_c * (x_new_mean - x_mean) / sigma
                else:
                    h_sigma = 1.0
                    p_c = (1 - c_c) * p_c + c_c * (x_new_mean - x_mean) / sigma
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_mu_update = np.zeros((self.dim, self.dim))
                for i in range(mu_eff):
                    y_i = C_sqrt_inv @ (population[i] - x_mean) / sigma
                    rank_mu_update += weights[i] * np.outer(y_i, y_i)
                
                C = (1 + c_1 * (c_mu - c_1) + c_mu * (2 - c_1)) * C \
                    - c_1 * np.outer(p_c, p_c) \
                    + c_mu * rank_mu_update \
                    - delta_h * c_1 * np.outer(p_c, p_c)
                
                # Symmetrize
                C = (C + C.T) / 2
                
                # Update eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                eigvals = np.maximum(eigvals, 1e-10)
                C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                
                # Update sigma
                if eigvals[0] > 1e-16:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(self.dim) - 1))
                else:
                    sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(self.dim) - 1))
                
                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 10.0)
                
                # Update mean
                x_mean = x_new_mean
                
                # Check for convergence
                if sigma < 1e-10:
                    break

        # Fallback if no valid result
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            try:
                self.best_f = func(np.clip(x0, -5, 5))
            except:
                self.best_f = np.inf

        return self.best_x, self.best_f
