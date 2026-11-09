"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES based black-box optimizer for Schwefel-style f20 on [-5,5]^5.
    Uses a standard CMA-ES implementation with boundary handling via reflection.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We target finishing in < 10s.
        # CMA-ES is fast for dim=5.
        deadline = start_time + 5.0 
        
        dim = self.dim
        budget = self.budget
        lo, hi = -5.0, 5.0
        
        # CMA-ES Parameters
        # Lambda (population size)
        lam = 4 + int(3 * np.log(dim))  # Standard rule: ~13 for dim=5
        # Sigma (initial step size)
        # Schwefel landscape is rough. Start moderately large to cover space.
        sigma = 2.0 
        # x0 (initial mean)
        # Center is a reasonable guess, or we could randomize. 
        # Given the hidden shift, center is safer than corner.
        x0 = np.zeros(dim)
        
        # CMA-ES State
        # Mean
        m = x0.copy()
        # Covariance matrix (initially identity)
        C = np.eye(dim)
        # Inverse square root of C (for sampling)
        # For dim=5, we can just use Cholesky or SVD. 
        # Standard CMA-ES uses eigen-decomposition or Cholesky.
        # Let's use Cholesky for speed if C is PD.
        try:
            B = np.linalg.cholesky(C)
        except:
            B = np.eye(dim)
        
        # Adaptation rates
        mu = lam // 2  # Weights for top mu individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff + 2) / (dim + 2) - 1) / ((dim + 1.3)**2 + mueff))
        
        # Path lengths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Evaluation counter
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                raise ValueError("Budget exceeded")
            calls_made += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Initial evaluation of mean? 
        # CMA-ES usually starts by evaluating the population.
        # We don't need to evaluate m separately if it's in the population, 
        # but often it's good to check. 
        # To save budget, we just evaluate the population.
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            for i in range(lam):
                if calls_made >= budget:
                    break
                z = np.random.randn(dim)
                y = m + sigma * (B @ z)
                ys.append(y)
            
            if not ys:
                break

            # Evaluate population
            fitnesses = []
            for y in ys:
                if calls_made >= budget:
                    break
                # Boundary handling: Clip to box. 
                # Note: Simple clipping can cause issues with CMA-ES distribution,
                # but for [-5,5]^5 it's usually acceptable for a short run.
                y_clipped = np.clip(y, lo, hi)
                f = call_func(y_clipped)
                fitnesses.append(f)
            
            if len(fitnesses) < mu:
                break

            # Sort by fitness
            sorted_indices = np.argsort(fitnesses)
            y_sorted = [ys[i] for i in sorted_indices[:mu]]
            f_sorted = [fitnesses[i] for i in sorted_indices[:mu]]
            
            # Update mean
            m_new = np.sum(np.array(y_sorted) * weights[:, np.newaxis], axis=0)
            
            # Update covariance matrix
            # y_k = (y_k - m) / sigma
            y_diffs = (np.array(y_sorted) - m) / sigma
            delta_s = (np.sum(weights * y_diffs, axis=0)) / np.sqrt(np.sum(weights**2))
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * delta_s
            
            # Hessian matrix update term
            # H = sum w_i * y_i y_i^T
            H = np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(np.array(y_sorted), np.array(y_sorted)), axis=0)
            
            # Update p_c
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((1 - c_sigma) / (1 + c_sigma))) else 0.0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * delta_s
            
            # Update C
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_mu * (np.outer(p_c, p_c) - C) + c_1 * (H - C)
            
            # Ensure C is symmetric and PD
            C = (C + C.T) / 2.0
            # Check eigenvalues for PD
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += 1e-12 * np.eye(dim)
            
            # Update sigma
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(2 * dim) - 1))
            
            # Update mean
            m = m_new
            
            # Update B (Cholesky)
            try:
                B = np.linalg.cholesky(C)
            except:
                # If not PD, reset to identity or use SVD
                # For robustness, just reset to identity if it fails
                C = np.eye(dim)
                B = np.eye(dim)
                sigma = max(sigma * 0.5, 0.1)

            # Termination criteria
            if sigma < 1e-6:
                break
            
            # If best fitness is very good, we might stop early? 
            # No, keep going to refine.

        return best_x, best_f
# EVOLVE-BLOCK-END
