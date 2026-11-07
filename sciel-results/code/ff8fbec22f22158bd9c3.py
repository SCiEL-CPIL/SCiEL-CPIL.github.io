import numpy as np
import time

class Optimizer:
    """Hybrid CMA-ES with Gaussian Process Infill for rotated Weierstrass landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin for 330s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Track state
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Data storage for GP
        X_data = []
        Y_data = []
        
        # Helper to call function and record
        def evaluate(x):
            nonlocal calls_made, best_x, best_f, X_data, Y_data
            if calls_made >= budget:
                raise ValueError("Budget Exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time Limit")
            
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            calls_made += 1
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            
            X_data.append(x)
            Y_data.append(val)
            
            return val
        
        # --- CMA-ES Parameters ---
        # Population size lambda
        # For dim=5, standard is 4 + 3*ln(5) ~ 9. Let's use 10 for robustness.
        # However, for the hybrid approach, we want to be efficient.
        # Let's start with a moderate population size.
        lambda_ = 10
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        m = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Eigen decomposition cache
        eigvals = np.ones(dim)
        eigvecs = np.eye(dim)
        
        def sample_cma(m, sigma, C, n_samples):
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                L = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            
            z = rng.standard_normal((n_samples, dim))
            samples = m + sigma * (z @ L.T)
            return samples
        
        def update_cma(pop, pop_fitness, m, sigma, C, p_sigma, p_c, gen):
            # Sort
            idx = np.argsort(pop_fitness)
            pop = pop[idx]
            pop_fitness = pop_fitness[idx]
            
            # Select best mu
            y_best = pop[:mu]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(weights[:, np.newaxis] * y_best, axis=0)
            
            # Update eigendecomposition periodically
            if gen % 5 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
            
            C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update p_sigma
            delta_m = (m - m_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ delta_m)
            
            # h_sigma
            phi = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(32*dim**2))
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 2/(dim+1)) * phi else 0.0
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(p_sigma) / phi - 1))
            
            # Update p_c
            y_norm = (y_best - m_old) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (m - m_old) / sigma)
            
            # Rank-1 and Rank-mu updates
            C = (1 + c_1 * h_sigma * (mu_eff - 1) - c_1 - c_mu) * C \
                + c_1 * (p_c @ p_c - C) \
                + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * (y_norm[:, np.newaxis, :] * y_norm[:, np.newaxis, :].T), axis=0)
            
            return m, sigma, C, p_sigma, p_c, eigvals, eigvecs

        # --- GP Infill Parameters ---
        # We will use GP to select the next point after every CMA-ES generation or every K evaluations.
        # To keep it simple and robust, we will do a few CMA-ES generations, then use GP to pick 1 point,
        # then continue CMA-ES? Or alternate?
        # Strategy: 
        # 1. Run CMA-ES for a few generations (e.g., 2 gens = 20 evals).
        # 2. Use GP to pick the next single point based on EI.
        # 3. Repeat.
        
        # GP Kernel: Matern 5/2 is good for smooth-ish but not perfectly smooth functions like Weierstrass.
        # Length scales: Initialize to 1.0.
        
        def gp_predict(X_new, X_train, Y_train, length_scales, variance):
            """
            Simple GP prediction.
            Returns mean and variance.
            """
            if len(X_train) == 0:
                return np.zeros(len(X_new)), np.ones(len(X_new)) * variance
                
            K = np.zeros((len(X_train), len(X_train)))
            for i in range(len(X_train)):
                for j in range(len(X_train)):
                    dist_sq = np.sum((X_train[i] - X_train[j]) ** 2) / length_scales
                    K[i, j] = variance * np.sqrt(5 * dist_sq) * np.exp(-np.sqrt(5 * dist_sq))
            
            # Add noise for numerical stability
            K += 1e-6 * np.eye(len(X_train))
            
            # Solve for weights
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                K_inv = np.linalg.pinv(K)
                
            alpha = K_inv @ Y_train
            
            # Predict at X_new
            K_star = np.zeros((len(X_new), len(X_train)))
            for i in range(len(X_new)):
                for j in range(len(X_train)):
                    dist_sq = np.sum((X_new[i] - X_train[j]) ** 2) / length_scales
                    K_star[i, j] = variance * np.sqrt(5 * dist_sq) * np.exp(-np.sqrt(5 * dist_sq))
                    
            mean = K_star @ alpha
            var = variance - np.sum(K_star @ K_inv * K_star, axis=1)
            var = np.clip(var, 1e-8, None)
            
            return mean, var

        def expected_improvement(mean, var, best_f, xi=0.01):
            """
            Calculate Expected Improvement.
            """
            eps = 1e-8
            var = np.clip(var, eps, None)
            z = (best_f - mean - xi) / np.sqrt(var)
            # Standard normal CDF and PDF
            # Approximation or use scipy if available, but let's stick to numpy
            # CDF: 0.5 * (1 + erf(z/sqrt(2)))
            # PDF: 1/sqrt(2*pi) * exp(-z^2/2)
            
            # Using normal cdf approximation
            cdf_z = 0.5 * (1 + np.vectorize(lambda x: 0.5 * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x**2))))(z))
            pdf_z = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            
            ei = (best_f - mean - xi) * cdf_z + np.sqrt(var) * pdf_z
            return ei

        # Main Loop
        gen = 0
        # We will alternate: 
        # - Run 1 CMA-ES generation (10 evals)
        # - Run 1 GP Infill step (1 eval)
        # This uses 11 evals per cycle.
        
        # Initial population for CMA-ES
        pop = sample_cma(m, sigma, C, lambda_)
        pop_fitness = np.full(lambda_, np.inf)
        
        for i in range(lambda_):
            try:
                pop_fitness[i] = evaluate(pop[i])
            except (ValueError, TimeoutError):
                break
        
        if np.any(np.isinf(pop_fitness)):
            if best_x is not None:
                return best_x, best_f
            else:
                return np.zeros(dim), np.inf

        # GP Hyperparameters (simple fixed for now, could be optimized)
        length_scales = 1.0
        variance = 100.0  # Initial variance estimate
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # 1. CMA-ES Generation
            # Check if we have enough budget for a full generation
            if calls_made + lambda_ > budget:
                break
                
            # Generate new population
            new_pop = sample_cma(m, sigma, C, lambda_)
            new_fitness = np.full(lambda_, np.inf)
            
            for i in range(lambda_):
                try:
                    new_fitness[i] = evaluate(new_pop[i])
                except (ValueError, TimeoutError):
                    break
            
            if np.any(np.isinf(new_fitness)):
                break
                
            # Update CMA-ES state
            m, sigma, C, p_sigma, p_c, eigvals, eigvecs = update_cma(
                new_pop, new_fitness, m, sigma, C, p_sigma, p_c, gen
            )
            gen += 1
            
            # 2. GP Infill Step
            # Use the GP to find a promising point
            # We need a set of candidate points to evaluate EI on.
            # Let's sample 100 random points in the box and pick the best EI.
            # Also include the current mean and best point neighborhood?
            # Random sampling in the box is a safe exploration strategy.
            
            if calls_made >= budget:
                break
                
            # Generate candidates
            n_candidates = 100
            candidates = rng.uniform(-5, 5, size=(n_candidates, dim))
            
            # Predict
            mean, var = gp_predict(candidates, np.array(X_data), np.array(Y_data), length_scales, variance)
            
            # Calculate EI
            ei = expected_improvement(mean, var, best_f)
            
            # Pick best candidate
            best_idx = np.argmax(ei)
            next_point = candidates[best_idx]
            
            try:
                evaluate(next_point)
            except (ValueError, TimeoutError):
                break
        
        return best_x, best_f
