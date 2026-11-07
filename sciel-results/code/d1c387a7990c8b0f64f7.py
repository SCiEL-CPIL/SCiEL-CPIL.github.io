"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_val = np.inf
        self.best_x = None
        self.start_time = None
        self.deadline = None

    def _call_func(self, func, x):
        self.call_count += 1
        val = func(x)
        if val < self.best_val:
            self.best_val = val
            self.best_x = np.array(x, copy=True)
        return val

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 280.0  # Safety margin
        self.call_count = 0
        self.best_val = np.inf
        self.best_x = None

        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lambda_pop = 4 + int(3 * np.log(dim))
        
        # Weights for recombination
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance equivalent population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size initialization
        # Heuristic for box [-5, 5]^dim, target to cover the box
        sigma = 2.0 
        
        # Mean
        m = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Eigenvalue decomposition for efficient sampling and updates
        D, B = np.linalg.eigh(C)
        D = np.sqrt(D)
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Damping for step size
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Threshold for hsig and hsig
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
        
        # Termination criteria
        max_iter = budget // lambda_pop + 10
        stall_counter = 0
        stall_threshold = 100 # No improvement in 100 evaluations
        
        best_val_local = np.inf
        best_x_local = None
        
        for it in range(max_iter):
            if self.call_count >= budget or time.time() > self.deadline:
                break
            
            # Generate population
            # Sample from N(0, C) using B @ (D * Z)
            Z = np.random.standard_normal((lambda_pop, dim))
            # Transform to N(0, C)
            Y = (B * D) @ Z.T
            Y = Y.T
            
            # New candidates
            X = m + sigma * Y
            
            # Evaluate population
            F = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                if self.call_count >= budget or time.time() > self.deadline:
                    break
                F[i] = self._call_func(func, X[i])
            
            if self.call_count < lambda_pop:
                break
                
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Check for improvement
            if F_sorted[0] < best_val_local - 1e-9:
                best_val_local = F_sorted[0]
                best_x_local = X_sorted[0].copy()
                stall_counter = 0
            else:
                stall_counter += 1
                
            if stall_counter > stall_threshold:
                # Optional: Restart or break. Breaking is safer to avoid wasting budget.
                # However, CMA-ES might still be useful. Let's just break to save time/calls.
                break

            # Update Mean
            # m_new = sum(weights[i] * X_sorted[i])
            m_new = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update Step Size (CSA)
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B^T D^-1 (m_new - m))
            diff = (m_new - m)
            # Transform diff to standard normal space
            # D^-1 B^T diff
            # Since Y = B D Z, Z = D^-1 B^T Y
            # We want Z_diff = D^-1 B^T diff
            # Note: B is orthogonal, so B^T = B^-1
            Z_diff = np.dot(B.T, diff) / D
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * Z_diff
            
            # Update Covariance Adaptation Control Variable
            # hsig = ||ps|| / sqrt(1 - (1-c_sigma)^{2(it+1)}) / chi_n < 1.4 + 2/(dim+1)
            # Approximation for large it
            hsig = np.linalg.norm(ps) / chi_n < 1.4 + 2.0 / (dim + 1)
            
            # Update Covariance Memory
            # pc = (1 - c_c) * pc + hsig * sqrt(c_c * (2 - c_c) * mueff) * (m_new - m) / sigma
            if hsig:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * diff / sigma
            else:
                pc = (1 - c_c) * pc
                
            # Update Covariance Matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc pc^T + (1 - hsig) * c_c * (2 - c_c) * C)
            #    + c_mu * sum(weights[i] * (y_i y_i^T))
            
            # Rank-1 update
            C_update = (1 - c_1 - c_mu) * C
            C_update += c_1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C)
            
            # Rank-mu update
            # Y_sorted = (X_sorted[:mu] - m) / sigma
            Y_sorted = (X_sorted[:mu] - m) / sigma
            # Weighted sum of outer products
            # C_update += c_mu * sum(weights[i] * outer(Y_i, Y_i))
            # Efficient calculation: (Y_sorted.T * weights) @ Y_sorted
            C_update += c_mu * np.dot(Y_sorted.T * weights, Y_sorted)
            
            C = C_update
            
            # Update Mean
            m = m_new
            
            # Update Eigenvalue decomposition
            # Regularize C if needed
            C = (C + C.T) / 2.0
            try:
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))
            except np.linalg.LinAlgError:
                # Fallback if Cholesky/Eigh fails
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)
                
            # Check convergence of sigma
            if sigma < 1e-10:
                break
                
        # Return best found
        if best_x_local is not None:
            self.best_x = best_x_local
            self.best_val = best_val_local
        elif self.best_x is None:
             # Fallback if no calls were made or something went wrong
             self.best_x = np.zeros(dim)
             self.best_val = func(self.best_x)
        return self.best_x, self.best_val
# EVOLVE-BLOCK-END
