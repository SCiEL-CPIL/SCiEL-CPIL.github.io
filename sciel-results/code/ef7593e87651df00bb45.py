"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style Sphere function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        N = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # lambda: population size
        # For N=5, lambda=4 + floor(3*log(5)) = 4 + 6 = 10
        lam = 4 + int(3 * np.log(N))
        mu = lam // 2  # number of best solutions used for update
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)  # effective population size
        
        # Step size control parameters
        c_sigma = (mueff + 2) / (N + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + c_sigma
        
        # Covariance matrix control parameters
        c1 = 2 / ((N + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((N + 2)**2 + mueff))
        cs = (mueff + 2) / (N + mueff + 5)
        
        # Initial values
        m = np.zeros(N)  # Start at origin
        sigma = 2.0
        C = np.eye(N)
        pc = np.zeros(N)
        psigma = np.zeros(N)
        
        calls = 0
        best_x = m.copy()
        best_f = float('inf')
        
        # Initial evaluation
        fx = func(m)
        calls += 1
        if fx < best_f:
            best_f = fx
            best_x = m.copy()
            
        if calls >= budget:
            return best_x, best_f

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate samples
            # Cholesky decomposition of C
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes singular, reset or perturb
                C = np.eye(N) * 1e-10
                C_chol = np.linalg.cholesky(C)
            
            # Sample lambda points
            ys = np.random.standard_normal((lam, N))
            xs = np.zeros((lam, N))
            fs = np.zeros(lam)
            
            for i in range(lam):
                x = m + sigma * ys[i] @ C_chol
                x = np.clip(x, -5, 5)
                fx_i = func(x)
                calls += 1
                xs[i] = x
                fs[i] = fx_i
                
                if fx_i < best_f:
                    best_f = fx_i
                    best_x = x.copy()
                    
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
                
            # Sort by fitness
            indices = np.argsort(fs)
            xs_sorted = xs[indices]
            fs_sorted = fs[indices]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
            
            # Update step size
            # Normalized step
            step = (m - m_old) / (sigma * np.linalg.norm(C_chol))
            psigma = (1 - c_sigma) * psigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * step
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(psigma) / np.sqrt(N) - 1))
            
            # Update covariance matrix
            # Rank-1 update
            y = (m - m_old) / sigma
            C = (1 + c1 * cs * (cs - 1 + cmu) * np.dot(psigma, psigma) - c1) * C \
                + c1 * np.outer(y, y) \
                + cmu * np.sum(weights[:mu] * np.outer(xs_sorted[:mu] - m_old, xs_sorted[:mu] - m_old) / (sigma**2), axis=0)
            
            # Ensure symmetry and positive semi-definiteness
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-10):
                C = np.dot(eigvecs, np.diag(np.maximum(eigvals, 1e-10))) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
