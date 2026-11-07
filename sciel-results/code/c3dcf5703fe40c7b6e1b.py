"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size: lambda = 4 + 3 * log(n)
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2  # Number of recombination weights

        # Weights: logarithmic ranking
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)

        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))

        # Initial state
        # Start at center of the box [-5, 5]^n
        m = np.zeros(n)
        sigma = 2.0  # Initial step size
        C = np.eye(n)  # Covariance matrix
        B = np.eye(n)  # Eigenvectors
        D = np.ones(n)  # Square roots of eigenvalues
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Thresholds for Heaviside operators
        hs = (1.4 + 2.0 / (n + 1)) * np.sqrt((mu_eff - 1) + 1)
        hs_c = (1.4 + 2.0 / (n + 1)) * np.sqrt((mu_eff - 1) + 1)

        best_x = m.copy()
        best_f = func(best_x)
        calls = 1

        # Precompute B @ D for sampling? No, B and D change.
        # We will use Cholesky or Eigen decomposition for sampling.
        # Standard CMA-ES samples y = B @ (D * z).
        
        # To speed up eigen decomposition, we can use scipy if available, 
        # but numpy.linalg.eigh is sufficient for n=5.
        
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a full generation
            if budget - calls < lam:
                # If not enough for a full generation, we could do a partial one 
                # or just break. Breaking is safer to avoid bias.
                break

            # Generate lambda samples
            Z = np.random.standard_normal((lam, n))
            # y_i = B @ (D * z_i)
            # We need B and D from C.
            # For efficiency, we maintain B and D from previous iteration.
            # Initially C=I, so B=I, D=1.
            
            Y = np.zeros((lam, n))
            for i in range(lam):
                Y[i] = B @ (D * Z[i])
            
            X = m + sigma * Y
            # Clip to bounds [-5, 5]
            X = np.clip(X, -5, 5)

            # Evaluate fitness
            F = np.empty(lam)
            for i in range(lam):
                F[i] = func(X[i])
                calls += 1
                if time.time() > deadline:
                    break
            
            # If we broke out of eval loop early due to deadline, 
            # we might not have all F values. 
            # The loop condition `calls < budget` will handle it, 
            # but we need to ensure F is fully populated if we proceed to update.
            # If we hit deadline inside eval, we should return best so far.
            if time.time() > deadline:
                break

            # Update best
            if F.min() < best_f:
                best_f = F.min()
                best_x = X[F.argmin()].copy()

            # Sort
            indices = np.argsort(F)
            F = F[indices]
            X = X[indices]
            Y = Y[indices]

            # Recombination
            m_old = m.copy()
            m = np.sum(w[:, None] * X[:mu], axis=0)

            # Step size adaptation (CSA)
            y = (m - m_old) / sigma
            z = B.T @ (y / D)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
            
            # Heaviside function for sigma update
            hs_norm = np.linalg.norm(p_sigma)
            if hs_norm / hs < (1 - c_sigma):
                sigma *= np.exp(c_sigma / hs * (hs_norm / hs - 1))

            # Covariance matrix adaptation
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * z
            
            # Rank-1 update
            h_c = 1.0
            # Simplified h_c calculation:
            # h_c = 1 if ||p_c|| / sqrt(1 - (1-c_c)^(2*gen)) < (1.4 + 2/(n+1)) * sqrt((mu_eff - 1) + 1)
            # We don't track gen count easily here without extra state, 
            # but for short runs, h_c=1 is a reasonable approximation 
            # or we can estimate the expected norm.
            # Let's use the standard approximation:
            # E[||z||^2] = n
            # E[||p_c||^2] = c_c(2-c_c)mu_eff / (1-(1-c_c)^2) * n ... this is getting complex.
            # For n=5, the impact of h_c is small. We set h_c=1.
            
            C = (1 - c_1) * C + c_1 * (np.outer(y, y) + (1 - h_c) * C)
            
            # Rank-mu update
            for i in range(mu):
                y_i = Y[i]  # Already scaled by sigma? No, Y is m + sigma * Y, so Y is the raw step.
                # Wait, X = m + sigma * Y. So Y is the raw step vector.
                # The update uses (X[i] - m_old) / sigma = Y[i] + (m - m_old)/sigma - (m_old - m_old)/sigma?
                # Actually, y_i = (X[i] - m_old) / sigma.
                # X[i] = m + sigma * Y[i] (where Y[i] is the sampled vector before adding m).
                # So X[i] - m_old = (m - m_old) + sigma * Y[i]
                # y_i = (m - m_old)/sigma + Y[i] = y + Y[i]
                # Wait, the standard CMA-ES update uses the deviations from the *old* mean.
                # y_i = (X[i] - m_old) / sigma
                # We have Y[i] such that X[i] = m + sigma * Y[i].
                # So X[i] - m_old = (m - m_old) + sigma * Y[i]
                # y_i = (m - m_old)/sigma + Y[i] = y + Y[i]
                
                y_i = y + Y[i]
                C += c_mu * w[i] * np.outer(y_i, y_i)
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Eigen decomposition
            # For n=5, this is fast.
            D_sq, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D_sq, 1e-12, None))
            
            # Check for convergence or ill-conditioning
            if np.max(D) / np.min(D) > 1e14:
                break
            if sigma < 1e-15:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
