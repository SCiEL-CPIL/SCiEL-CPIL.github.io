import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for black-box continuous functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        dim = self.dim
        budget = self.budget
        calls = 0
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        # Initial mean: start at origin, common for centered problems
        # If the shift is hidden, origin is a reasonable guess for the center of the box
        m = np.zeros(dim)
        # Initial step size
        # Rastrigin has local minima spaced by 1 unit. 
        # A sigma of 1.0 allows exploring the local landscape.
        # However, if the rotation is significant, we might need a larger sigma to find the global basin.
        # Let's start with sigma=1.0.
        sigma = 1.0
        # Covariance matrix
        C = np.eye(dim)
        # Inverse covariance
        C_inv = np.eye(dim)
        
        # Adaptation parameters
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Step size control
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        # Distribution parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cmu1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu2 = min(1 - cmu1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Evolution path for covariance
        pc = np.zeros(dim)
        # Evolution path for step size
        ps = np.zeros(dim)
        
        # Precompute matrix square root
        # C = C_inv^{-1}, so C^{1/2} is needed for sampling
        # We will update C and recompute sqrt each iteration or use eigendecomposition
        # For simplicity and robustness in a small script, we'll do eigendecomposition if C becomes ill-conditioned
        # Or just use scipy.linalg.sqrtm if available, but let's stick to numpy
        # Actually, for dim=5, we can just compute the Cholesky or Eigen decomposition of C each step if needed, 
        # but standard CMA-ES updates C and maintains its inverse. 
        # To sample, we need $z \sim N(0, C)$. This is $C^{1/2} z'$.
        # We can maintain the eigen decomposition of C.
        
        eigvals, eigvecs = np.linalg.eigh(C)
        
        def sample():
            """Sample from N(0, C)"""
            z = np.random.randn(dim)
            # Scale by sqrt of eigenvalues
            z = z * np.sqrt(eigvals)
            # Rotate by eigenvectors
            return eigvecs @ z

        def update_cma_matrices():
            """Update C, C_inv, and eigen decomposition"""
            nonlocal C, C_inv, eigvals, eigvecs
            # Update C
            # C = (1-cmu1-cmu2)C + cmu1 * sum wi (yi-yi)yi^T + cmu2 * (ym ym^T)
            # This is computationally heavy to do exactly every step in pure python loops for large dim,
            # but dim=5 is small.
            
            # Recalculate C from scratch every step to avoid drift? 
            # No, standard CMA-ES updates incrementally.
            # Let's implement the standard update.
            
            # We need the selected points y_i
            # This function is called with the history of selected y's
            pass 

        # Standard CMA-ES loop
        # We will store the selected points for the C update
        
        best_x = m.copy()
        best_f = float('inf')
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # 1. Sample population
            # We need to sample lam points
            # To save time, we generate them all at once
            # z = N(0, I) (lam x dim)
            # y = m + sigma * C^{1/2} z
            # C^{1/2} = eigvecs @ diag(sqrt(eigvals)) @ eigvecs.T
            # So y = m + sigma * eigvecs @ (diag(sqrt(eigvals)) @ z.T).T
            
            Z = np.random.randn(lam, dim)
            # Project Z to the C space
            # Z_scaled = Z * sqrt(eigvals) (broadcasting)
            Z_scaled = Z * np.sqrt(eigvals)
            # Rotate
            # Y = Z_scaled @ eigvecs.T
            Y = Z_scaled @ eigvecs.T
            
            X = m[None, :] + sigma * Y
            
            # 2. Evaluate
            F = np.zeros(lam)
            for i in range(lam):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                # Clip to box? The problem says func clips, but we can clip here to avoid out-of-bounds issues if func doesn't handle it well (though it says it does).
                # Let's trust func to clip, but ensure our inputs are finite.
                F[i] = func(X[i])
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()
            
            if calls >= budget:
                break
            
            # 3. Sort and select
            # Sort indices by fitness
            order = np.argsort(F)
            # Selected indices
            idx = order[:mu]
            # Selected points
            X_sel = X[idx]
            # Selected fitnesses
            F_sel = F[idx]
            
            # 4. Update mean
            m_new = np.sum(w[:, None] * X_sel, axis=0)
            
            # 5. Update evolution paths
            # y_i = (x_i - m) / sigma
            Y_sel = (X_sel - m[None, :]) / sigma
            
            # Mean of selected Y
            Y_mean = np.sum(w[:, None] * Y_sel, axis=0)
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv @ Y_mean
            # Check if ps is too large (optional, but good for robustness)
            # If ||ps|| / sqrt(1 - (1-cs)^{2(it+1)}) > chiN + 1.4 + 2/dim, then...
            # We skip this check for brevity as it's less critical for short runs
            
            # Update pc
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls // lam + 1))) < (chiN + 1.4 + 2/dim) / np.sqrt(1 - (1 - cs)**(2 * (calls // lam + 1)))
            # chiN is the expected norm of a standard normal vector
            chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            if hsig:
                pc = (1 - cc) * pc + cc * Y_mean
            
            # 6. Update step size
            # log sigma update
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
            
            # 7. Update covariance matrix
            # C = (1 - cmu1 - cmu2) C + cmu1 * sum wi (yi - ym)(yi - ym)^T + cmu2 * ym ym^T
            # Note: The standard formula uses y_i = (x_i - m_old)/sigma
            # And ym = sum w_i y_i
            
            # Calculate the outer products
            # This is a 5x5 matrix, so it's cheap
            C_update = np.zeros((dim, dim))
            for i in range(mu):
                C_update += w[i] * np.outer(Y_sel[i], Y_sel[i])
            
            C_update += np.outer(Y_mean, Y_mean)
            
            C = (1 - cmu1 - cmu2) * C + cmu1 * C_update + cmu2 * np.outer(Y_mean, Y_mean)
            
            # Ensure C is positive definite
            # Sometimes numerical errors can make it indefinite
            # Re-eigendecompose
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues to 0 or a small positive number
                eigvals = np.clip(eigvals, 1e-10, None)
                # Recompute C_inv? We don't strictly need C_inv for sampling if we use eigenvectors.
                # But we need C_inv for the ps update: ps = ... + C_inv @ Y_mean
                # C_inv = eigvecs @ diag(1/eigvals) @ eigvecs.T
                C_inv = (eigvecs * (1.0 / eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                # If eigendecomposition fails, reset C to identity
                C = np.eye(dim)
                C_inv = np.eye(dim)
                eigvals, eigvecs = np.linalg.eigh(C)
                sigma = 1.0
                m = np.zeros(dim)
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                continue

            # Update mean
            m = m_new
            
            # Check for convergence
            if sigma < 1e-12 or np.all(np.abs(m) < 1e-12):
                break

        return best_x, best_f
