"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        n = self.dim
        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(n))
        # Number of selected parents
        mu = lam // 2
        
        # Weights for the selected parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance-effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Initial step size
        # Heuristic for sigma: typically 0.5 * range or based on initial spread
        # Since box is [-5, 5], initial spread is 10. 
        # A common choice is sigma0 = 0.5 * (max - min) / 2 or similar.
        # Let's start with a reasonably large sigma to explore the space quickly.
        sigma = 2.0
        
        # Initial mean: Center of the box
        m = np.zeros(n)
        
        # Initial covariance: Identity matrix
        C = np.eye(n)
        
        # Precompute inverse square root of C for sampling (initially identity)
        # We will update this via eigen-decomposition or Cholesky every generation
        # For small n (5), Cholesky is fast.
        
        # Evaluation budget tracking
        evals_used = 0
        max_evals = self.budget
        
        # Initial evaluation
        # We need to ensure we stay within bounds. 
        # CMA-ES sampling might go out of bounds. 
        # Strategy: Clip the sampled points. This distorts the distribution slightly 
        # but is standard for box-constrained problems when not using re-normalization.
        
        # To be safe and maximize early progress, let's do a quick local search 
        # or just start CMA-ES directly. 
        # Given the "Bent Cigar" nature, finding the valley is key.
        
        # Let's initialize with a random point to avoid symmetry issues if the optimum is shifted
        rng = np.random.default_rng(0)
        m = rng.uniform(-5, 5, n)
        
        # First evaluation
        m_clipped = np.clip(m, -5, 5)
        fvals = [func(m_clipped)]
        evals_used += 1
        
        # Main loop
        while evals_used < max_evals:
            if time.time() >= deadline:
                break
                
            # Sample offspring
            # We need to sample from N(m, sigma^2 * C)
            # Use Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset to identity or add jitter
                C = np.eye(n)
                L = np.eye(n)
            
            # Sample standard normals
            # We need lam samples
            # To save time, we can generate them in batches if needed, but lam is small
            N = rng.standard_normal((lam, n))
            # Transform: m + sigma * N @ L.T
            ys = m[None, :] + sigma * (N @ L.T)
            
            # Clip to bounds
            ys = np.clip(ys, -5, 5)
            
            # Evaluate offspring
            fvals = []
            for y in ys:
                if evals_used >= max_evals:
                    break
                fval = func(y)
                fvals.append(fval)
                evals_used += 1
            
            # If we ran out of budget during evaluation, break
            if evals_used >= max_evals:
                break
                
            # Sort offspring by fitness
            # We have fvals corresponding to ys
            # If we broke early, fvals might be shorter than lam. 
            # Let's assume we always complete the batch unless budget is exhausted exactly.
            # If budget is exhausted, we break the loop anyway.
            
            # Sort
            idx = np.argsort(fvals)
            # Take best mu
            selected_idx = idx[:mu]
            selected_ys = ys[selected_idx]
            
            # Update mean
            # m_new = sum(weights_i * y_i)
            m_new = np.sum(weights[:, None] * selected_ys, axis=0)
            
            # Update step size (1/5 success rule approximation or CS adaptation)
            # Simple 1/5 rule: 
            # Count successes in the last generation? 
            # Standard CMA-ES uses a cumulation path for step size control.
            # For simplicity and robustness in this context, let's use a simplified 
            # step size adaptation based on the improvement or a fixed rule.
            # Actually, standard CMA-ES step size control is crucial.
            # Let's implement the standard CS (Cumulative Step) adaptation.
            
            # We need to maintain path_sigma and path_c
            # This requires storing previous state. Let's refactor to include these.
            
            # Since I'm writing this inline, let's use a simpler, robust heuristic 
            # often used in lightweight CMA-ES implementations:
            # If the best fitness improved significantly, increase sigma slightly.
            # If not, decrease it.
            # However, proper CMA-ES is better.
            
            # Let's restart the logic to properly maintain CMA-ES state variables.
            pass # Placeholder, will rewrite the loop properly below

        # The above loop structure was a bit messy with state. 
        # Let's rewrite the __call__ method cleanly with full CMA-ES state.
        
        # RESTART LOGIC WITH PROPER STATE
        
        n = self.dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initial state
        rng = np.random.default_rng(0)
        m = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n) # Eigenvectors
        D = np.ones(n) # Eigenvalues (square roots)
        
        # Path vectors
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Precompute inverse square root of C (initially identity)
        # For sampling, we use B and D.
        # Sample: m + sigma * (B @ (D * z)) where z ~ N(0, I)
        
        evals_used = 0
        max_evals = self.budget
        
        # First eval
        m_clipped = np.clip(m, -5, 5)
        best_x = m_clipped.copy()
        best_fx = func(m_clipped)
        evals_used += 1
        
        while evals_used < max_evals:
            if time.time() >= deadline:
                break
                
            # Sample
            Z = rng.standard_normal((lam, n))
            # Transform Z to standard normal with covariance C
            # y = m + sigma * (B @ (D * z))
            # Note: B is n x n, D is n, z is lam x n
            # D * z -> lam x n
            # B @ (D * z).T -> n x lam
            # (B @ (D * z).T).T -> lam x n
            # Actually: y_i = m + sigma * (B @ (D * z_i))
            # Vectorized: Y = m[None, :] + sigma * ((Z * D[None, :]) @ B.T)
            Y = m[None, :] + sigma * ((Z * D[None, :]) @ B.T)
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            fvals = np.empty(lam)
            for i in range(lam):
                if evals_used >= max_evals:
                    break
                fvals[i] = func(Y[i])
                evals_used += 1
            
            if evals_used >= max_evals:
                break
            
            # Sort
            idx = np.argsort(fvals)
            # Check if best improved
            if fvals[idx[0]] < best_fx:
                best_fx = fvals[idx[0]]
                best_x = Y[idx[0]].copy()
                
            # Update Mean
            # We need the best mu points
            selected_ys = Y[idx[:mu]]
            m_new = np.sum(weights[:, None] * selected_ys, axis=0)
            
            # Update Step Size (CS)
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mueff) * B.T @ (sigma * (m_new - m)) / sigma
            # Note: (m_new - m) is the weighted sum of (y_i - m)
            # Let delta = m_new - m
            delta = m_new - m
            # We need to transform delta to the "principal axes" defined by B
            # The formula is: p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mueff) * (B.T @ delta) / sigma
            # Wait, the standard formula uses the standardized step.
            # The step taken is sigma * (B @ (D * z)).
            # The mean update is a weighted sum of steps.
            # The term in the path is the weighted sum of the *standardized* steps.
            # Actually, simpler: 
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mueff) * (B.T @ (m_new - m)) / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B.T @ delta) / sigma
            
            # Check norm
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals_used / lam))) > (1.4 + 2 / (n + 1)) * np.sqrt(n + 1 - 2):
                # This condition is for detecting step size increase
                # Simplified: if norm is large, increase sigma
                # Standard CMA-ES uses:
                # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_N - 1))
                # chi_N is the expected norm of standard normal in n dims
                chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update Covariance
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mueff) * delta
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * delta
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c.T + (1 - (1 - c_c)**2) * C) + c_mu * sum(weights_i * (y_i - m) * (y_i - m).T)
            
            # Calculate the sum of outer products for the selected parents
            # y_i - m = delta_i (where delta_i is the step for parent i)
            # We have selected_ys. m is the OLD mean.
            # So steps = selected_ys - m[None, :]
            steps = selected_ys - m[None, :]
            
            # c_mu * sum(weights_i * steps_i * steps_i.T)
            # Vectorized: (steps.T * weights) @ steps
            # steps is mu x n
            # steps.T is n x mu
            # weights is mu
            # (steps.T * weights[None, :]) @ steps -> n x n
            cov_update = (steps.T * weights[None, :]) @ steps
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * cov_update
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Update Eigenvalues and Vectors (for sampling in next iteration)
            # We need to update B and D such that C = B @ diag(D^2) @ B.T
            # We can use the eigendecomposition of C
            # However, doing full eigendecomposition every step is O(n^3). For n=5, it's fine.
            # Alternative: Use the update formula for B and D.
            # For simplicity and robustness, let's just recompute B and D from C.
            # This is safe.
            eigvals, eigvecs = np.linalg.eigh(C)
            # eigvals are D^2
            # Handle numerical issues
            eigvals = np.maximum(eigvals, 1e-16)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            # Update mean
            m = m_new
            
        return best_x, best_fx

# EVOLVE-BLOCK-END
