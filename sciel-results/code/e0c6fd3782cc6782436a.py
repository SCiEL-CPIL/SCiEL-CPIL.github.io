import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        # CMA-ES parameters
        lambda_ = 9
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(lambda_ + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        csigma = (mueff + 2) / (n + mueff + 5)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initialization
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        Cinv = np.eye(n) # Not strictly needed for sampling if we use Cholesky, but for step size control?
        # Actually, standard CMA-ES uses Cholesky of C.
        # pc and hs are path accumulators
        pc = np.zeros(n)
        hs = np.sqrt(1 - np.exp(-2 / (n + 3)))
        
        best_x = x
        best_f = func(x)
        
        start_time = time.time()
        eval_count = 1
        
        while eval_count < self.budget:
            if time.time() - start_time > 290: # Safety margin
                break
            
            # Generate population
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset or regularize
                C = np.eye(n)
                L = np.eye(n)
            
            z = rng.standard_normal((lambda_, n))
            pop = np.array([x + sigma * (L @ z[i]) for i in range(lambda_)])
            
            # Evaluate
            fvals = np.array([func(pop[i]) for i in range(lambda_)])
            eval_count += lambda_
            
            # Sort
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0]
                
            # Mean update
            x_new = np.sum(weights[:, None] * pop[:mu], axis=0)
            
            # Path covariance adaptation
            # Need C^{-1/2} x_new - x
            # We can approximate or compute inverse sqrt
            # For small n=5, this is cheap
            Cinv = np.linalg.inv(C)
            Cinv_sqrt = np.linalg.cholesky(Cinv).T # This is actually C^{-1/2} if we are careful, or just use eig
            # Simpler: use eigendecomposition for C to get C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            Cinv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            
            pc = (1 - csigma) * pc + csigma * Cinv_sqrt @ (x_new - x)
            pc_norm = np.linalg.norm(pc) / np.sqrt(1 - (1 - csigma)**(2 * eval_count / lambda_))
            
            # Step size update
            if pc_norm < (1.4 + 1 / n) * hs:
                sigma *= np.exp(csigma * (pc_norm / hs - 1))
            
            # Covariance update
            delta_sig = (1 - (1 - csigma)**2) if pc_norm < (1.4 + 1 / n) * hs else 0
            C = (1 + c1 + cmu - c1 - delta_sig) * C \
                + c1 * (np.outer(pc, pc) - C) \
                + cmu * np.sum(weights**2 * np.outer(pop[:mu] - x, pop[:mu] - x), axis=0)
            
            x = x_new
            C = (C + C.T) / 2 # Symmetrize
            
            # Clip x to box? No, CMA-ES works in unconstrained space usually, 
            # but the function clips. However, if x goes far outside, it's inefficient.
            # The problem says func(x) clips. So we can let x go outside, 
            # but it's better to keep the center somewhat bounded.
            # Given the minimizer is in [-4, 4], and we start in [-5, 5], 
            # and sigma starts at 2, it's unlikely to go too far.
            
            # Optional: Restart if sigma is too small or large? 
            # With budget 500, we just run until budget exhausted.
            
        return best_x, best_f
