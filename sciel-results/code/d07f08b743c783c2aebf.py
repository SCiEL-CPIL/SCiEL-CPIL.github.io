"""Model-and-optimize optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB f11."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # Safety margin before 330s limit
        
        n = self.dim
        # Population size
        lam = 4 + int(3 * np.log(n))
        
        # Initialize mean (center of search)
        # Use a fixed seed for reproducibility of the initial point, 
        # but the problem instance has hidden shifts/rotations, so 
        # starting near center is usually safe for BBOB.
        rng = np.random.default_rng(0)
        m = np.zeros(n)
        
        # Initial sigma (step size)
        # For BBOB, the optimum is in [-4, 4]^n. 
        # A reasonable initial step size is around 2.0.
        sigma = 2.0
        
        # Covariance matrix C
        C = np.eye(n)
        
        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # Update parameters
        mu = lam // 2  # Number of selected parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Chi-square expectation for step size control
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (25 * n**2))
        
        best_x = m
        best_f = func(m)
        
        # Track evaluations
        eval_count = 1
        
        while eval_count < self.budget:
            # Check time limit
            if time.time() > deadline:
                break
                
            # Sample population
            # Generate random vectors
            ys = rng.standard_normal((lam, n))
            
            # Transform to x-space
            # We need to apply the covariance matrix square root.
            # For small n, we can use Cholesky or eigen decomposition.
            # To be efficient, we update C and use its Cholesky factor.
            # However, standard CMA-ES often just uses the matrix directly 
            # via (C @ y) or similar if we keep it simple.
            # Actually, the standard way is:
            # x = m + sigma * (L @ z) where L is Cholesky of C.
            
            # To avoid expensive Cholesky every step (though n=5 is tiny),
            # we can just do:
            # v = C @ y
            # x = m + sigma * (v / norm(v) * norm(z))? No.
            
            # Correct CMA-ES sampling:
            # x_i = m + sigma * (C^{1/2} @ z_i)
            
            # Since n=5, Cholesky is cheap.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset it
                C = np.eye(n)
                L = np.eye(n)
                sigma = 2.0
                ps = np.zeros(n)
                pc = np.zeros(n)
                
            # Generate candidates
            zs = rng.standard_normal((lam, n))
            vs = np.dot(zs, L.T) # z @ L^T is equivalent to L @ z for each row if we transpose properly?
            # Actually, if C = L L^T, then C^{1/2} is L.
            # We want x = m + sigma * L @ z.
            # If zs is (lam, n), we want (L @ z_i) for each i.
            # This is (L @ zs.T).T
            steps = np.dot(L, zs.T).T
            xs = m + sigma * steps
            
            # Evaluate
            fs = np.array([func(x) for x in xs])
            eval_count += lam
            
            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = xs_sorted[0]
                
            # Update mean
            m_new = m + sigma * np.sum(weights[:, None] * (xs_sorted[:mu] - m), axis=0)
            
            # Update evolution paths
            # ps = (1 - c_sigma) * ps + c_sigma * sqrt(mueff) * (1/sigma) * (m_new - m)
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mueff) * (m_new - m) / sigma
            
            # Check h_sigma
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (self.budget - eval_count))) < (1.4 + 2 / (n + 1)) * chi_n
            hs = float(hs)
            
            # Update pc
            pc = (1 - c_c) * pc + hs * c_c * np.sqrt(mueff) * (m_new - m) / sigma
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - hs) * hs * c_c * (2 - c_c) * C) + c_mu * sum(weights * (xi - m)(xi - m)T)
            
            # Rank-one update
            rank_one = pc * pc.T + (1 - hs) * hs * c_c * (2 - c_c) * C
            
            # Rank-mu update
            diff = xs_sorted[:mu] - m
            rank_mu = np.sum(weights[:, None, None] * np.outer(diff, diff), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (norm(ps) / chi_n - 1))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Limit sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-12, 1e6)
            
            m = m_new
            
            # Check for convergence or stagnation
            if sigma < 1e-10:
                break
            if np.max(C) - np.min(C) < 1e-10 * np.max(C):
                # Variance is very low, maybe stuck
                # We can try to restart or just break
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
