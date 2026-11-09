"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the BBOB f16 (Weierstrass) function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        # Number of offspring selected
        mu = lam // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Variance reduction factor
        mueff = int(np.sum(weights[:int(np.ceil(mu / 2))]) ** 2 / np.sum(weights ** 2))
        # Step size adaptation
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initial parameters
        x_mean = np.zeros(dim)  # Start at center, likely near global min after shift
        sigma = 2.0             # Initial step size
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cs = (1 + c_sigma + c_c) / (1 - c_sigma + c_c)
        hsig = (1.4 + 2 / (dim + 1)) * (1 - np.prod(C) ** (1 / (2 * dim)))  # Approximation, usually sqrt(prod(C))
        
        # Actually, hsig calculation in standard CMA-ES is:
        # hsig = (1.4 + 2/(dim+1)) * (1 - np.prod(C)**(1/(2*dim))) is not quite right.
        # Standard: hsig = (1.4 + 2/(dim+1)) * (1 - np.prod(C)**(1/(2*dim))) is for the norm of C^0.5?
        # Let's use the standard formula: hsig = (1.4 + 2/(dim+1)) * (1 - np.prod(C)**(1/(2*dim)))
        # Wait, the product of eigenvalues is det(C). The geometric mean of eigenvalues is (det(C))^(1/dim).
        # The term is 1 - (det(C))^(1/(2*dim))? No.
        # Standard CMA-ES: hsig = (1.4 + 2/(dim+1)) * (1 - np.prod(C)**(1/(2*dim))) is incorrect.
        # It is: hsig = (1.4 + 2/(dim+1)) * (1 - np.prod(C)**(1/(2*dim)))
        # Actually, let's just use a fixed hsig or calculate it properly.
        # hsig = (1.4 + 2/(dim+1)) * (1 - np.prod(C)**(1/(2*dim)))
        # For C=I, prod(C)=1, so hsig = 1.4 + 2/(dim+1).
        
        # Let's use a simpler, robust CMA-ES implementation structure.
        
        calls_made = 0
        best_f = np.inf
        best_x = x_mean.copy()
        
        # We need to evaluate the initial mean? No, usually we generate offspring.
        # But to ensure we have a valid starting point and record progress early, 
        # we might want to evaluate x_mean first.
        # The task says "Every call is recorded".
        
        # Let's evaluate x_mean first to get a baseline.
        f_mean = func(x_mean)
        calls_made += 1
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()
            
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes singular or non-positive definite, reset
                C = np.eye(dim)
                L = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                sigma = max(sigma * 0.5, 1e-4)
                
            # Standard CMA-ES update step
            # 1. Generate lambda samples
            z = np.random.randn(lam, dim)
            y = z @ L.T  # y = L.T * z
            x = np.tile(x_mean, (lam, 1)) + sigma * y
            x = np.clip(x, -5, 5)
            
            # 2. Evaluate
            f = np.array([func(xi) for xi in x])
            calls_made += lam
            
            # 3. Select best mu
            order = np.argsort(f)
            x_best = x[order[:mu]]
            f_best = f[order[:mu]]
            
            # Update best
            if f_best[0] < best_f:
                best_f = f_best[0]
                best_x = x_best[0].copy()
                
            # 4. Update mean
            x_old = x_mean
            x_mean = np.sum(weights[:, np.newaxis] * x_best, axis=0)
            
            # 5. Update step size (sigma)
            y_old = (x_mean - x_old) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / np.linalg.norm(L) * y_old
            norm_ps = np.linalg.norm(ps)
            
            # hsig check
            # hsig = (1.4 + 2/(dim+1)) * (1 - np.prod(C)**(1/(2*dim)))
            # For efficiency, we can approximate or use a fixed value if dim is small.
            # Let's use the standard formula.
            prod_c = np.prod(np.linalg.eigvalsh(C))
            if prod_c < 0: prod_c = 0
            hsig = (1.4 + 2 / (dim + 1)) * (1 - prod_c ** (1 / (2 * dim)))
            
            if norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - lam) / (dim * c_sigma))) < (1.4 + 2 / (dim + 1)) * hsig:
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / hsig - 1))
                
            # 6. Update covariance matrix
            if norm_ps / hsig < (1.4 + 2 / (dim + 1)) * np.sqrt((1 + c_c / c_c) * 2):
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * y_old
                # C update
                C = (1 - c_c * (2 - c_c) - c_1 - c_mu) * C \
                    + c_c * (2 - c_c) * np.outer(pc, pc) \
                    + c_1 * (C - np.eye(dim)) \
                    + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(y_best, y_best) for y_best in (x_best - x_old) / sigma)
                    
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Check for termination or stagnation
            if sigma < 1e-10:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
