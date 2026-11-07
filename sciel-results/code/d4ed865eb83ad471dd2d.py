"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lambda_pop = 6  # Population size
        self.mu = 3          # Number of best solutions to select

    def __call__(self, func):
        d = self.dim
        lam = self.lambda_pop
        mu = self.mu
        
        # CMA-ES parameters
        # Weights for the best mu solutions
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)  # Effective selection mass

        # Adaptation rates
        cm = 2.0 / ((lam + 2.0)**2 + mueff)
        cs = (mueff + 2.0) / (lam + mueff + 5.0)
        cc = (4.0 + mueff / lam) / (lam + 4.0 + 2.0 * mueff / lam)
        c1 = 2.0 / (lam**2 + 8.0)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0) / (lam**2 + 2.0))

        # Initial state
        m = np.zeros(d)  # Mean vector
        sigma = 2.0      # Initial step size
        C = np.eye(d)    # Covariance matrix
        pc = np.zeros(d) # Evolution path for covariance
        ps = np.zeros(d) # Evolution path for step size
        sqrtcs = np.sqrt(cs * (2 - cs) * (lam + 2) * mueff)
        inv_sqrtcs = 1.0 / sqrtcs

        # Precompute square root of covariance matrix
        # We will update this every generation
        # Initial Cholesky decomposition
        try:
            B = np.linalg.cholesky(C)
            D = np.sqrt(np.diag(B))
            B = B / D[:, np.newaxis]
            minvD = 1.0 / D
        except np.linalg.LinAlgError:
            B = np.eye(d)
            D = np.ones(d)
            minvD = np.ones(d)

        # Best solution tracking
        best_x = m.copy()
        best_f = func(m)
        calls_made = 1
        
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        while calls_made < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                # Standard normal sample
                z = np.random.randn(d)
                # Transform to normal with covariance C
                y = B @ (D * z)
                # Candidate
                x = m + sigma * y
                # Evaluate
                fx = func(x)
                calls_made += 1
                
                ys.append(y)
                xs.append(x)
                fs.append(fx)
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # Sort by fitness
            idx = np.argsort(fs)
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            ys = ys[idx]
            xs = xs[idx]
            fs = fs[idx]

            # Update mean
            yw = np.dot(w, ys[:mu])
            m = m + sigma * cm * yw

            # Update step size
            ps = (1 - cs) * ps + sqrtcs * inv_sqrtcs * (B @ (D * yw))
            # Chi-square approximation
            chi_lam = np.sqrt(d) * (1 + 1/(4*d) - 1/(21*d**2))
            sigma = sigma * np.exp(cs / chi_lam * (np.linalg.norm(ps) / chi_lam - 1))
            
            # Clamp sigma
            if sigma < 1e-12:
                sigma = 1e-12
            if sigma > 1e6:
                sigma = 1e6

            # Update covariance
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls_made - lam)))) < (1.4 + 2 / (d + 1)) * cc + 1
            if hsig:
                pc = (1 - cc) * pc + cc * (1 / (1 - (1 - cs)**(calls_made - lam))) * (B @ (D * yw))
            else:
                pc = (1 - cc) * pc

            # Rank-one update
            C = (1 + c1 * hsig * (cc * (2 - cc)) - c1) * C + c1 * np.outer(pc, pc)
            
            # Rank-mu update
            ymu = ys[:mu]
            ymu = B @ (D * ymu.T).T  # Transform selected ys
            C = C + cmu * np.sum(w * np.outer(ymu.T, ymu.T), axis=0)

            # Update Cholesky decomposition
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-12):
                C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T
            
            try:
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(B))
                B = B / D[:, np.newaxis]
                minvD = 1.0 / D
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                C = C + 1e-6 * np.eye(d)
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(B))
                B = B / D[:, np.newaxis]
                minvD = 1.0 / D

        return best_x, best_f
# EVOLVE-BLOCK-END
