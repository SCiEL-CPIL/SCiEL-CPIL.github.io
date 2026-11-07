import numpy as np

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.default_rng(42) # Fixed seed for reproducibility in eval, though task says hidden seed for func, optimizer can be deterministic or random. Task says "hidden seeded shift... for func". Optimizer seed doesn't matter as long as it's valid. Using fixed seed for optimizer is safe.

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(n))
        # Number of weighted points
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        # Variance equivalent population size
        mueff = 1.0 / np.sum(w**2)
        
        # Initial mean: center of the box
        x = np.zeros(n)
        # Initial step size: roughly 1/4 of the box width
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(n)
        
        # Paths
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Adaptation parameters
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        cmu = 2 / ((n + 1.3)**2 + mueff)
        c1 = (2 - cmu) / ((n + 1)**1.5 + mueff)
        csig = (mueff + 1.5) / (n + mueff + 5)
        
        # Eigenvalue decay for C update
        # We will update C using the formula:
        # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - delta) * C) + cmu * sum(wi * (yi * yi.T))
        # where yi = (xi - x_old) / sigma
        
        best_x = x
        best_f = func(best_x)
        calls = 1
        
        # Main loop
        while calls < budget:
            # Generate population
            # Cholesky decomposition of C
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, add jitter
                C += 1e-6 * np.eye(n)
                B = np.linalg.cholesky(C)
                
            z = self.rng.standard_normal((lam, n))
            pop = x + sigma * np.dot(z, B.T)
            
            # Clip to box
            pop = np.clip(pop, -5, 5)
            
            # Evaluate population
            fvals = np.array([func(p) for p in pop])
            calls += lam
            
            if calls > budget:
                break
                
            # Sort by fitness
            order = np.argsort(fvals)
            y = pop[order]
            f_sorted = fvals[order]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = y[0].copy()
                
            # Compute new mean
            x_old = x.copy()
            x = np.dot(w, y[:mu, :])
            
            # Compute step vectors
            y_vec = (y[:mu, :] - x_old) / sigma
            
            # Update path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(B.T, (x - x_old) / sigma)
            
            # Update step size sigma
            # Heuristic: if ||ps|| is large, increase sigma; if small, decrease
            # Standard formula:
            # sigma = sigma * exp(csig/dsigma * (||ps|| / E||N(0,I)|| - 1))
            # E||N(0,I)|| = sqrt(n) * (1 - 1/(4n) - 1/(25n^2)) approx
            chi_n = np.sqrt(n) * (1 - 1/(4*n) - 1/(25*n**2))
            sigma = sigma * np.exp(csig / ds * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update path pc
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * (x - x_old) / sigma
            
            # Update covariance matrix C
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - delta) * C) + cmu * sum(wi * yi * yi.T)
            # delta is a correction term, often approximated as 1 + 2 * max(0, sqrt((mueff-1)/(n+1)) - 1)
            delta = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1)
            
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((n, n))
            for i in range(mu):
                rankmu += w[i] * np.outer(y_vec[i], y_vec[i])
                
            C = (1 - c1 - cmu) * C + c1 * (rank1 + (1 - delta) * C) + cmu * rankmu
            
            # Symmetrize C to avoid numerical drift
            C = (C + C.T) / 2
            
            # Ensure positive definiteness by checking eigenvalues
            # If eigenvalues are too small, add jitter
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += 1e-6 * np.eye(n)

        return best_x, best_f
