"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        self._call_count = 0
        dim = self.dim
        budget = self.budget

        # Counting wrapper
        def wrapped(x):
            if self._call_count >= budget:
                return 1e300
            self._call_count += 1
            try:
                return float(func(x))
            except Exception:
                return 1e300

        # CMA-ES Parameters
        lam = 12  # Population size
        mu = lam // 2  # Number of selected points
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Initial parameters
        m = np.zeros(dim)
        sigma = 2.0 # Initial step size
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Path variables
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Time constants
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        cmu = 2 / ((dim + 1.3)**2 + mueff)
        c1 = (2 - cmu + mueff / (dim + 2)) / ((dim + 2)**2 + mueff)
        cc = (4 - mueff / 2 + 2) / (dim + 4 - mueff / 2)
        
        # Precompute
        chiN = np.sqrt(dim) * (1 - 1. / (4. * dim) + 1. / (21. * dim**2))
        
        best_x = m.copy()
        best_f = float('inf')
        
        rng = np.random.default_rng(42)
        
        while self._call_count < budget:
            # Check if we have enough budget for a full population
            # If not, we can still evaluate the remaining calls, but adaptation is tricky.
            # For simplicity, if remaining < lam, we do a local random search around best_x
            # or just stop if we can't complete a generation. 
            # Given 500 budget and lam=12, we have 41 full generations + 8 calls.
            # We will handle the "tail" by doing a simple local search or just evaluating random points near best.
            
            if budget - self._call_count < lam:
                # Use remaining calls for a simple local search / polish
                for _ in range(budget - self._call_count):
                    if self._call_count >= budget: break
                    # Perturb best_x slightly
                    x_try = best_x + rng.normal(0, sigma * 0.5, dim)
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = wrapped(x_try)
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                break
            
            # Sample population
            # Generate lam samples from N(m, sigma^2 C)
            # We need to sample from the multivariate normal.
            # x = m + sigma * C^{1/2} z
            # We can use Cholesky decomposition of C.
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset or fix it
                C = np.eye(dim)
                L = np.eye(dim)
            
            population = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                population[i] = m + sigma * (L @ z)
                # Clip to bounds
                population[i] = np.clip(population[i], -5.0, 5.0)
            
            # Evaluate population
            fitness = np.array([wrapped(x) for x in population])
            
            # Update best
            idx_min = np.argmin(fitness)
            if fitness[idx_min] < best_f:
                best_f = fitness[idx_min]
                best_x = population[idx_min].copy()
            
            # Sort by fitness
            order = np.argsort(fitness)
            population = population[order]
            fitness = fitness[order]
            
            # Recombination
            # Weighted sum of the best mu points
            y = np.zeros(dim)
            for i in range(mu):
                y += weights[i] * (population[i] - m)
            
            # Update mean
            m_new = m + y
            
            # Update covariance matrix paths
            # ps = (1-cs)ps + sqrt(cs(2-cs)dmu) * C^{-1/2} * (m_new - m)/sigma
            # Note: C^{-1/2} is L^{-1}
            try:
                Linv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                Linv = np.eye(dim)
                
            y_sigma = (m_new - m) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Linv @ y_sigma)
            
            # Update step size sigma
            # log(sigma_new/sigma) = (1/ds) * (||ps||/chiN - 1)
            sigma_new = sigma * np.exp((1.0 / ds) * (np.linalg.norm(ps) / chiN - 1.0))
            
            # Update pc
            # pc = (1-cc)pc + sqrt(cc(2-cc)dmu) * (||ps||/chiN > 1.4+2/(dim+1)) * (m_new-m)/sigma
            cond = (np.linalg.norm(ps) / chiN) > (1.4 + 2.0 / (dim + 1.0))
            if cond:
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * y_sigma
            else:
                pc = (1 - cc) * pc
            
            # Update covariance matrix C
            # C = (1 + c1 + cmu - c1 - cmu/2) C + c1 (pc pc^T - C) + cmu sum wi (yi yi^T)
            # Actually, the standard update is:
            # C = (1 - c1 - cmu) C + c1 (pc pc^T + (1 - (1-cc)^2) C) + cmu sum wi (yi yi^T)
            # Let's use the simplified form often found in implementations:
            # C = (1 + c1 + cmu - c1 - cmu/2) ... no, let's stick to the paper's formula.
            # C <- (1 - c1 - cmu) C + c1 (pc pc^T + (1 - (1-cc)^2) C) + cmu * sum(wi * yi * yi.T)
            
            C_old = C
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + (1 - (1 - cc)**2) * C_old)
            
            # Add the weighted outer products of the successful steps
            # yi = (xi - m_old) / sigma
            for i in range(mu):
                yi = (population[i] - m) / sigma
                C += cmu * weights[i] * np.outer(yi, yi)
            
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Update state
            m = m_new
            sigma = sigma_new
            
            # Prevent sigma from becoming too small or large
            if sigma < 1e-12:
                sigma = 1e-12
            if sigma > 1e12:
                sigma = 1e12

        return None
# EVOLVE-BLOCK-END
