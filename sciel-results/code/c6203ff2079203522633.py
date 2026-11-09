"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    with multi-start initialization for the Bent Cigar function (BBOB f12).
    
    CMA-ES is a derivative-free optimizer that adapts the search distribution
    to the local landscape geometry. For the highly anisotropic Bent Cigar,
    it can align its covariance with the narrow valley direction, making
    efficient progress along all axes simultaneously.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if self.calls >= budget:
                raise ValueError("Budget exceeded")
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            self.calls += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        try:
            # Phase 1: Multi-start random search to find promising basins
            n_starts = 8
            starts = []
            for i in range(n_starts):
                if i == 0:
                    x0 = np.zeros(dim)
                elif i == 1:
                    x0 = np.ones(dim) * 2.0
                else:
                    x0 = np.random.uniform(-4.0, 4.0, size=dim)
                starts.append(x0)
            
            f_starts = []
            for x0 in starts:
                try:
                    fx = wrapped_func(x0)
                    f_starts.append(fx)
                except (TimeoutError, ValueError):
                    break
            
            if not f_starts:
                return np.zeros(dim), float('inf')

            # Pick the best starting point for CMA-ES
            best_idx = int(np.argmin(f_starts))
            x0 = starts[best_idx].copy()
            f0 = f_starts[best_idx]

            # Phase 2: CMA-ES optimization
            # Standard CMA-ES parameters
            lambda_pop = 4 + int(3 * np.log(dim))  # Population size
            sigma = 1.0  # Initial step size
            c1 = (mu + 2 / (dim + 3.7 - mu)) / ((dim + 1.3)**2 + mu)
            cmu = min(1 - c1, 2 * (mu - 1 + 1 / mu) / ((dim + 2)**2 + mu))
            mu = int(lambda_pop / 2)  # Number of recombination weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1. / np.sum(weights**2)
            cs = (mueff + 2) / (dim + mueff + 5)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            cc = (4 - mueff / mueff + 3) / (dim + 4 - mueff / mueff)
            chiN = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim**2))

            # Initialize CMA-ES state
            mean = x0.copy()
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            sigma = 1.0
            best_cma_x = x0.copy()
            best_cma_f = f0

            for gen in range(1000):
                if time.time() > deadline or self.calls >= budget:
                    break

                # Generate population
                offsets = []
                for _ in range(lambda_pop):
                    z = np.random.randn(dim)
                    offsets.append(C_sqrt @ z)

                # Evaluate population
                x_candidates = []
                f_values = []
                for offset in offsets:
                    x_candidate = mean + sigma * offset
                    try:
                        fx = wrapped_func(x_candidate)
                        x_candidates.append(x_candidate)
                        f_values.append(fx)
                    except (TimeoutError, ValueError):
                        break

                if len(f_values) < lambda_pop:
                    break

                # Sort by fitness
                order = np.argsort(f_values)
                x_sorted = [x_candidates[i] for i in order]
                f_sorted = [f_values[i] for i in order]

                # Update best
                if f_sorted[0] < best_cma_f:
                    best_cma_f = f_sorted[0]
                    best_cma_x = x_sorted[0].copy()

                # Update mean
                xw = np.sum([weights[i] * x_sorted[i] for i in range(mu)], axis=0)
                
                # Update path accumulators
                old_sigma = sigma
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_sqrt @ (xw - mean))
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * (xw - mean) / sigma
                
                # Update covariance matrix
                delta_d = (1 + 2 * max(0, (np.linalg.norm(ps)**2 / chiN**2 - 1)) * cc) - 1
                outer = np.outer(pc, pc)
                C = (1 + c1 * delta_d - c1 - cmu) * C \
                    + c1 * (outer - C) \
                    + cmu / sigma**2 * np.sum([
                        weights[i] * np.outer(x_sorted[i] - mean, x_sorted[i] - mean)
                        for i in range(mu)
                    ], axis=0)
                
                # Reconstruct C_sqrt (Cholesky factor)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if Cholesky fails
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                
                # Update mean
                mean = xw

            return best_cma_x, best_cma_f

        except (TimeoutError, ValueError):
            return self.best_x if self.best_x is not None else np.zeros(dim), self.best_f if self.best_f != float('inf') else 0.0

# EVOLVE-BLOCK-END
