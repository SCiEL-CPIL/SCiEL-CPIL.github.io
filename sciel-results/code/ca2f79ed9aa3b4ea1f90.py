"""Two-phase optimizer: deterministic Halton sweep then CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-scale Halton sweep followed by CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 285.0  # safety margin before 300s limit

        # Call counter and safe wrapper
        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            if time.time() > deadline:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # ---- Phase 1: Deterministic Halton sequence sweep ----
        # Generate a space-filling set of initial evaluations using Halton sequence
        # This gives strong early AOCC progress without stochastic variance.
        
        def halton(index, base):
            """Generate the Halton sequence value for given index and base."""
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result

        # First prime bases for each dimension
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        # Number of Halton points to evaluate
        # Use about 20% of budget for initial sweep
        n_halton = max(10, min(50, int(0.2 * budget)))
        
        best_x = None
        best_f = np.inf

        for i in range(1, n_halton + 1):
            if calls[0] >= budget or time.time() > deadline:
                break
            # Generate Halton point in [0,1]^dim
            point = np.array([halton(i, primes[d]) for d in range(dim)])
            # Map to [-5, 5]^dim
            x = -5.0 + 10.0 * point
            x = np.clip(x, -5, 5)
            try:
                fx = safe_func(x)
            except StopIteration:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Fallback if no successful calls
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = safe_func(best_x)
            except StopIteration:
                best_f = 0.0
            return best_x, best_f

        # ---- Phase 2: CMA-ES from the best Halton point ----
        # Implement a simplified CMA-ES with covariance adaptation
        
        remaining = budget - calls[0]
        if remaining < 10:
            return best_x, best_f

        # CMA-ES parameters
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Initial step size
        
        # Population size: lambda = 4 + floor(3 * log(dim))
        lam = 4 + int(3 * np.log(dim)) if dim > 1 else 4
        mu = lam // 2  # Number of best individuals used for update
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - c_1_default(mueff, dim) + mueff / dim) / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, (mueff - 2 + 2 / mueff) / ((dim + 2) ** 2 + mueff))
        
        def c_1_default(mueff, dim):
            return 2 / ((dim + 1.3) ** 2 + mueff)

        # Recalculate c_1 properly
        c_1 = (2 - 2 / ((dim + 1.3) ** 2 + mueff) + mueff / dim) / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, (mueff - 2 + 2 / mueff) / ((dim + 2) ** 2 + mueff))

        # State variables
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Track best
        best_x_cma = x.copy()
        best_f_cma = fx

        # Main CMA-ES loop
        while calls[0] < budget and time.time() < deadline:
            # Check if we have enough budget for a generation
            if calls[0] + lam > budget:
                break
            
            # Generate population
            ys = []
            fvals = []
            for i in range(lam):
                if calls[0] >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim) if 'rng' in dir() else np.random.standard_normal(dim)
                y = x + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                try:
                    fy = safe_func(y)
                except StopIteration:
                    break
                ys.append(y)
                fvals.append(fy)
            
            if len(ys) < mu:
                break
            
            # Sort by fitness
            order = np.argsort(fvals)
            ys_sorted = [ys[i] for i in order[:mu]]
            fvals_sorted = [fvals[i] for i in order[:mu]]
            
            # Update best
            for y, fy in zip(ys_sorted, fvals_sorted):
                if fy < best_f_cma:
                    best_f_cma = fy
                    best_x_cma = y.copy()
            
            # Compute new mean
            new_x = np.sum([weights[i] * ys_sorted[i] for i in range(mu)], axis=0)
            
            # Update covariance
            diff = [(y - x) / sigma for y in ys_sorted]
            C_update = c_1 * (C + np.outer(diff[0], diff[0]) - C)
            for i in range(1, mu):
                C_update += c_mu * weights[i] * (np.outer(diff[i], diff[i]) - C)
            C = C_update
            C = (C + C.T) / 2  # Symmetrize
            
            # Update step size
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (new_x - x) / sigma)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)
            
            # Update correlation matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (new_x - x) / sigma
            C = (1 - c_1 - c_mu + c_1 * (1 + (2 - c_c) / c_c)) * C \
                + c_1 * (np.outer(p_c, p_c) + (1 - c_c) * (2 - c_c) * C)
            C = (C + C.T) / 2
            
            # Recompute inverse square root periodically
            if calls[0] % 10 == 0:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-10, None)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except:
                    C = np.eye(dim)
                    C_inv_sqrt = np.eye(dim)
            
            x = new_x

        # Use the best found by CMA-ES
        if best_f_cma < best_f:
            best_f = best_f_cma
            best_x = best_x_cma.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
