"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based black-box optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES starting from x0 with initial sigma sigma0."""
        n = self.dim
        m = 2 + int(3 * np.log(n))  # population size
        m = min(m, max(6, budget // 10))
        m = max(m, 4)
        
        # CMA-ES parameters
        cmu = 1 / (n + 4)
        cs = (n + 1.3) / (n + 3)
        c1 = (2 - cmu) / ((n + 1.3) ** 2 + 2)
        cmu = 2 / ((n + 1.3) ** 2 + 5)
        damps = 1 + 2 * max(0, np.sqrt((m / 2 - 1) / (n + 1)) - 1) + cs
        sigma = sigma0
        mean = x0.copy()
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        inv_sqr = 1.0
        chi_n = n * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        best_x = x0.copy()
        best_f = func(x0)
        count = 1
        
        while count < budget and time.time() < deadline:
            # Generate population
            pops = np.zeros((m, n))
            for i in range(m):
                B = np.linalg.cholesky(C)
                z = rng.standard_normal(n)
                pops[i] = mean + sigma * (B @ z)
                pops[i] = np.clip(pops[i], -5, 5)
            
            # Evaluate
            fits = np.array([func(pops[i]) for i in range(m)])
            count += m
            
            # Sort by fitness
            order = np.argsort(fits)
            mu = max(1, m // 2)  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)
            
            # Update mean
            new_mean = np.zeros(n)
            for i in range(mu):
                new_mean += weights[i] * pops[order[i]]
            
            # Update step size
            old_mean = mean
            diff = new_mean - old_mean
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (
                inv_sqr * (diff / sigma) @ np.linalg.cholesky(C).T
            )
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, 1e8)
            
            # Update covariance
            y = diff / sigma
            pc = (1 - cmu) * pc + np.sqrt(cmu * (2 - cmu) * mueff) * y
            C = (1 - c1 - cmu) * C
            C += c1 * (pc @ pc + (np.dot(pc, pc) > (4 + 2 / cs) * (1 - cmu)) * np.eye(n))
            for i in range(mu):
                zi = (pops[order[i]] - old_mean) / sigma
                C += cmu * weights[i] * (zi @ zi.T)
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n) * np.trace(C) / n
            
            mean = new_mean
            mean = np.clip(mean, -5, 5)
            
            # Track best
            best_idx = order[0]
            if fits[best_idx] < best_f:
                best_f = fits[best_idx]
                best_x = pops[best_idx].copy()
        
        return best_x, best_f, count

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # Phase 1: Global exploration with diverse starts
        # Use a Hammersley-like sequence plus origin
        n_starts = min(8, budget // 50)
        starts = []
        starts.append(np.zeros(n))
        # Hammersley sequence
        for i in range(1, n_starts):
            x = np.zeros(n)
            for d in range(n):
                # Halving method for quasi-random
                x[d] = -5 + 10 * (1 - (2 * i + 1) * (0.5 + 0.5 * np.cos(2 * np.pi * (i + 1) / (n_starts + 1))))
                # Actually use a simpler approach: golden ratio
            # Use golden ratio based
            phi = 0.6180339887498949
            for d in range(n):
                frac = (i * phi ** (d + 1)) % 1.0
                x[d] = -5 + 10 * frac
            starts.append(x)
        
        # Evaluate all starts
        start_fits = []
        for s in starts:
            s = np.clip(s, -5, 5)
            f = func(s)
            start_fits.append(f)
        
        # Sort starts by fitness
        start_order = np.argsort(start_fits)
        
        # Phase 2: Run CMA-ES from the best few starts
        # Allocate remaining budget
        remaining = budget - len(starts)
        if remaining < 10:
            # Not enough budget, just return best start
            best_idx = start_order[0]
            return starts[best_idx], start_fits[best_idx]
        
        # Run CMA-ES from the best start with most of the budget
        best_start = starts[start_order[0]]
        best_f = start_fits[start_order[0]]
        
        # Give 70% of remaining budget to best, 30% to second best if available
        alloc1 = int(remaining * 0.7)
        alloc2 = remaining - alloc1 if len(start_order) > 1 else 0
        
        x1, f1, used1 = self._cma_es(func, best_start, 1.0, alloc1, rng, deadline)
        if f1 < best_f:
            best_f = f1
            best_x = x1
        else:
            best_x = best_start.copy()
        
        if alloc2 > 10 and len(start_order) > 1:
            second_start = starts[start_order[1]]
            x2, f2, used2 = self._cma_es(func, second_start, 1.0, alloc2, rng, deadline)
            if f2 < best_f:
                best_f = f2
                best_x = x2
        
        return best_x, best_f
# EVOLVE-BLOCK-END
