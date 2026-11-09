"""Rewritten optimizer: CMA-ES with multi-start and time guard for Rastrigin f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for Rastrigin (f3) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES from x0 with initial sigma0, using up to budget evaluations."""
        d = self.dim
        n = budget
        if n < 10:
            # Too few evals, just do a few random steps
            x = x0.copy()
            fx = func(x)
            for _ in range(n - 1):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma0 * rng.standard_normal(d), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
            return x, fx

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(d)))  # population size
        mu = lam // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = (2 - cc + mueff / (d + 3)) / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        x = x0.copy()
        sigma = sigma0
        pc = np.zeros(d)
        ps = np.zeros(d)
        B = np.eye(d)
        D = np.ones(d)
        C = np.eye(d)
        evals = 0

        def sample(n_samples):
            nonlocal evals
            # Generate n_samples candidates
            z = rng.standard_normal((n_samples, d))
            # Transform using C
            # For small d, we can do this directly
            # y = x + sigma * z @ B * D
            candidates = x[None, :] + sigma * (z @ B) * D
            candidates = np.clip(candidates, -5, 5)
            evals += n_samples
            fvals = np.array([func(c) for c in candidates])
            return candidates, fvals

        while evals < n:
            if time.time() > deadline:
                break
            # Sample population
            remaining = n - evals
            lam_iter = min(lam, remaining)
            if lam_iter < 2:
                break
            pops, fvals = sample(lam_iter)
            # Sort
            order = np.argsort(fvals)
            pops = pops[order]
            fvals = fvals[order]
            # Update best
            x_best = pops[0]
            f_best = fvals[0]
            # Weighted mean of top mu
            w = weights[:min(mu, lam_iter)]
            w_sum = w.sum()
            w = w / w_sum
            mueff_actual = 1.0 / np.sum(w ** 2)
            m = int(min(mu, lam_iter))
            yw = np.sum(w[:, None] * pops[:m], axis=0)
            # Update step size control
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff_actual) / (sigma * np.sqrt(np.trace(C))) * B.T @ (yw - x)
            # Update evolution path for covariance
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff_actual) / (sigma) * (yw - x)
            # Update mean
            x = yw
            # Update sigma
            sigma *= np.exp(cs / (d + 2) * (np.linalg.norm(ps) / chi_d - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 1e6)
            # Update covariance
            delta = (1 + 2 * (cs / (1 - cs) - 1) * (np.linalg.norm(ps) ** 2 > (2 + 2 / cs) * chi_d ** 2))
            rank1 = np.outer(pc, pc)
            rankmu = np.sum(w[:, None, None] * np.outer(pops[:m] - x, pops[:m] - x), axis=0)
            C = (1 + c1 * delta - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
            # Symmetrize
            C = (C + C.T) / 2
            # Ensure positive definite: eigenvalue correction
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-10:
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Update B, D
            eigvals, eigvecs = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(eigvals, 1e-15))
            B = eigvecs
            # Clip x to box
            x = np.clip(x, -5, 5)

        return x, f_best

    def __call__(self, func):
        deadline = time.time() + 280  # Safety margin
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Multi-start strategy: try several starting points
        # For Rastrigin, the global minimum is at the origin (after hidden shift/rotation)
        # But we don't know the shift, so we try multiple starts

        # Strategy: 
        # 1. Start at origin (common for Rastrigin)
        # 2. Start at a few random points
        # 3. Use CMA-ES from each, with a budget split

        # We'll do a coarse search first to find promising regions, then refine

        # Phase 1: Coarse sampling to find good starting points
        n_coarse = min(50, budget // 10)
        coarse_x = []
        coarse_f = []
        for i in range(n_coarse):
            if time.time() > deadline:
                break
            if i == 0:
                x0 = np.zeros(d)
            elif i == 1:
                x0 = np.ones(d) * 2.0
            elif i == 2:
                x0 = np.ones(d) * (-2.0)
            else:
                x0 = rng.uniform(-5, 5, d)
            fx = func(x0)
            coarse_x.append(x0)
            coarse_f.append(fx)

        # Sort by fitness, take top few
        coarse_f = np.array(coarse_f)
        coarse_x = np.array(coarse_x)
        order = np.argsort(coarse_f)
        top_k = min(3, len(order))
        top_starts = coarse_x[order[:top_k]]
        top_fvals = coarse_f[order[:top_k]]

        # Phase 2: CMA-ES from each top start
        # Allocate budget proportionally
        remaining_budget = budget - n_coarse
        # Give more budget to better starts
        # Simple: split evenly among top_k
        per_start_budget = remaining_budget // top_k
        if per_start_budget < 20:
            per_start_budget = max(20, remaining_budget)

        best_x = top_starts[0]
        best_f = top_fvals[0]

        for i in range(top_k):
            if time.time() > deadline:
                break
            alloc = per_start_budget if i < top_k - 1 else remaining_budget - per_start_budget * i
            alloc = max(alloc, 10)
            # Don't exceed remaining time
            x_opt, f_opt = self._cma_es(func, top_starts[i], 1.5, alloc, rng, deadline)
            if f_opt < best_f:
                best_x, best_f = x_opt, f_opt

        # Phase 3: Final refinement with a tight CMA-ES around the best found
        if time.time() < deadline - 10:
            # Use remaining budget for a final polish
            remaining = budget - n_coarse - sum(per_start_budget for _ in range(top_k - 1)) - per_start_budget
            remaining = max(remaining, 20)
            # Actually, let's just check how many evals we've used
            # We don't track total evals precisely, so use time-based guard
            x_final, f_final = self._cma_es(func, best_x, 0.5, 30, rng, deadline)
            if f_final < best_f:
                best_x, best_f = x_final, f_final

        return best_x, best_f
# EVOLVE-BLOCK-END
