import numpy as np

class Optimizer:
    """CMA-ES with multi-start strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, sigma, budget, rng):
        """Run CMA-ES starting from x0 with given sigma and budget."""
        n = self.dim
        if budget <= 0:
            return x0, func(x0)

        # Population size
        lambda_ = 4 + int(3 * np.log(n))
        # Ensure lambda_ doesn't exceed budget
        if lambda_ > budget:
            lambda_ = max(2, budget)
            
        # Weights
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)

        # Learning rates
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 2 / mueff) / ((n + 2)**2 + mueff))

        # State
        mean = x0.copy()
        ps = np.zeros(n)
        C = np.eye(n)
        Csqrt = np.eye(n)
        Cinv = np.eye(n)
        Cinv_sqrt = np.eye(n)

        best_x = x0.copy()
        best_f = func(x0)
        evals_used = 1  # Initial evaluation

        while evals_used < budget:
            remaining = budget - evals_used
            current_lambda = min(lambda_, remaining)
            if current_lambda == 0:
                break

            Y = np.zeros((current_lambda, n))
            F = np.zeros(current_lambda)

            for i in range(current_lambda):
                z = rng.normal(0, 1, n)
                x = mean + sigma * (Csqrt @ z)
                x = np.clip(x, -5, 5)
                Y[i] = x
                F[i] = func(x)
                evals_used += 1

                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x.copy()

            # Update CMA-ES parameters
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]

            # Use top mu
            Y = Y[:mu]

            m_new = np.sum(weights[:, np.newaxis] * Y, axis=0)

            # Update step size control vector
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (Cinv_sqrt @ ((mean - m_new) / sigma))

            # Hessian correction
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2)) < (1.4 + 2 / (n + 1)) * np.sqrt(n + 1)

            # Update covariance matrix
            delta_h = (1 - hsig) * cs * (2 - cs)
            C = (1 + c1 * delta_h * cc) * C + c1 * (np.outer(ps, ps) - C) + \
                cmu * np.sum(weights[:, np.newaxis] * np.outer(Y - m_new, Y - m_new), axis=0)

            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Update square root and inverse square root
            try:
                U, S, Vt = np.linalg.svd(C)
                Csqrt = U @ np.diag(np.sqrt(S)) @ Vt
                Cinv = np.linalg.inv(C)
                U, S, Vt = np.linalg.svd(Cinv)
                Cinv_sqrt = U @ np.diag(1 / np.sqrt(S)) @ Vt
            except np.linalg.LinAlgError:
                # Reset if numerical issues
                C = np.eye(n)
                Csqrt = np.eye(n)
                Cinv = np.eye(n)
                Cinv_sqrt = np.eye(n)
                sigma = max(sigma * 0.5, 1e-3)

            mean = m_new

            # Adapt sigma
            sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(ps) / np.sqrt(n) - 1))

            # Safety bounds
            if sigma < 1e-5:
                break
            if sigma > 5:
                sigma = 5

        return best_x, best_f

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Strategy: Multi-start CMA-ES
        # 1. Sample a few starting points
        # 2. Run CMA-ES from the best ones
        
        # Number of starts
        n_starts = 3
        # Budget per start
        # Reserve 10% for initial sampling?
        # Let's use 50% of budget for sampling and finding best starts?
        # No, CMA-ES is efficient.
        # Let's do:
        # 1. Evaluate 20 random points.
        # 2. Pick the best n_starts.
        # 3. Run CMA-ES from each with remaining budget / n_starts.
        
        n_sample = 20
        if budget < n_sample + n_starts * 10:
            # Fallback to single CMA-ES
            x0 = np.zeros(dim)
            return self._cma_es(func, x0, 2.0, budget, rng)

        # Sample points
        X_sample = rng.uniform(-5, 5, (n_sample, dim))
        F_sample = np.array([func(x) for x in X_sample])
        
        # Find best starts
        idx = np.argsort(F_sample)[:n_starts]
        starts = X_sample[idx]
        
        remaining_budget = budget - n_sample
        budget_per_start = remaining_budget // n_starts
        
        best_x = starts[0]
        best_f = F_sample[idx[0]]
        
        for i in range(n_starts):
            x_opt, f_opt = self._cma_es(func, starts[i], 1.5, budget_per_start, rng)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt
                
        return best_x, best_f
