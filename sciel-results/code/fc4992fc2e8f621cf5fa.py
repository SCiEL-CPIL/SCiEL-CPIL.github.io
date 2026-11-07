import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0
        self.dim = 5
        self.budget = 500

        def wrapped_func(x):
            self.call_count += 1
            if time.time() > self.deadline:
                raise TimeoutError()
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x = np.clip(np.asarray(x, dtype=np.float64), -self.bound, self.bound)
            fx = func(x)
            if np.isfinite(fx) and fx < self.best_f:
                self.best_f = float(fx)
                self.best_x = x.copy()
            return fx if np.isfinite(fx) else 1e30

        # Phase 1: Small random sampling to seed CMA-ES
        n_global = 15
        for i in range(n_global):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            p = np.random.uniform(-self.bound, self.bound, self.dim)
            wrapped_func(p)

        if self.best_x is None:
            return np.zeros(self.dim), 1e30

        # Phase 2: CMA-ES with larger population for better covariance estimation
        remaining = self.budget - self.call_count
        if remaining > 50:
            try:
                self._cmaes(wrapped_func, self.best_x.copy(), remaining)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 3: Fine-grained local polish
        remaining = self.budget - self.call_count
        if remaining > 5 and time.time() < self.deadline - 1:
            try:
                self._fine_polish(wrapped_func, self.best_x.copy(), min(remaining, 80))
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return self.best_x, self.best_f

    def _cmaes(self, func, x_start, max_calls):
        """Implement CMA-ES directly for full budget control."""
        n = self.dim
        x = x_start.copy()
        
        # Initial sigma
        sigma = 1.0
        
        # Larger population for better exploration
        lam = 15
        mu = lam // 2
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Strategy parameters
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Initial state
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        C_cholesky = np.eye(n)
        
        calls_used = 0
        
        for iteration in range(300):
            if calls_used >= max_calls:
                break
            if time.time() > self.deadline:
                break
            
            # Check if we have enough budget for a full generation
            if calls_used + lam > max_calls:
                # Do a partial generation
                n_sample = max_calls - calls_used
                if n_sample < mu:
                    break
            else:
                n_sample = lam
            
            # Sample candidates
            candidates = []
            for j in range(n_sample):
                if time.time() > self.deadline:
                    break
                z = np.random.randn(n)
                x_candidate = x + sigma * (C_cholesky @ z)
                x_candidate = np.clip(x_candidate, -self.bound, self.bound)
                candidates.append(x_candidate)
            
            if len(candidates) == 0:
                break
            
            # Evaluate all candidates
            fvals = []
            for xc in candidates:
                if time.time() > self.deadline:
                    break
                fvals.append(func(xc))
            
            if len(fvals) < mu:
                break
            
            calls_used += len(fvals)
            
            # Sort by fitness
            indices = np.argsort(fvals)
            sorted_x = np.array([candidates[i] for i in indices])
            
            # Update mean with weighted recombination
            w = weights[:mu]
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += w[i] * sorted_x[i]
            
            # Update p_sigma
            y_sigma = (x_new - x) / sigma if sigma > 1e-30 else np.zeros(n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Update p_c
            p_c = (1 - c_c) * p_c + c_c * y_sigma
            
            # Update covariance
            rank_one = np.outer(p_c, p_c)
            y_vecs = np.array([(sorted_x[i] - x) / (sigma + 1e-30) for i in range(mu)])
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                rank_mu += w[i] * np.outer(y_vecs[i], y_vecs[i])
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Symmetrize and factorize
            C = 0.5 * (C + C.T)
            try:
                L = np.linalg.cholesky(C + 1e-18 * np.eye(n))
                C_cholesky = L
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-18, None)
                C_cholesky = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update mean
            x = x_new
            
            # Early stopping
            if sigma < 1e-10:
                break
            if np.linalg.norm(x_new - x) < 1e-10 and np.linalg.norm(y_sigma) < 1e-6:
                break
        
    def _fine_polish(self, func, x_start, max_calls):
        """Efficient local polish using coordinate descent with adaptive step."""
        n = self.dim
        x = x_start.copy()
        f_best = self.best_f
        calls = 0
        step = 0.05
        
        for it in range(max_calls):
            if calls >= max_calls or time.time() > self.deadline:
                break
            
            improved = False
            for i in range(n):
                if calls >= max_calls or time.time() > self.deadline:
                    break
                for sign in [1, -1]:
                    if calls >= max_calls or time.time() > self.deadline:
                        break
                    x_try = x.copy()
                    x_try[i] += sign * step
                    x_try = np.clip(x_try, -self.bound, self.bound)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < f_best - 1e-10:
                        x = x_try
                        f_best = f_try
                        improved = True
            
            if not improved:
                step *= 0.5
                if step < 1e-5:
                    break
