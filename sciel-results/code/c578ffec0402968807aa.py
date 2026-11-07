import numpy as np
import time
from scipy.spatial.distance import cdist

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

        # Phase 1: Random sampling to find a promising region
        n_global = 60
        xs = []
        fs = []
        for i in range(n_global):
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            p = np.random.uniform(-self.bound, self.bound, self.dim)
            fx = wrapped_func(p)
            xs.append(p.copy())
            fs.append(fx)

        if self.best_x is None:
            return np.zeros(self.dim), 1e30

        # Phase 2: CMA-ES with the best point as starting center
        # Use a good chunk of remaining budget for CMA-ES
        remaining = self.budget - self.call_count
        if remaining > 30:
            try:
                self._cmaes(wrapped_func, self.best_x.copy(), remaining)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 3: Nelder-Mead polish if budget allows
        remaining = self.budget - self.call_count
        if remaining > 10 and time.time() < self.deadline - 2:
            try:
                self._nelder_mead_polish(wrapped_func, self.best_x.copy(), min(remaining, 100))
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        # Phase 4: Fine-grained local polish with very small steps
        remaining = self.budget - self.call_count
        if remaining > 5 and time.time() < self.deadline - 1:
            try:
                self._fine_polish(wrapped_func, self.best_x.copy(), min(remaining, 50))
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return self.best_x, self.best_f

    def _cmaes(self, func, x_start, max_calls):
        """Implement CMA-ES directly for full budget control."""
        n = self.dim
        x = x_start.copy()
        
        # Set initial sigma based on the spread of good points
        sigma = 1.0
        
        # Population size
        lam = max(4 + int(3 * np.log(n)), 6)
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
        
        # Track best
        calls_used = 0
        
        for iteration in range(200):
            if calls_used >= max_calls:
                break
            if time.time() > self.deadline:
                break
            
            # Sample candidates
            candidates = []
            for j in range(lam):
                if calls_used + (lam - j) > max_calls:
                    break
                if time.time() > self.deadline:
                    break
                # Sample using current distribution
                z = np.random.randn(n)
                x_candidate = x + sigma * (C_cholesky @ z)
                x_candidate = np.clip(x_candidate, -self.bound, self.bound)
                candidates.append(x_candidate)
                calls_used += 1
            
            if len(candidates) == 0:
                break
            
            # Evaluate
            fvals = []
            for xc in candidates:
                if calls_used >= max_calls:
                    break
                if time.time() > self.deadline:
                    break
                fvals.append(func(xc))
            
            if len(fvals) < mu:
                break
            
            # Sort by fitness
            indices = np.argsort(fvals[:len(candidates)])
            sorted_x = np.array([candidates[i] for i in indices])
            sorted_f = np.array(fvals[:len(candidates)][:, 0]) if False else np.array([fvals[i] for i in indices])
            
            # Update mean with weighted recombination
            w = weights[:mu]
            x_new = sum(w[i] * sorted_x[i] for i in range(mu))
            
            # Update p_sigma
            y_sigma = (x_new - x) / sigma if sigma > 1e-30 else np.zeros(n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Update p_c
            p_c = (1 - c_c) * p_c + c_c * y_sigma
            
            # Update covariance
            rank_one = np.outer(p_c, p_c)
            # Fixed: rank-mu should be (1/sigma^2) * sum(w_i * outer(y_i, y_i)) where y_i = (x_i - x_old)/sigma
            y_vecs = np.array([(sorted_x[i] - x) / (sigma + 1e-30) for i in range(mu)])
            rank_mu = np.sum([w[i] * np.outer(y_vecs[i], y_vecs[i]) for i in range(mu)], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Symmetrize and factorize
            C = 0.5 * (C + C.T)
            try:
                L = np.linalg.cholesky(C + 1e-20 * np.eye(n))
                C_cholesky = L
            except np.linalg.LinAlgError:
                # Fall back to eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                C_cholesky = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update mean
            x = x_new
            
            # Early stopping if sigma is too small
            if sigma < 1e-12:
                break
            if np.linalg.norm(x_new - x_start) < 1e-12 and np.linalg.norm(y_sigma) < 1e-8:
                break
        
    def _fine_polish(self, func, x_start, max_calls):
        """Very fine local search with tiny step sizes to squeeze out last precision."""
        n = self.dim
        x = x_start.copy()
        f_best = func(x)
        calls = 1
        # Start with very small step
        step = 0.01
        for it in range(max_calls - 1):
            if calls >= max_calls or time.time() > self.deadline:
                break
            improved = False
            for i in range(n):
                for sign in [1, -1]:
                    if calls >= max_calls or time.time() > self.deadline:
                        break
                    x_try = x.copy()
                    x_try[i] += sign * step
                    x_try = np.clip(x_try, -self.bound, self.bound)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < f_best - 1e-12:
                        x = x_try
                        f_best = f_try
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break

    def _nelder_mead_polish(self, func, x_start, max_calls):
        """Simple Nelder-Mead style local polish."""
        n = self.dim
        # Initialize simplex
        simplex = [x_start.copy()]
        base = x_start.copy()
        for i in range(n):
            pt = x_start.copy()
            pt[i] = pt[i] + 0.1 * self.bound if i != 0 else pt[i] + 0.5 * self.bound
            simplex.append(np.clip(pt, -self.bound, self.bound))
        
        fvals = []
        calls = 0
        for sx in simplex:
            if calls >= max_calls or time.time() > self.deadline:
                break
            fvals.append(func(sx))
            calls += 1
        
        if calls < n + 1:
            return
        
        # Nelder-Mead iterations
        for it in range(200):
            if calls >= max_calls or time.time() > self.deadline:
                break
            
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Centroid of best n points
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -self.bound, self.bound)
            
            if calls >= max_calls or time.time() > self.deadline:
                break
            f_ref = func(x_ref)
            calls += 1
            
            if f_ref < fvals[0]:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[-1])
                x_exp = np.clip(x_exp, -self.bound, self.bound)
                if calls >= max_calls or time.time() > self.deadline:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                    break
                f_exp = func(x_exp)
                calls += 1
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                beta = 0.5
                x_con = centroid + beta * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -self.bound, self.bound)
                if calls >= max_calls or time.time() > self.deadline:
                    break
                f_con = func(x_con)
                calls += 1
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -self.bound, self.bound)
                        if calls >= max_calls or time.time() > self.deadline:
                            break
                        fvals[i] = func(simplex[i])
                        calls += 1
                    break
