import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        self.best_x = np.zeros(dim)
        self.best_f = np.inf
        self.timeout = 290.0  # Safety margin from 300s limit

    def _evaluate(self, x, func):
        if time.time() - self.start_time > self.timeout:
            raise TimeoutError("Time limit exceeded")
        
        if self.calls >= self.budget:
            raise ValueError("Budget exceeded")
            
        self.calls += 1
        x = np.clip(x, -5.0, 5.0)
        f = func(x)
        
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
            
        return f

    def _nelder_mead(self, x0, func, max_iters=100):
        """
        Simple Nelder-Mead implementation with budget tracking.
        """
        n = self.dim
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(1, n + 1):
            simplex[i] = x0.copy()
            simplex[i][i - 1] += 0.5  # Perturb along axis i-1
        
        # Evaluate initial simplex
        fvals = np.zeros(n + 1)
        for i in range(n + 1):
            try:
                fvals[i] = self._evaluate(simplex[i], func)
            except (TimeoutError, ValueError):
                return

        # Main loop
        for _ in range(max_iters):
            # Check time/budget
            if time.time() - self.start_time > self.timeout or self.calls >= self.budget:
                return

            # Sort simplex
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = fvals[-1]

            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - worst)
            try:
                f_ref = self._evaluate(x_ref, func)
            except (TimeoutError, ValueError):
                return

            if f_ref < fvals[0]:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - worst)
                try:
                    f_exp = self._evaluate(x_exp, func)
                except (TimeoutError, ValueError):
                    return

                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                beta = 0.5
                x_con = centroid + beta * (worst - centroid)
                try:
                    f_con = self._evaluate(x_con, func)
                except (TimeoutError, ValueError):
                    return

                if f_con < worst_f:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink: move all points except best toward best
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        try:
                            fvals[i] = self._evaluate(simplex[i], func)
                        except (TimeoutError, ValueError):
                            return
                    # fvals[0] stays the same (best point unchanged)

            # Convergence check
            if np.std(fvals) < 1e-8:
                break

    def _cma_es(self, x0, sigma0, func, max_evals=100):
        """
        CMA-ES implementation with budget tracking.
        """
        n = self.dim
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        mu = max(1, n // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Initialize
        mean = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        # Evaluate initial point
        try:
            f_mean = self._evaluate(mean, func)
        except (TimeoutError, ValueError):
            return
        
        evals_used = 1
        max_evals = min(max_evals, self.budget - self.calls)
        
        for gen in range(1000):
            if self.calls >= self.budget or time.time() - self.start_time > self.timeout:
                break
            if evals_used >= max_evals:
                break
            
            # Generate lambda samples
            lam = 2 * mu
            x_samples = np.zeros((lam, n))
            for i in range(lam):
                try:
                    z = rng.standard_normal(n)
                    x_samples[i] = mean + sigma * (C @ z)
                    x_samples[i] = np.clip(x_samples[i], -5.0, 5.0)
                    f_samples_i = self._evaluate(x_samples[i], func)
                    evals_used += 1
                except (TimeoutError, ValueError):
                    break
            
            if self.calls >= self.budget:
                break
            
            # Sort and select top mu
            f_samples = np.array([self._evaluate(x, func) for x in x_samples[:lam]])
            idx = np.argsort(f_samples)
            x_selected = x_samples[idx[:mu]]
            f_selected = f_samples[idx[:mu]]
            
            # Update mean
            new_mean = w @ x_selected
            mean = new_mean
            
            # Update step size control
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) / chi_n * (C @ (w @ (np.linalg.inv(np.sqrt(C)) @ (x_selected.T / sigma)).T).T)
            
            # Update covariance
            delta_sigma = 1 - np.sqrt(1 - min(1, np.linalg.norm(ps)**2 / (n * (1 + 1e-12))))
            C = (1 - c_1 - c_mu) * C + c_1 * (2 - c_1) * (pc @ pc.T) + c_mu * (w @ np.outer(x_selected / sigma, x_selected / sigma))
            pc = (1 - c_c) * pc + c_c * (mean - new_mean) / sigma
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Track best
            if f_selected[0] < self.best_f:
                self.best_f = f_selected[0]
                self.best_x = x_selected[0].copy()
        
        return mean, self.best_f

    def __call__(self, func):
        # Phase 1: Latin Hypercube Sampling for global exploration
        n_lhs = 20
        rng = np.random.default_rng(42)
        
        # Generate LHS points
        lhs_points = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            intervals = np.linspace(-5, 5, n_lhs + 1)
            points_d = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points_d)
            lhs_points[:, d] = points_d

        # Evaluate LHS points
        lhs_fvals = np.zeros(n_lhs)
        for i in range(n_lhs):
            try:
                lhs_fvals[i] = self._evaluate(lhs_points[i], func)
            except (TimeoutError, ValueError):
                break

        # Phase 2: CMA-ES on best candidates for intermediate refinement
        sorted_idx = np.argsort(lhs_fvals)
        
        top_k_cma = min(3, n_lhs)
        for k in range(top_k_cma):
            if self.calls >= self.budget or time.time() - self.start_time > self.timeout:
                break
            
            start_x = lhs_points[sorted_idx[k]]
            self._cma_es(start_x, sigma0=1.0, func=func, max_evals=80)

        # Phase 3: Nelder-Mead polish on best found so far
        if self.calls < self.budget and time.time() - self.start_time < self.timeout:
            self._nelder_mead(self.best_x.copy(), func, max_iters=30)

        return self.best_x, self.best_f
