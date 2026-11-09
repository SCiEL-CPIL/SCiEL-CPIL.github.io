"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """
    Optimizer for BBOB f16 (Weierstrass) using a hybrid strategy:
    1. Initial random sampling to establish a baseline.
    2. CMA-ES with adaptive restarts to navigate the multimodal landscape.
    3. Bayesian Optimization (GP + EI) for global exploration when CMA-ES stagnates.
    4. Local coordinate descent polish for final refinement.
    
    The Weierstrass function is highly multimodal and non-convex. CMA-ES is
    robust to this, but can get stuck in local minima. Restarts help escape.
    The BO phase uses a Gaussian Process surrogate to model the landscape
    and guide sampling via Expected Improvement, providing a different
    structural perspective than population-based methods.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._eval_count = 0
        self._start_time = None
        self._best_x = None
        self._best_f = np.inf
        self._X = []  # observed points for BO
        self._Y = []  # observed values for BO

    def _record(self, x, fx):
        """Record an evaluation and update best."""
        self._eval_count += 1
        self._X.append(np.array(x, dtype=np.float64))
        self._Y.append(float(fx))
        if fx < self._best_f:
            self._best_f = fx
            self._best_x = np.array(x, dtype=np.float64)

    def _cmaes_step(self, func, x_mean, sigma, C, pc, ps, step_size, rng):
        """
        Single CMA-ES step. Returns updated parameters and best from this step.
        Uses a simplified but effective CMA-ES implementation.
        """
        n = self.dim
        # Compute eigendecomposition of C
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-20)
        
        # Sample lambda points
        lam = int(4 + 3 * np.log(n))
        weights = np.log(lam / 2) - np.arange(1, lam + 1)
        weights = np.maximum(weights, 1e-10)
        weights /= weights.sum()
        mu = int(lam / 2)
        nu = 1.0 / (2 * (mu + 0.5))
        
        # Sampling
        z = rng.standard_normal((lam, n))
        # Transform using Cholesky-like: C = V * D * V^T
        samples = (z @ eigvecs.T * np.sqrt(eigvals)).T  # shape (lam, n)
        candidates = x_mean + sigma * samples
        
        # Evaluate
        fvals = np.array([float(func(candidates[i])) for i in range(lam)])
        for i in range(lam):
            self._record(candidates[i], fvals[i])
        
        # Sort by fitness
        order = np.argsort(fvals)
        selected = candidates[order[:mu]]
        
        # Weighted mean
        w = weights[:mu]
        w = w / w.sum()
        x_new = np.sum(w[:, None] * selected, axis=0)
        
        # Update covariance
        delta = (x_new - x_mean) / sigma
        pc = (1 - step_size) * pc + step_size * delta
        
        # Rank-1 update
        C = (1 + 2 * nu * (np.dot(pc, pc) - 1) - step_size) * C + \
            (1 + 2 * nu - step_size) * np.outer(delta, delta)
        
        # Rank-mu update
        if step_size > 0:
            diff = (selected - x_mean) / sigma
            C = (1 - 2 * nu * (np.dot(pc, pc) - 1) - step_size + step_size * nu) * C
            for i in range(mu):
                C += w[i] * np.outer(diff[i], diff[i])
        
        # Ensure positive semi-definite
        eigvals_new, eigvecs_new = np.linalg.eigh(C)
        eigvals_new = np.maximum(eigvals_new, 1e-20)
        C = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T
        
        # Adaptive sigma
        ps = (1 - step_size) * ps + step_size * (delta / np.sqrt(np.trace(C) / n))
        sigma *= np.exp(0.5 * (np.dot(ps, ps) / n - 1))
        sigma = np.clip(sigma, 1e-10, 10.0)
        
        return x_new, sigma, C, pc, ps, fvals[order[0]], candidates[order[0]]

    def _gp_predict(self, X_new, X_train, Y_train, length_scale=1.0, sigma_f=1.0, noise=1e-8):
        """
        Gaussian Process prediction.
        Returns mean and std for X_new.
        """
        n = len(X_train)
        if n == 0:
            return np.zeros(len(X_new)), np.full(len(X_new), sigma_f)
        
        # RBF kernel
        def kernel(A, B):
            sq_dists = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
            sq_dists = np.maximum(sq_dists, 0)
            return sigma_f**2 * np.exp(-0.5 * sq_dists / (length_scale**2))
        
        K = kernel(X_train, X_train) + noise * np.eye(n)
        K_star = kernel(X_train, X_new)
        
        try:
            alpha = np.linalg.solve(K, Y_train)
            mean = K_star.T @ alpha
            v = np.linalg.solve(K, K_star)
            var = sigma_f**2 - np.sum(K_star**2 * v, axis=0)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
        except np.linalg.LinAlgError:
            mean = np.full(len(X_new), np.mean(Y_train))
            std = np.full(len(X_new), np.std(Y_train) + 1e-6)
        
        return mean, std

    def _expected_improvement(self, mean, std, best_f, xi=0.01):
        """Compute Expected Improvement."""
        improvement = best_f - mean - xi
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        ei = np.where(std > 1e-10, ei, 0)
        return ei

    def _acquire_next_point(self, rng, n_candidates=200):
        """
        Select next point using EI acquisition with random candidates.
        """
        n = self.dim
        X_train = np.array(self._X)
        Y_train = np.array(self._Y)
        
        # Adaptive length scale based on spread of data
        if len(self._X) >= 2:
            spread = np.std(X_train, axis=0)
            length_scale = np.clip(np.mean(spread), 0.1, 5.0)
        else:
            length_scale = 1.0
        
        # Adaptive sigma_f
        if len(self._Y) >= 2:
            sigma_f = np.clip(np.std(Y_train), 0.1, 10.0)
        else:
            sigma_f = 1.0
        
        # Generate random candidates
        candidates = rng.uniform(-5, 5, (n_candidates, n))
        
        # Predict and compute EI
        mean, std = self._gp_predict(candidates, X_train, Y_train, length_scale, sigma_f)
        ei = self._expected_improvement(mean, std, self._best_f)
        
        # Also consider points near current best with small perturbations
        if self._best_x is not None and len(self._X) > 5:
            local_candidates = self._best_x[None, :] + rng.normal(0, 0.1, (50, n))
            local_candidates = np.clip(local_candidates, -5, 5)
            mean_local, std_local = self._gp_predict(local_candidates, X_train, Y_train, length_scale, sigma_f)
            ei_local = self._expected_improvement(mean_local, std_local, self._best_f)
            
            # Combine: pick from global if EI is high, otherwise from local
            global_best_idx = np.argmax(ei)
            local_best_idx = np.argmax(ei_local)
            
            # 70% chance global, 30% chance local
            if rng.random() < 0.7:
                return candidates[global_best_idx]
            else:
                return local_candidates[local_best_idx]
        
        return candidates[np.argmax(ei)]

    def __call__(self, func):
        self._start_time = time.time()
        deadline = 295.0  # safety margin before 330s kill
        
        n = self.dim
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial Sampling
        # Sample 10 points to get a good start and estimate scale
        n_init = 10
        init_points = rng.uniform(-4.5, 4.5, (n_init, n))
        init_fvals = np.array([float(func(p)) for p in init_points])
        for i in range(n_init):
            self._record(init_points[i], init_fvals[i])
        
        best_idx = np.argmin(init_fvals)
        x_mean = init_points[best_idx].copy()
        self._best_x = x_mean.copy()
        self._best_f = init_fvals[best_idx]
        
        if self._eval_count >= self.budget:
            return self._best_x, self._best_f
        
        # Phase 2: CMA-ES with Restarts
        step_size = 1.0 / (n + 3) ** 1.5
        max_evals_per_run = 80  # Limit evaluations per CMA-ES run to encourage restarts
        stagnation_threshold = 30 # If no improvement in 30 evals, restart
        
        # Track when to switch to BO
        cmaes_total_evals = 0
        bo_triggered = False
        bo_max_evals = 100  # Max evaluations for BO phase
        
        while self._eval_count < self.budget:
            # Check time
            if (time.time() - self._start_time) > deadline:
                break
            
            # Reserve budget for local polish
            if self.budget - self._eval_count < 40:
                break
                
            # Decide whether to use CMA-ES or BO
            # If CMA-ES has been running for a while without much progress, switch to BO
            if not bo_triggered and cmaes_total_evals > 150:
                bo_triggered = True
                bo_evals = 0
            elif bo_triggered:
                # Use BO
                if (time.time() - self._start_time) > deadline:
                    break
                if self._eval_count >= self.budget:
                    break
                if bo_evals >= bo_max_evals:
                    # Go back to CMA-ES
                    bo_triggered = False
                    cmaes_total_evals = 0
                    continue
                
                try:
                    x_next = self._acquire_next_point(rng)
                    f_next = float(func(x_next))
                    self._record(x_next, f_next)
                    bo_evals += 1
                except Exception:
                    # Fallback to random sampling
                    x = rng.uniform(-5, 5, n)
                    f = float(func(x))
                    self._record(x, f)
                    bo_evals += 1
                continue
            
            # Use CMA-ES
            # Reset CMA-ES state for this run
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            
            # Start from current best
            x_mean = self._best_x.copy()
            
            # Initial sigma for this run
            # First run: larger sigma for exploration
            # Subsequent runs: smaller sigma for exploitation
            if self._eval_count == n_init:
                sigma = 2.0
            else:
                sigma = 0.5
                
            # Track stagnation
            best_f_at_start = self._best_f
            evals_at_start = self._eval_count
            last_improvement_eval = self._eval_count
            
            # Inner loop for this CMA-ES run
            while self._eval_count < self.budget:
                if (time.time() - self._start_time) > deadline:
                    break
                
                # Check budget
                if self._eval_count >= self.budget:
                    break
                    
                # Check stagnation
                if self._eval_count - last_improvement_eval > stagnation_threshold:
                    break
                    
                # Check if sigma is too small
                if sigma < 1e-8:
                    break
                
                try:
                    x_mean, sigma, C, pc, ps, best_f_step, best_x_step = \
                        self._cmaes_step(func, x_mean, sigma, C, pc, ps, step_size, rng)
                except Exception:
                    break
                
                cmaes_total_evals += 1
                
                # If we found a better point, update our global best
                if best_f_step < self._best_f:
                    self._best_f = best_f_step
                    self._best_x = best_x_step.copy()
                    last_improvement_eval = self._eval_count
            
            # Safety break to prevent infinite loop if no progress at all
            if self._eval_count == evals_at_start:
                break

        # Phase 3: Local Polish
        # Use coordinate descent to refine the best point found
        remaining = self.budget - self._eval_count
        if remaining > 5 and self._best_x is not None:
            x = self._best_x.copy()
            f = self._best_f
            # Start with larger steps, then refine
            step_sizes = [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]
            
            for ss in step_sizes:
                if self._eval_count >= self.budget or (time.time() - self._start_time) > deadline:
                    break
                improved = True
                while improved and self._eval_count < self.budget:
                    improved = False
                    for i in range(n):
                        if (time.time() - self._start_time) > deadline:
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[i] += direction * ss
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = float(func(x_trial))
                            self._record(x_trial, f_trial)
                            if f_trial < f - 1e-12:
                                x = x_trial
                                f = f_trial
                                improved = True
                                break
                        if self._eval_count >= self.budget:
                            break
                    if self._eval_count >= self.budget:
                        break
                
                if f < self._best_f:
                    self._best_f = f
                    self._best_x = x.copy()
        
        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
