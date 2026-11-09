"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Bayesian Optimization with Gaussian Process surrogate for BBOB f16 (Weierstrass).
    Uses Expected Improvement acquisition function to guide sampling.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._eval_count = 0
        self._start_time = None
        self._best_x = None
        self._best_f = np.inf

    def _record(self, x, fx):
        """Record an evaluation and update best."""
        self._eval_count += 1
        if fx < self._best_f:
            self._best_f = fx
            self._best_x = np.array(x, dtype=np.float64)

    def _gp_predict(self, X_new, X_train, y_train, length_scale, noise_var):
        """
        Gaussian Process prediction with RBF kernel.
        Returns mean and variance at X_new points.
        """
        n_train = X_train.shape[0]
        
        # Compute kernel matrix K between training points
        # RBF kernel: k(x, x') = exp(-0.5 * ||x - x'||^2 / l^2)
        # Use squared distances for efficiency
        dist_sq = np.sum(X_train ** 2, axis=1)[:, None] + np.sum(X_train ** 2, axis=1)[None, :] - 2 * X_train @ X_train.T
        dist_sq = np.maximum(dist_sq, 0)
        K = np.exp(-0.5 * dist_sq / (length_scale ** 2))
        
        # Add noise
        K[np.diag_indices_from(K)] += noise_var
        
        # Compute kernel between test and train
        dist_sq_new = np.sum(X_new ** 2, axis=1)[:, None] + np.sum(X_train ** 2, axis=1)[None, :] - 2 * X_new @ X_train.T
        dist_sq_new = np.maximum(dist_sq_new, 0)
        K_s = np.exp(-0.5 * dist_sq_new / (length_scale ** 2))
        
        # Solve for alpha
        try:
            alpha = np.linalg.solve(K, y_train)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(K, y_train, rcond=None)[0]
        
        # Predictions
        mean = K_s @ alpha
        v = np.linalg.solve(K, K_s.T)
        var = np.exp(0) - np.sum(v ** 2, axis=1)  # exp(0) = 1 for unit signal variance
        var = np.maximum(var, 1e-10)
        
        return mean, var

    def _expected_improvement(self, mean, var, best_f, xi=0.01):
        """Compute Expected Improvement acquisition function."""
        sigma = np.sqrt(var)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (best_f - mean - xi) / sigma
            ei = (best_f - mean - xi) * (z * 0.5 + 0.5) + sigma * np.exp(-0.5 * z ** 2)
        # Handle where sigma is very small
        mask = sigma < 1e-10
        ei[mask] = 0.0
        # Handle where z is very negative (far below best)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei

    def __call__(self, func):
        self._start_time = time.time()
        deadline = 295.0  # safety margin before 330s kill
        
        n = self.dim
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial Sampling
        # Sample 15 points to get a good start for the GP
        n_init = 15
        init_points = rng.uniform(-4.5, 4.5, (n_init, n))
        init_fvals = np.array([float(func(p)) for p in init_points])
        for i in range(n_init):
            self._record(init_points[i], init_fvals[i])
        
        best_idx = np.argmin(init_fvals)
        X_train = init_points.copy()
        y_train = init_fvals.copy()
        
        if self._eval_count >= self.budget:
            return self._best_x, self._best_f
        
        # GP hyperparameters
        length_scale = 1.0
        noise_var = 1e-6
        signal_var = 1.0
        
        # Phase 2: Bayesian Optimization Loop
        # We'll use a batch size of 1 for simplicity, but could use batch EI
        max_iter = self.budget - self._eval_count
        
        for iteration in range(max_iter):
            # Check time
            if (time.time() - self._start_time) > deadline:
                break
            
            # Check budget
            if self._eval_count >= self.budget:
                break
            
            # Sample candidate points for acquisition function optimization
            n_candidates = 200
            candidates = rng.uniform(-5.0, 5.0, (n_candidates, n))
            
            # Predict mean and variance
            mean, var = self._gp_predict(candidates, X_train, y_train, length_scale, noise_var)
            
            # Compute Expected Improvement
            best_f_so_far = self._best_f
            ei = self._expected_improvement(mean, var, best_f_so_far, xi=0.01)
            
            # Select best candidate
            best_candidate_idx = np.argmax(ei)
            x_new = candidates[best_candidate_idx].copy()
            
            # Evaluate
            fx_new = float(func(x_new))
            self._record(x_new, fx_new)
            
            # Update training data
            X_train = np.vstack([X_train, x_new])
            y_train = np.append(y_train, fx_new)
            
            # Adaptive length scale: decrease as we converge
            # Simple heuristic: if improvement is small, reduce length scale
            if iteration > 10:
                # Reduce length scale slightly to focus on local regions
                length_scale *= 0.99
                length_scale = max(length_scale, 0.1)
        
        # Phase 3: Local Polish
        # Use a simple coordinate descent or Nelder-Mead for final polish
        remaining = self.budget - self._eval_count
        if remaining > 5 and self._best_x is not None:
            # Simple coordinate descent polish
            x = self._best_x.copy()
            fx = self._best_f
            step = 0.1
            
            for _ in range(remaining):
                if (time.time() - self._start_time) > deadline:
                    break
                if self._eval_count >= self.budget:
                    break
                
                improved = False
                for d in range(n):
                    # Try positive step
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x_trial[d] + step, -5.0, 5.0)
                    fx_trial = float(func(x_trial))
                    self._record(x_trial, fx_trial)
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        improved = True
                    
                    # Try negative step
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x_trial[d] - step, -5.0, 5.0)
                    fx_trial = float(func(x_trial))
                    self._record(x_trial, fx_trial)
                    if fx_trial < fx:
                        x = x_trial
                        fx = fx_trial
                        improved = True
                
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
        
        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
