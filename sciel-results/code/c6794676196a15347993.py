import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: 300s limit, use 280s
        deadline = start_time + 280 

        calls = [0]
        
        def eval(x):
            calls[0] += 1
            return float(func(np.asarray(x, dtype=float)))

        # --- Phase 1: Global Exploration ---
        # We need a diverse set of points to fit the initial surrogate model.
        # The Bent Cigar can have a shifted minimum.
        rng = np.random.default_rng(42)
        
        points = []
        values = []
        
        best_x = None
        best_f = np.inf
        
        # Start with center and some random points
        init_points = [np.zeros(dim)]
        for _ in range(10):
            init_points.append(rng.uniform(-5, 5, dim))
            
        for x in init_points:
            if calls[0] >= budget or time.time() > deadline:
                break
            f = eval(x)
            points.append(x)
            values.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(dim)
            best_f = eval(best_x)
            points.append(best_x)
            values.append(best_f)

        # --- Phase 2: Surrogate Model Optimization ---
        # We will maintain a history of the last K points to fit the quadratic model.
        # K should be at least the number of parameters in the quadratic model.
        # For dim d, params = d(d+1)/2 + d + 1.
        # For d=5, params = 15 + 5 + 1 = 21.
        # Let's use K = 30 for a slightly overdetermined system for stability.
        K = 30
        
        history_x = points.copy()
        history_f = values.copy()
        
        # Regularization parameter for Hessian inverse
        reg = 1e-6
        
        # Line search parameters
        alpha_start = 1.0
        alpha_factor = 0.5
        max_alpha_trials = 10
        
        stagnation_count = 0
        max_stagnation = 5

        while calls[0] < budget and time.time() < deadline:
            # 1. Fit Quadratic Surrogate Model
            # We use the last K points from history.
            # If we have fewer than K points, use all.
            n_hist = len(history_x)
            if n_hist < 2:
                break
                
            # Select points for fitting. 
            # To avoid stale data, we prioritize recent points, but keep the best point in the fit if possible.
            # Actually, for local convergence, recent points are more relevant.
            fit_x = history_x[-K:]
            fit_f = history_f[-K:]
            
            # Fit Q(x) = c0 + sum ci xi + sum cij xi xj
            # Linear least squares: y = X * theta
            # Features: 1, x1..xd, x1^2..xd^2, x1x2..x_{d-1}xd
            n_pts = len(fit_x)
            if n_pts < 10:
                # Not enough points for robust fit, do a random step
                step = rng.normal(0, 0.5, dim)
                new_x = np.clip(best_x + step, -5, 5)
                new_f = eval(new_x)
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_x.copy()
                history_x.append(new_x)
                history_f.append(new_f)
                if len(history_x) > K * 2:
                    history_x = history_x[-K:]
                    history_f = history_f[-K:]
                continue

            # Build design matrix
            # 1 constant term
            # d linear terms
            # d(d+1)/2 quadratic terms
            n_lin = dim
            n_quad = dim * (dim + 1) // 2
            n_params = 1 + n_lin + n_quad
            
            X = np.ones((n_pts, n_params))
            for i in range(n_pts):
                xi = fit_x[i]
                # Linear terms
                X[i, 1:1+n_lin] = xi
                # Quadratic terms
                idx = 1 + n_lin
                for r in range(dim):
                    for c in range(r, dim):
                        X[i, idx] = xi[r] * xi[c]
                        idx += 1
                        
            y = np.array(fit_f)
            
            # Solve least squares
            try:
                theta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                theta = np.zeros(n_params)
            
            # Extract coefficients
            c0 = theta[0]
            c_lin = theta[1:1+n_lin]
            
            # Reconstruct Hessian H from theta
            # Q(x) = c0 + c_lin^T x + x^T A x
            # where H = 2A
            # theta indices for quad terms:
            # idx starts at 1 + n_lin
            A = np.zeros((dim, dim))
            idx = 1 + n_lin
            for r in range(dim):
                for c in range(r, dim):
                    coeff = theta[idx]
                    A[r, c] = coeff
                    if r != c:
                        A[c, r] = coeff
                    idx += 1
                    
            H = 2 * A
            
            # Compute Gradient and Newton Step at best_x
            grad = c_lin + 2 * A.dot(best_x)
            
            # Regularize Hessian to ensure invertibility and positive definiteness
            # If H is not positive definite, the "minimum" might be a saddle or max.
            # We add a small identity to H.
            H_reg = H + reg * np.eye(dim)
            
            try:
                # Solve H_reg * step = -grad
                step = np.linalg.solve(H_reg, -grad)
            except np.linalg.LinAlgError:
                # Fallback to gradient step if singular
                step = -grad
                # Scale step to reasonable size
                norm_step = np.linalg.norm(step)
                if norm_step > 1e-8:
                    step = step / norm_step * 0.5
                else:
                    step = rng.normal(0, 0.1, dim)

            # Clip step to ensure we stay within bounds? 
            # It's better to clip the resulting x.
            
            # Line Search
            alpha = alpha_start
            found_improvement = False
            best_trial_x = None
            best_trial_f = np.inf
            
            for _ in range(max_alpha_trials):
                trial_x = best_x + alpha * step
                # Clip to box
                trial_x = np.clip(trial_x, -5, 5)
                
                if np.all(trial_x == best_x):
                    alpha *= alpha_factor
                    continue
                    
                trial_f = eval(trial_x)
                
                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial_x.copy()
                    found_improvement = True
                    break
                else:
                    alpha *= alpha_factor
                    if alpha < 1e-4:
                        break
            
            if found_improvement:
                stagnation_count = 0
                # Update history
                history_x.append(best_x)
                history_f.append(best_f)
                if len(history_x) > K:
                    history_x = history_x[-K:]
                    history_f = history_f[-K:]
            else:
                stagnation_count += 1
                # If stuck, add a random perturbation to escape
                # This helps if the surrogate model is misleading or we are in a flat region
                pert_size = 0.5 * (1 + stagnation_count)
                pert_x = np.clip(best_x + rng.normal(0, pert_size, dim), -5, 5)
                pert_f = eval(pert_x)
                
                if pert_f < best_f:
                    best_f = pert_f
                    best_x = pert_x.copy()
                    found_improvement = True
                    stagnation_count = 0
                else:
                    # Even if not better, add to history to update model
                    # But be careful not to fill history with bad points.
                    # Only add if it's somewhat close or if we are desperate.
                    if stagnation_count > max_stagnation:
                         history_x.append(pert_x)
                         history_f.append(pert_f)
                         if len(history_x) > K:
                             history_x = history_x[-K:]
                             history_f = history_f[-K:]
                         stagnation_count = 0
                         # Reset alpha search for next time
                         alpha_start = 1.0
                    else:
                         # Try a smaller random step
                         small_pert_x = np.clip(best_x + rng.normal(0, 0.1, dim), -5, 5)
                         small_pert_f = eval(small_pert_x)
                         if small_pert_f < best_f:
                             best_f = small_pert_f
                             best_x = small_pert_x.copy()
                         history_x.append(small_pert_x)
                         history_f.append(small_pert_f)
                         if len(history_x) > K:
                             history_x = history_x[-K:]
                             history_f = history_f[-K:]

            # Check time
            if time.time() > deadline - 5:
                break

        return best_x, best_f
