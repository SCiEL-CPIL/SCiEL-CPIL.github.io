"""Surrogate-guided optimizer with coordinate line search and CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Surrogate-guided multi-start optimizer with coordinate line search and CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(7)

        # Track best solution
        best_x = None
        best_f = np.inf
        eval_count = 0

        def check_time():
            return time.time() < deadline - 3.0

        def try_update(x, fx):
            nonlocal best_x, best_f, eval_count
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        def clip_box(x):
            return np.clip(x, -5.0, 5.0)

        # ===== Phase 1: Structured Initialization with Surrogate Guidance =====
        # Evaluate a diverse set of points to find a good starting basin.
        # This is crucial for AOCC as early progress is heavily weighted.
        n_init = min(30, budget - 20)
        init_points = []
        init_points.append(np.zeros(n))
        
        # Random uniform points in the likely region [-4, 4]
        for _ in range(max(1, n_init // 3)):
            init_points.append(rng.uniform(-4.0, 4.0, n))
        
        # Closer to center
        for _ in range(max(1, n_init // 3)):
            init_points.append(rng.uniform(-2.0, 2.0, n))
        
        # Axis-aligned probes (helps if function is separable like Sphere)
        for axis in range(n):
            for val in [-3.0, 3.0, -1.0, 1.0]:
                pt = np.zeros(n)
                pt[axis] = val
                init_points.append(pt)

        # Evaluate initial points
        points = []
        values = []
        for pt in init_points[:n_init]:
            if not check_time() or eval_count >= budget:
                break
            pt = clip_box(pt)
            fv = func(pt)
            try_update(pt, fv)
            points.append(pt)
            values.append(fv)

        if best_x is None:
            # Fallback if something went wrong
            best_x = np.zeros(n)
            best_f = func(best_x)
            try_update(best_x, best_f)
            points.append(best_x)
            values.append(best_f)

        # Fit quadratic surrogate to guide next steps
        X = np.array(points)
        y = np.array(values)

        # Build design matrix for quadratic model
        n_features = 1 + n + n * (n + 1) // 2
        design = np.ones((len(X), n_features))
        for i in range(len(X)):
            x = X[i]
            col = 1
            for d in range(n):
                design[i, col] = x[d]
                col += 1
            for d1 in range(n):
                for d2 in range(d1, n):
                    if d1 == d2:
                        design[i, col] = x[d1] * x[d2]
                    else:
                        design[i, col] = x[d1] * x[d2]
                    col += 1

        # Least squares fit
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        except:
            coeffs = np.zeros(n_features)

        # Extract quadratic form
        a0 = coeffs[0]
        b = coeffs[1:1 + n].copy()
        C_flat = coeffs[1 + n:].copy()

        # Reconstruct C matrix
        C = np.zeros((n, n))
        idx = 0
        for d1 in range(n):
            for d2 in range(d1, n):
                if d1 == d2:
                    C[d1, d2] = C_flat[idx]
                else:
                    C[d1, d2] = C_flat[idx] / 2
                    C[d2, d1] = C_flat[idx] / 2
                idx += 1

        # Ensure C is symmetric
        C = (C + C.T) / 2

        # Find minimum of surrogate
        try:
            if np.linalg.cond(C) < 1e6:
                x_min_surrogate = -np.linalg.solve(C, b / 2)
            else:
                x_min_surrogate = -np.linalg.pinv(C) @ (b / 2)
            x_min_surrogate = np.clip(x_min_surrogate, -4.0, 4.0)
        except:
            x_min_surrogate = best_x.copy()

        # Evaluate the surrogate minimum
        if check_time() and eval_count < budget:
            x_test = clip_box(x_min_surrogate)
            f_test = func(x_test)
            try_update(x_test, f_test)
            if f_test < best_f:
                best_x = x_test.copy()

        # ===== Phase 2: Coordinate Line Search from Best =====
        # Exploit the separable structure of BBOB functions
        if check_time() and eval_count < budget:
            offsets = [2.0, -2.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25]
            
            for coord in range(n):
                if not check_time() or eval_count >= budget:
                    break
                
                for off in offsets:
                    if not check_time() or eval_count >= budget:
                        break
                    
                    new_val = np.clip(best_x[coord] + off, -5.0, 5.0)
                    x_try = best_x.copy()
                    x_try[coord] = new_val
                    
                    f_try = func(x_try)
                    try_update(x_try, f_try)

        # ===== Phase 3: CMA-ES Refinement =====
        # Use CMA-ES for robust refinement of the solution
        
        x_mean = best_x.copy()
        f_best = best_f
        sigma = 0.5
        
        # CMA-ES parameters
        lambda_ = min(2 * n + 1, 15)
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (n + 4)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 2) - 1) + c_sigma
        
        # Covariance matrix
        C_mat = np.eye(n)
        C_sqrt_inv = np.eye(n)
        
        # Evolution path
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Trust region radius
        trust_radius = 1.0
        no_improve = 0
        max_no_improve = 40
        
        while check_time() and eval_count < budget:
            n_offspring = min(lambda_, budget - eval_count)
            if n_offspring <= 0:
                break
            
            # Generate offspring
            offspring = []
            for _ in range(n_offspring):
                if not check_time() or eval_count >= budget:
                    break
                
                # Generate random direction
                z = rng.standard_normal(n)
                # Apply covariance transform (simplified)
                y = x_mean + sigma * z
                
                # Clip to trust region
                y_rel = y - x_mean
                norm_y = np.linalg.norm(y_rel)
                if norm_y > trust_radius:
                    y = x_mean + y_rel * (trust_radius / norm_y)
                
                y = clip_box(y)
                f_y = func(y)
                try_update(y, f_y)
                offspring.append((y, f_y))
            
            if not offspring:
                break
            
            # Sort offspring
            offspring.sort(key=lambda t: t[1])
            
            # Update mean
            x_new_mean = np.zeros(n)
            for i in range(mu):
                x_new_mean += weights[i] * offspring[i][0]
            
            # Update paths
            sigma_old = sigma
            y_mean = (x_new_mean - x_mean) / sigma_old if sigma_old > 1e-12 else np.zeros(n)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            
            # Update sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            norm_p_sigma_h = np.sqrt(n) * (1 - c_sigma) ** (-0.5)
            sigma = sigma * np.exp(c_sigma * (norm_p_sigma / norm_p_sigma_h - 1))
            sigma = np.clip(sigma, 1e-8, 5.0)
            
            # Update covariance
            for i in range(mu):
                y_i = (offspring[i][0] - x_mean) / sigma_old if sigma_old > 1e-12 else np.zeros(n)
                C_mat = (1 + c_c * (2 - c_c) - c_sigma) * C_mat + c_sigma * (y_i * y_i.T - np.eye(n))
            
            # Ensure C_mat is positive definite
            C_mat = (C_mat + C_mat.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt_inv = np.linalg.inv(eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T)
            except:
                C_mat = np.eye(n)
                C_sqrt_inv = np.eye(n)
            
            # Update mean
            x_mean = x_new_mean
            
            # Check for improvement
            if offspring[0][1] < f_best - 1e-10:
                f_best = offspring[0][1]
                best_x = offspring[0][0].copy()
                no_improve = 0
                trust_radius = min(trust_radius * 1.2, 5.0)
            else:
                no_improve += 1
                trust_radius = max(trust_radius * 0.9, 0.1)
            
            # Diversity injection on stagnation
            if no_improve > 20 and eval_count < budget - 10:
                if not check_time():
                    break
                x_inject = clip_box(rng.uniform(-4.0, 4.0, n))
                f_inject = func(x_inject)
                try_update(x_inject, f_inject)
                if f_inject < offspring[0][1]:
                    x_mean = x_inject
                    f_best = f_inject
                    best_x = x_inject.copy()
                    no_improve = 0
                    sigma = 1.0
                    trust_radius = 2.0
                else:
                    no_improve = 0
                    sigma = max(sigma * 1.5, 0.5)

        # ===== Phase 4: Fine-tuning =====
        # If we have budget left, do a final (1+1)-ES refinement with small sigma
        if check_time() and eval_count < budget:
            sigma_fine = max(sigma * 0.1, 0.01)
            x_fine = best_x.copy()
            f_fine = best_f
            no_improve_fine = 0
            max_fine_steps = min(20, budget - eval_count)
            
            for _ in range(max_fine_steps):
                if not check_time() or eval_count >= budget:
                    break
                y = clip_box(x_fine + sigma_fine * rng.standard_normal(n))
                fy = func(y)
                try_update(y, fy)
                
                if fy < f_fine - 1e-12:
                    x_fine = y
                    f_fine = fy
                    no_improve_fine = 0
                    sigma_fine = min(sigma_fine * 1.05, 0.5)
                else:
                    sigma_fine *= 0.9
                    no_improve_fine += 1
                
                if sigma_fine < 1e-6 or no_improve_fine > 10:
                    break
            
            if f_fine < best_f:
                best_x = x_fine
                best_f = f_fine

        return best_x, best_f
# EVOLVE-BLOCK-END
