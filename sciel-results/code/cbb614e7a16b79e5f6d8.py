"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Model-guided CMA-ES optimizer for BBOB f3."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def call_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            val = func(x)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def rastrigin_model(x, center, offset):
            """Evaluate Rastrigin shifted by center with additive offset."""
            x_shifted = x - center
            # Vectorized Rastrigin
            return offset + np.sum(10 + x_shifted**2 - 10 * np.cos(2 * np.pi * x_shifted))

        # Phase 1: Initial sampling to fit the model
        # We use a mix of structured and random points to cover the space
        n_probe = 40
        probe_points = []
        
        # Structured points: Origin, and +/- 1, +/- 2 in each dimension
        probe_points.append(np.zeros(dim))
        for i in range(dim):
            for sign in [1, -1]:
                for dist in [1.0, 2.0]:
                    p = np.zeros(dim)
                    p[i] = sign * dist
                    probe_points.append(p)
        
        # Fill rest with random points
        while len(probe_points) < n_probe:
            probe_points.append(rng.uniform(-4.5, 4.5, dim))
        
        probe_vals = []
        for p in probe_points:
            if call_count >= budget or time.time() > deadline:
                break
            val = call_func(np.clip(p, -5, 5))
            if val is None:
                break
            probe_vals.append(val)
        
        if len(probe_vals) < 10:
            # Fallback: just random search if we can't fit
            while call_count < budget and time.time() < deadline:
                x = rng.uniform(-5, 5, dim)
                call_func(x)
            return best_x, best_f
        
        probe_points = np.array(probe_points[:len(probe_vals)])
        probe_vals = np.array(probe_vals)
        
        # Phase 2: Fit the model parameters (center and offset)
        # Objective: minimize sum of squared residuals
        def model_residual(params):
            center = params[:dim]
            offset = params[dim]
            # Clip center to reasonable range to prevent divergence
            center = np.clip(center, -5, 5)
            model_vals = np.array([rastrigin_model(probe_points[i], center, offset) for i in range(len(probe_vals))])
            return np.sum((model_vals - probe_vals)**2)

        # Initial guess: weighted average of probe points by inverse fitness
        weights = 1.0 / (probe_vals - probe_vals.min() + 1e-10)
        weights /= weights.sum()
        center_est = np.sum(weights[:, None] * probe_points, axis=0)
        offset_est = probe_vals.min() - 10 * dim  # Rough offset
        
        best_model_params = np.concatenate([center_est, [offset_est]])
        best_model_obj = model_residual(best_model_params)
        
        # Refine with Nelder-Mead
        for init_try in range(3):
            center_init = center_est + rng.normal(0, 1.5, dim)
            offset_init = offset_est + rng.normal(0, 10.0)
            params_init = np.concatenate([center_init, [offset_init]])
            try:
                result = minimize(model_residual, params_init, method='Nelder-Mead',
                                 options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
                if result.fun < best_model_obj:
                    best_model_obj = result.fun
                    best_model_params = result.x
            except:
                pass
        
        center_fit = np.clip(best_model_params[:dim], -5, 5)
        offset_fit = best_model_params[dim]
        
        # Phase 3: Generate candidates based on the fitted model
        # The global minimum of Rastrigin is at x = center + k, where k is integer vector
        # We test center_fit and its immediate neighbors (integer shifts)
        candidates = [center_fit.copy()]
        
        # Add neighbors: center_fit +/- 1 in each dimension
        for d in range(dim):
            for sign in [1, -1]:
                c = center_fit.copy()
                c[d] += sign * 1.0
                candidates.append(c)
        
        # Also include the best point found so far from Phase 1
        candidates.append(best_x.copy())
        
        # Evaluate candidates
        for c in candidates:
            if call_count >= budget or time.time() > deadline:
                break
            c_clipped = np.clip(c, -5, 5)
            val = call_func(c_clipped)
            if val is None:
                break
        
        # Phase 4: Run CMA-ES from the best point
        # This ensures we are in the correct basin
        x_mean = best_x.copy()
        sigma = 1.0  # Start with moderate step size
        
        lambda_ = 12
        mu = lambda_ // 2
        
        # Weights
        weights_cma = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights_cma /= weights_cma.sum()
        mueff = 1.0 / np.sum(weights_cma**2)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # State variables
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        gen = 0
        stagnation = 0
        max_stagnation = 20
        
        while call_count < budget and time.time() < deadline:
            # Update C_inv_sqrt periodically
            if gen % 10 == 0:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
                except:
                    C_inv_sqrt = np.eye(dim)
            
            offspring = []
            off_f = []
            
            for k in range(lambda_):
                if call_count >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
                fy = call_func(y)
                if fy is None:
                    break
                off_f.append(fy)
            
            if len(off_f) < mu:
                break
            
            # Sort and select
            sorted_idx = np.argsort(off_f)
            selected_x = [offspring[i] for i in sorted_idx[:mu]]
            selected_f = [off_f[i] for i in sorted_idx[:mu]]
            
            # Check stagnation
            if selected_f[0] < best_f - 1e-10:
                stagnation = 0
            else:
                stagnation += 1
            
            if stagnation > max_stagnation:
                break

            # Update mean
            m = sum(w * xk for w, xk in zip(weights_cma, selected_x))
            
            # Update path memories
            diff = m - x_mean
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ diff)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * diff
            
            # Update sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1)))) / chi_n
            if h_sigma < (1.4 + 2 / (dim + 1)):
                sigma *= np.exp(c_sigma / chi_n * (h_sigma - 1))
            
            # Update covariance matrix
            delta_d = (1 + c_mu * (mueff - 1)) * diff
            rank_mu = np.outer(delta_d, delta_d)
            p_c_norm_sq = np.linalg.norm(p_c)**2
            h_c = 1.0 if p_c_norm_sq > 1 / (1 - c_c)**2 else 0.0
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + h_c * np.eye(dim)) + c_mu * rank_mu
            
            x_mean = m
            gen += 1
            
            # Termination criterion based on spread
            if sigma * np.max(np.sqrt(np.diag(C))) < 1e-6:
                break

        # Phase 5: Local refinement (Coordinate Descent)
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            for step in [0.1, 0.05, 0.02, 0.01]:
                if call_count >= budget or time.time() > deadline:
                    break
                improved = True
                while improved and call_count < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if call_count >= budget or time.time() > deadline:
                            break
                        for direction in [1, -1]:
                            x_new = x.copy()
                            x_new[d] += direction * step
                            x_new = np.clip(x_new, -5, 5)
                            fx_new = call_func(x_new)
                            if fx_new is None:
                                break
                            if fx_new < fx - 1e-12:
                                x = x_new
                                fx = fx_new
                                improved = True
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
