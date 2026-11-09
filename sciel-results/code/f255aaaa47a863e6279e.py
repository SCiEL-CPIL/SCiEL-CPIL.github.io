"""Optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Robust CMA-ES optimizer for BBOB f3."""

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

        # Phase 1: Initial random sampling to avoid getting stuck in a bad local minimum
        # Rastrigin is multimodal. We sample a few points to get a decent starting point.
        n_init = 10
        for _ in range(n_init):
            if call_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            val = call_func(x)
            if val is None:
                break
        
        # If we haven't found anything reasonable, try origin
        if call_count < budget and time.time() < deadline:
             call_func(np.zeros(dim))

        # Phase 2: Run CMA-ES
        # CMA-ES is robust to rotations and shifts.
        # We initialize the distribution around the current best point.
        
        x_mean = best_x.copy()
        sigma = 1.5  # Initial step size, large enough to explore
        
        lambda_ = 12
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
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
        max_stagnation = 20 # Allow more generations before stopping
        
        while call_count < budget and time.time() < deadline:
            # Update C_inv_sqrt periodically to avoid numerical issues and cost
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
                # Clip to box
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
                # Restart logic: perturb the mean and increase sigma slightly?
                # For now, just break as CMA-ES should have converged or failed.
                # Given the budget, restarting might be risky if we are close to the limit.
                # However, if we are stuck early, we might want to try a different spot.
                # Let's try a simple restart: move mean to a random point near current best?
                # Actually, standard CMA-ES termination is usually based on sigma or spread.
                # We'll just break to be safe and use remaining budget for local polish if needed.
                break

            # Update mean
            m = sum(w * xk for w, xk in zip(weights, selected_x))
            
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

        # Phase 3: Local refinement (Coordinate Descent)
        # CMA-ES might stop early if it thinks it converged, but Rastrigin has flat bottoms.
        # A few steps of local search can squeeze out the last bits of precision.
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            # Use smaller steps
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
