import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # --- Phase 1: Structured Exploration ---
        # Quickly locate the basin of attraction using a structured set of points.
        
        candidates = []
        
        # Origin
        candidates.append(np.zeros(dim))
        
        # Axis-aligned points
        for d in range(dim):
            for dist in [1.0, 2.0, 3.0, 4.0]:
                for sign in [1, -1]:
                    p = np.zeros(dim)
                    p[d] = sign * dist
                    candidates.append(p)
        
        # 2D plane points
        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                for dist in [1.5, 2.5]:
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            p = np.zeros(dim)
                            p[d1] = s1 * dist
                            p[d2] = s2 * dist
                            candidates.append(p)
        
        # Random points
        for _ in range(10):
            candidates.append(rng.uniform(-4.5, 4.5, dim))
        
        # Shuffle to avoid bias
        rng.shuffle(candidates)
        
        # Limit exploration budget to ~15% of total
        explore_budget = min(75, budget // 6)
        
        best_x = None
        best_fx = np.inf
        calls = 0
        
        for x_cand in candidates:
            if calls >= explore_budget:
                break
            if time.time() > deadline:
                break
                
            x_cand = np.clip(x_cand, -5, 5)
            fx = func(x_cand)
            calls += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x_cand.copy()
        
        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls += 1
            
        # --- Phase 2: CMA-ES Refinement ---
        # Use CMA-ES to adapt to the rotation and shift of the function.
        
        # CMA-ES Parameters
        lambda_pop = int(4 + 3 * np.log(dim))
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Initialize CMA-ES state
        x_mean = best_x.copy()
        sigma = 1.0  # Start with a moderate step size
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        def eval_func(x):
            nonlocal calls, best_x, best_fx
            if calls >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_fx:
                best_fx = f
                best_x = x.copy()
            return f
        
        # Main CMA-ES Loop
        while calls + lambda_pop <= budget and time.time() < deadline:
            # Sample population
            Y = np.zeros((lambda_pop, dim))
            X = np.zeros((lambda_pop, dim))
            F = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                x_new = x_mean + sigma * y
                
                # Boundary handling: Reflection
                for j in range(dim):
                    if x_new[j] < -5:
                        x_new[j] = -5 + (-5 - x_new[j]) * 0.5
                        x_new[j] = np.clip(x_new[j], -5, 5)
                    elif x_new[j] > 5:
                        x_new[j] = 5 - (x_new[j] - 5) * 0.5
                        x_new[j] = np.clip(x_new[j], -5, 5)
                
                Y[i] = z
                X[i] = x_new
                F[i] = eval_func(x_new)
                
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            F = F[idx]
            X = X[idx]
            Y = Y[idx]
            
            # Update mean
            x_new_mean = np.sum(weights[:, np.newaxis] * X[:mu], axis=0)
            
            # Update p_sigma
            y_bar = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * y_bar))
            
            # Calculate h_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if norm_p_sigma < (1.4 + 1.0/dim) * sigma * chi_n else 0.0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma
            
            # Update Covariance Matrix C
            v = (x_new_mean - x_mean) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            # Rank-mu update
            for i in range(mu):
                v_i = (X[i] - x_mean) / sigma
                C += c_mu * weights[i] * (v_i @ v_i)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                # Update B and D via eigendecomposition
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))
            except np.linalg.LinAlgError:
                # Fallback if eigendecomposition fails
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            if sigma > 10.0:
                sigma = 10.0
            elif sigma < 1e-8:
                sigma = 1e-8
            
            x_mean = x_new_mean
            
            # Check for stagnation
            if np.linalg.norm(x_new_mean - x_mean) < 1e-12 and sigma < 1e-4:
                break

        # --- Phase 3: Coordinate Ascent Polish ---
        # Use remaining budget for precise local refinement.
        
        remaining_budget = budget - calls
        if remaining_budget > 5 and time.time() < deadline - 2.0:
            polish_budget = min(remaining_budget, 50)
            step = max(sigma * 0.5, 0.01)
            
            for _ in range(polish_budget):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                    
                improved = False
                for d in range(dim):
                    if calls >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Try moving in positive direction
                    x_trial = best_x.copy()
                    x_trial[d] += step
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls += 1
                    
                    if f_trial < best_fx:
                        best_x = x_trial
                        best_fx = f_trial
                        improved = True
                    else:
                        # Try moving in negative direction
                        x_trial = best_x.copy()
                        x_trial[d] -= step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        calls += 1
                        
                        if f_trial < best_fx:
                            best_x = x_trial
                            best_fx = f_trial
                            improved = True
                    
                    if improved:
                        step *= 0.8
                        break
                else:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_fx
