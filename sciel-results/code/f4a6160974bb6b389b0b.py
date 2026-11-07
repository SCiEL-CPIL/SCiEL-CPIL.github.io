"""Optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer using a multi-start local refinement strategy.
    
    1. Global exploration with random sampling to identify promising regions.
    2. Local refinement using L-BFGS-B from the best initial points.
    3. A short CMA-ES phase for final polishing if budget remains.
    
    This approach leverages gradient-free local solvers which are robust to
    the smooth nature of BBOB functions, combined with global sampling to
    handle the hidden shift/rotation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 300s limit, stop at 290s
        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = float('inf')
        calls = 0

        def check_time():
            return time.time() > deadline

        # --- Phase 1: Global Exploration ---
        # Sample a diverse set of points to find the basin of attraction.
        # For 5D, 30-50 points is reasonable. We use 40.
        n_global = 40
        global_points = rng.uniform(-5, 5, (n_global, n))
        
        # Add center point explicitly
        center = np.zeros(n)
        global_points = np.vstack([center, global_points])
        
        for i in range(len(global_points)):
            if calls >= total_budget or check_time():
                break
            x_i = global_points[i]
            fx = func(x_i)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_i.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1

        # --- Phase 2: Local Refinement with L-BFGS-B ---
        # Use the best points from global search to start local optimizations.
        # We'll keep track of the top K points and run L-BFGS-B from each.
        # Since we don't store all points, let's re-sample or just use the best one?
        # Better: Store the top few points during global phase.
        
        # Let's restart the logic slightly to store top points.
        # Actually, let's just use the best point found so far and a few random restarts around it.
        
        # A more robust strategy: 
        # 1. Find a good global point.
        # 2. Run L-BFGS-B from that point.
        # 3. If budget remains, run L-BFGS-B from other promising points or random restarts.
        
        # Let's collect the top 5 points from the global phase.
        # We need to modify Phase 1 to store them.
        
        # Let's restart Phase 1 properly to store points.
        points_evaluated = []
        best_x = None
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(12345)
        
        n_global = 40
        global_points = rng.uniform(-5, 5, (n_global, n))
        center = np.zeros(n)
        global_points = np.vstack([center, global_points])
        
        for i in range(len(global_points)):
            if calls >= total_budget or check_time():
                break
            x_i = global_points[i]
            fx = func(x_i)
            calls += 1
            points_evaluated.append((fx, x_i.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x_i.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1
            points_evaluated.append((best_f, best_x.copy()))

        # Sort by function value
        points_evaluated.sort(key=lambda x: x[0])
        
        # Take top 3 unique-ish points for local refinement
        top_points = [p[1] for p in points_evaluated[:3]]
        
        # If we have very little budget left, skip local refinement
        remaining_budget = total_budget - calls
        if remaining_budget < 20 or check_time():
            return best_x, best_f

        # Run L-BFGS-B from each top point
        # L-BFGS-B is a local solver, so it will converge to a local minimum.
        # We need to ensure we don't exceed the budget.
        # Each L-BFGS-B call will use a subset of the remaining budget.
        
        # We'll allocate remaining_budget / len(top_points) to each.
        # But L-BFGS-B doesn't have a strict "budget" in terms of function calls in the same way.
        # We can wrap func to count calls and stop if we exceed the allocated budget.
        
        for start_x in top_points:
            if calls >= total_budget or check_time():
                break
            
            # Allocate budget for this local search
            # Let's give each local search a portion of the remaining budget
            # e.g., remaining_budget // len(top_points)
            local_budget = max(10, remaining_budget // len(top_points))
            
            # Create a wrapper function that counts calls and stops if budget exceeded
            local_calls = 0
            def local_func(x):
                nonlocal local_calls, calls, best_f, best_x
                if local_calls >= local_budget or calls >= total_budget or check_time():
                    return float('inf')
                fx = func(x)
                local_calls += 1
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            
            # Use L-BFGS-B
            try:
                res = minimize(local_func, start_x, method='L-BFGS-B', 
                               bounds=[(-5, 5)] * n, 
                               options={'maxiter': 100, 'maxfun': local_budget})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

            if check_time() or calls >= total_budget:
                break

        # --- Phase 3: Final Polish with CMA-ES (Optional) ---
        # If we still have budget, we can run a short CMA-ES from the best point.
        remaining_budget = total_budget - calls
        if remaining_budget > 20 and not check_time():
            # Simple CMA-ES implementation
            x = best_x.copy()
            sigma = 0.5  # Smaller sigma for polishing
            
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            gen = 0
            while calls < total_budget and not check_time():
                gen += 1
                
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                
                Y = np.empty((lam, n))
                for i in range(lam):
                    z = rng.standard_normal(n)
                    Y[i] = x + sigma * (sqrt_C @ z)
                    Y[i] = np.clip(Y[i], -5, 5)
                
                F = np.empty(lam)
                valid_count = 0
                for i in range(lam):
                    if calls >= total_budget or check_time():
                        break
                    F[i] = func(Y[i])
                    calls += 1
                    valid_count += 1
                    if F[i] < best_f:
                        best_f = F[i]
                        best_x = Y[i].copy()
                
                if valid_count < 2:
                    break
                
                if valid_count < lam:
                    F[valid_count:] = np.inf
                
                idx = np.argsort(F)
                Y_sorted = Y[idx]
                
                if valid_count < mu:
                    break
                
                Y_best = Y_sorted[:mu]
                x_new = np.dot(w, Y_best)
                
                y_invC = (Y_best - x) / sigma
                weighted_sum = np.dot(w, y_invC)
                
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ weighted_sum)
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
                
                delta_h = (1 - np.exp(-2 + 5 / (n + 1)))
                if gen > 0:
                    denom = np.sqrt(1 - (1 - c_sigma)**(gen / d_sigma))
                else:
                    denom = 1.0
                
                if denom > 1e-12:
                    norm_ps = np.linalg.norm(ps)
                    c_n = 1 - 1.0/(2*n) + 1.0/(12*n**2)
                    threshold = (1.4 + 2.0/(n + 1)) * c_n * np.sqrt(np.trace(C))
                    if norm_ps / denom < threshold:
                        h_pc = delta_h
                    else:
                        h_pc = 0.0
                else:
                    h_pc = 0.0
                
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) + (1 - h_pc) * c_c * (2 - c_c) * C)
                
                for i in range(mu):
                    y_i = y_invC[i]
                    C += c_mu * w[i] * np.outer(y_i, y_i)
                
                C = (C + C.T) / 2
                x = x_new
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(n) - 1))
                x = np.clip(x, -10, 10)
                
                if sigma < 1e-6:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
