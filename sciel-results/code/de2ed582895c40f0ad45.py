import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        n_calls = 0
        best_x = None
        best_fx = np.inf

        def evaluate(x):
            nonlocal best_x, best_fx, n_calls
            x = np.clip(x, -5, 5)
            fx = func(x)
            n_calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Search using Random Sampling + Local Polish
        # Strategy: Use a significant portion of the budget (e.g., 30%) to sample
        # random points. For each point, run a very cheap local search (Nelder-Mead
        # with low maxfev) to see if we can descend into a basin. This helps escape
        # the "step" plateaus where gradient-based methods or large steps fail.
        
        # We implement a custom lightweight Nelder-Mead to control evaluation count precisely
        # and avoid overhead from scipy's minimize if it's too slow, though scipy is fine.
        # Let's use scipy.optimize.minimize with Nelder-Mead but tightly controlled.
        
        # To be safe and fast, let's do a mix:
        # 1. Sample K points.
        # 2. For the best few, do a local refinement.
        
        # Actually, the "Step Ellipsoidal" function is non-differentiable.
        # A robust approach is:
        # 1. Coarse grid or random sampling to locate the basin.
        # 2. CMA-ES or Nelder-Mead to refine.
        
        # Given the budget is only 500, we can afford:
        # - 200 evaluations for global exploration (random restarts of NM or just random sampling)
        # - 300 evaluations for the final CMA-ES or NM run from the best point.
        
        # Let's try a simpler, robust strategy:
        # 1. Generate 50 random points.
        # 2. Evaluate them.
        # 3. Take the best 5.
        # 4. Run Nelder-Mead from each of the best 5, with a budget of ~20 evals each.
        # 5. Take the best result from those.
        # 6. Run a final CMA-ES or NM from that best result with remaining budget.
        
        # However, Nelder-Mead can be tricky with non-smooth functions.
        # Let's stick to a modified CMA-ES but with a stronger global phase.
        
        # Phase 1: Random Sampling
        num_samples = 50
        samples = np.random.uniform(-5, 5, (num_samples, dim))
        f_vals = np.array([evaluate(x) for x in samples])
        
        # Sort samples by fitness
        sorted_idx = np.argsort(f_vals)
        
        # Phase 2: Local Refinement of top candidates
        # Pick top 3 candidates
        top_k = 3
        local_budget_per_candidate = 20
        local_starts = samples[sorted_idx[:top_k]]
        
        # We will run a simple Nelder-Mead-like local search or just a coordinate descent
        # to avoid scipy overhead if possible, but scipy is available.
        # Let's use scipy.optimize.minimize with Nelder-Mead for the local phase.
        
        from scipy.optimize import minimize
        
        best_local_x = best_x.copy()
        best_local_fx = best_fx
        
        for i in range(top_k):
            if time.time() > deadline - 30:
                break
            if n_calls >= budget - 100: # Reserve budget for final phase
                break
                
            x0 = local_starts[i]
            # Use Nelder-Mead with limited iterations
            try:
                res = minimize(func, x0, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4, 'maxfev': local_budget_per_candidate})
                if res.fun < best_local_fx:
                    best_local_fx = res.fun
                    best_local_x = res.x
            except Exception:
                pass
        
        # Update global best
        if best_local_fx < best_fx:
            best_fx = best_local_fx
            best_x = best_local_x.copy()
        else:
            # Ensure best_x is updated to the best we've seen
            # best_x and best_fx are already updated in evaluate()
            pass

        # Phase 3: Final CMA-ES run from the best point
        # Use remaining budget
        remaining_budget = budget - n_calls
        
        if remaining_budget < 10:
            return best_x, best_fx
        
        # Initialize CMA-ES around best_x
        x_mean = best_x.copy()
        sigma = 0.5 # Start with a smaller sigma since we are in a basin
        
        # CMA-ES Parameters
        lam = 4
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 1) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        dsigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c1
        
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        DC = np.linalg.cholesky(C)
        
        chi_N = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        while n_calls < budget:
            if time.time() > deadline - 5:
                break
            
            # Generate population
            Z = np.random.randn(lam, dim)
            Y = Z @ DC.T
            
            F = np.empty(lam)
            for i in range(lam):
                if n_calls >= budget:
                    break
                xi = x_mean + sigma * Y[i]
                F[i] = evaluate(xi)
            
            if n_calls >= budget:
                break
                
            indices = np.argsort(F)
            Z_sorted = Z[indices]
            Y_sorted = Y[indices]
            
            z_bar = np.dot(weights, Z_sorted[:mu])
            mean_step = DC @ z_bar
            x_mean = x_mean + sigma * mean_step
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * z_bar
            
            sigma = sigma * np.exp(cs / dsigma * (np.linalg.norm(ps) / chi_N - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * mean_step
            
            Y_sel = Y_sorted[:mu]
            M = np.zeros((dim, dim))
            for i in range(mu):
                M += weights[i] * np.outer(Y_sel[i], Y_sel[i])
                
            h_sig = 1.0 if np.linalg.norm(ps) / chi_N < (1.4 + 2/(dim+1)) else 0.0
            
            C = (1 - c1 - cmu * h_sig) * C + c1 * np.outer(pc, pc) + cmu * M
            C = (C + C.T) / 2
            
            try:
                DC = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 0.01 * np.eye(dim)
                try:
                    DC = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    DC = np.eye(dim)
                    C = np.eye(dim)

        return best_x, best_fx
