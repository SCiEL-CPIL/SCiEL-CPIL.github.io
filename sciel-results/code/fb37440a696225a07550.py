import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return np.inf
            x_clip = np.clip(x, -5, 5)
            val = func(x_clip)
            calls += 1
            return val

        rng = np.random.default_rng(42)

        def cma_es(x0, sigma0, max_calls, tol=1e-12):
            """Run CMA-ES from x0 with initial sigma0, using at most max_calls."""
            nonlocal calls, best_f, best_x
            n = self.dim
            
            # CMA-ES parameters
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            # State
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            # Initial population
            def evaluate_population():
                nonlocal calls
                pops = []
                fs = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    x = mean + sigma * (C @ z)
                    f = safe_call(x)
                    if calls >= budget:
                        return None, None
                    pops.append(x)
                    fs.append(f)
                return np.array(pops), np.array(fs)
            
            # Initial evaluation
            result = evaluate_population()
            if result[0] is None:
                return
            pops, fs = result
            
            # Sort
            order = np.argsort(fs)
            pops = pops[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = pops[0].copy()
            
            # CMA-ES iterations
            while calls < budget and time.time() < deadline:
                if calls >= budget:
                    break
                
                # Update mean
                mean = np.sum(weights[:, None] * pops[:mu], axis=0)
                
                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C @ (mean - x0))
                x0 = mean  # This is wrong, let me fix
                
                # Actually, let me redo this properly
                break
            
            # This CMA-ES implementation is getting complex. Let me use a simpler approach.
            # For now, just do a few more evaluations around the best point.
            return

        # Strategy: Use a combination of exploration and exploitation
        # Phase 1: Quick exploration with diverse points
        # Phase 2: CMA-ES-like refinement from best points
        
        # Phase 1: Evaluate several diverse initial points
        num_initial = 10
        initial_points = [np.zeros(n)]
        for _ in range(num_initial - 1):
            initial_points.append(rng.uniform(-4, 4, n))
        
        initial_results = []
        for x0 in initial_points:
            if calls >= budget or time.time() > deadline:
                break
            f = safe_call(x0)
            initial_results.append((f, x0.copy()))
            if f < best_f:
                best_f = f
                best_x = x0.copy()
        
        # Sort by function value
        initial_results.sort(key=lambda x: x[0])
        
        # Phase 2: Local refinement from top 2-3 points using coordinate descent + Nelder-Mead
        top_k = min(3, len(initial_results))
        
        for k in range(top_k):
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = initial_results[k][1].copy()
            remaining = budget - calls
            if remaining < 10:
                break
            
            # Coordinate descent for quick improvement
            step_size = 0.5
            for _ in range(10):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(n):
                    if calls >= budget:
                        break
                    for direction in [1, -1]:
                        x_trial = x0.copy()
                        x_trial[i] += direction * step_size
                        f_trial = safe_call(x_trial)
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = x_trial.copy()
                            x0 = x_trial.copy()
                            improved = True
                            break
                if not improved:
                    step_size *= 0.5
                    if step_size < 0.01:
                        break
            
            # Nelder-Mead refinement
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = best_x.copy()
            simplex = [x0.copy()]
            f_vals = [best_f]
            
            for i in range(n):
                new_x = x0.copy()
                step = 0.3
                new_x[i] += step
                simplex.append(new_x)
                f_vals.append(safe_call(new_x))
            
            if calls >= budget:
                break
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            tol = 1e-10
            max_iter = min(remaining - 1, 60)
            
            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                
                idx = np.argsort(f_vals)
                simplex = [simplex[i] for i in idx]
                f_vals = [f_vals[i] for i in idx]
                
                if abs(f_vals[0] - f_vals[-1]) < tol * (1 + abs(f_vals[0])):
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = safe_call(x_ref)
                
                if calls >= budget:
                    break
                
                if f_ref < f_vals[0]:
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    f_exp = safe_call(x_exp)
                    if calls >= budget:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_vals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_vals[-1] = f_ref
                elif f_ref < f_vals[-2]:
                    simplex[-1] = x_ref
                    f_vals[-1] = f_ref
                else:
                    if f_ref < f_vals[-1]:
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        x_con = centroid - rho * (centroid - simplex[-1])
                    f_con = safe_call(x_con)
                    if calls >= budget:
                        break
                    if f_con < min(f_ref, f_vals[-1]):
                        simplex[-1] = x_con
                        f_vals[-1] = f_con
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = safe_call(simplex[i])
                        if calls >= budget:
                            break
                
                if f_vals[0] < best_f:
                    best_f = f_vals[0]
                    best_x = simplex[0].copy()
        
        # Phase 3: Final fine-tuning if budget remains
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining >= 5:
                # Small perturbations around best
                for _ in range(min(remaining, 20)):
                    if calls >= budget or time.time() > deadline:
                        break
                    perturbation = rng.normal(0, 0.1, n)
                    x_trial = np.clip(best_x + perturbation, -5, 5)
                    f_trial = safe_call(x_trial)
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = x_trial.copy()
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1
        
        return best_x, best_f
