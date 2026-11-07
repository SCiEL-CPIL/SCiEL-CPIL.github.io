import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lambda_ = 4
        mu = lambda_ // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (2 + np.log(n / np.log(n + 1))) / (4 * n + 4)
        c_1 = 2 / ((n + 1.3)**2 + lambda_)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + lambda_))
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Helper to run CMA-ES for a certain number of evals
        def run_cmaes(x0, sigma0, max_evals):
            nonlocal calls, best_x, best_f
            
            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            h_sigma = (1 + 0.3 * (n + 1)) * np.sqrt((mu_eff + 2) / (n + 2)) # Approximation
            
            # Pre-compute Hessian update parts? No, just use standard update.
            # For small n, np.linalg.inv is fine.
            
            last_improvement = 0
            local_best_f = np.inf
            
            while calls < budget and calls < max_evals:
                # Generate lambda samples
                xs = []
                fs = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    y = x_mean + sigma * (B @ (D * z))
                    # Clip to box? The problem says func clips, but we should stay in bounds for better CMA behavior?
                    # "func(x) clips a finite length-5 vector to the box"
                    # It is better to let CMA-ES work in the unbounded space and let func clip,
                    # or clip x_mean and y to stay inside.
                    # Clipping y is safer for CMA-ES stability near boundaries.
                    y = np.clip(y, -5, 5)
                    xs.append(y)
                    f_y = func(y)
                    calls += 1
                    fs.append(f_y)
                    
                # Sort by fitness
                order = np.argsort(fs)
                xs_sorted = [xs[i] for i in order]
                fs_sorted = [fs[i] for i in order]
                
                # Update best
                if fs_sorted[0] < local_best_f:
                    local_best_f = fs_sorted[0]
                    if fs_sorted[0] < best_f:
                        best_f = fs_sorted[0]
                        best_x = xs_sorted[0].copy()
                        last_improvement = calls
                
                # Update mean
                x_new_mean = np.sum([w[i] * xs_sorted[i] for i in range(mu)], axis=0)
                
                # Calculate step
                step = x_new_mean - x_mean
                
                # Update covariance matrix
                # C = B D^2 B^T
                # Update C
                # Standard CMA-ES update:
                # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - chi_n) * delta(h_sigma) * C) + c_mu * sum(w_i (y_i - y_bar) (y_i - y_bar)^T)
                
                # For efficiency and simplicity with small n, we can use eigenvalue decomposition
                # or just maintain B and D.
                # Let's use the B, D representation update.
                
                # Calculate y_bar (weighted mean of y)
                y_bar = np.sum([w[i] * (xs_sorted[i] - x_mean) for i in range(mu)], axis=0)
                
                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_bar
                
                # h_sigma check
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - last_improvement))) < (1.4 + 2 / (n + 1)) * h_sigma:
                    h_sigma = 1
                else:
                    h_sigma = 0
                
                # Update p_c
                p_c = (1 - c_1) * p_c + h_sigma * np.sqrt(c_1 * (2 - c_1) * mu_eff) * step / sigma
                
                # Update B and D
                # This is complex to implement from scratch correctly.
                # Alternative: Use the full covariance matrix C.
                # Since n=5, inverting a 5x5 matrix is cheap.
                
                # Let's switch to full C matrix for correctness.
                # Re-implement run_cmaes with C matrix.
                
                break # Placeholder, we need to rewrite the loop with C matrix.

            return

        # Since implementing CMA-ES from scratch with B,D is error-prone,
        # and n=5 is small, let's use a simpler but robust approach:
        # **Multi-Start Nelder-Mead**
        
        # Reset calls
        calls = 0
        best_x = None
        best_f = np.inf
        
        num_restarts = 5
        evals_per_restart = budget // num_restarts
        
        for k in range(num_restarts):
            # Initial point
            if k == 0:
                x0 = np.zeros(n) # Try origin first
            else:
                x0 = rng.uniform(-5, 5, n)
            
            # Nelder-Mead
            # Simplex initialization
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                # Standard deviation for initial simplex
                # Rastrigin has features of size ~1.
                # Use 0.5 or 1.0?
                # 0.5 is safe.
                xi[i] += 0.5
                simplex.append(xi)
            
            fs = [func(x) for x in simplex]
            calls += n + 1
            
            # Nelder-Mead loop
            max_iter = 50 # Generations
            for _ in range(max_iter):
                if calls >= budget:
                    break
                    
                # Sort
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]
                
                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = simplex[0].copy()
                
                # Centroid of best n points
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls += 1
                
                if f_ref < fs[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls += 1
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fs[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fs[-1] = f_ref
                elif f_ref < fs[-2]:
                    simplex[-1] = x_ref
                    fs[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls += 1
                    
                    if f_con < fs[-1]:
                        simplex[-1] = x_con
                        fs[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fs[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break
                        
                        if calls >= budget:
                            break
                
                # Check convergence
                if np.max(fs) - np.min(fs) < 1e-8:
                    break
        
        return best_x, best_f
