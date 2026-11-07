import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        eval_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            if eval_count >= budget or not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial Sampling
        # Use 25 points to get a good spread
        n_initial = 25
        candidates = []
        for i in range(n_initial):
            if eval_count >= budget or not time_ok():
                break
            x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x.copy()))

        if not candidates:
            x = rng.uniform(-5, 5, n)
            evaluate(x)
            return best_x, best_f

        # Sort candidates
        candidates.sort(key=lambda t: t[0])
        x0 = candidates[0][1]
        f0 = candidates[0][0]
        
        # Phase 2: CMA-ES
        # Parameters for CMA-ES
        # lambda = 4 + floor(3 * ln(n))
        lambda_ = 4 + int(3 * np.log(n))
        if lambda_ < 4:
            lambda_ = 4
            
        # Initial sigma: heuristic based on spread of initial points or fixed
        # For BBOB, 1.0 is a reasonable start, but let's try 0.5 to be more local
        sigma = 0.5
        
        # Means and covariances
        mean = x0.copy()
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)
        C_sqrt = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Learning rates
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        # Precompute
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Main loop
        while eval_count < budget and time_ok():
            # Generate population
            ys = []
            xs = []
            for i in range(lambda_):
                if eval_count >= budget or not time_ok():
                    break
                z = rng.standard_normal(n)
                y = C_sqrt @ z
                x = mean + sigma * y
                x = np.clip(x, -5, 5)
                
                fx = evaluate(x)
                if fx is None:
                    break
                ys.append(y)
                xs.append((x, fx))
            
            if not xs:
                break
                
            # Sort by fitness
            xs.sort(key=lambda t: t[1])
            # Take top mu
            xs = xs[:mu]
            ys = ys[:mu]
            
            # Weighted mean
            new_mean = sum(weights[i] * xs[i][0] for i in range(mu))
            
            # Update evolution paths
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (C_sqrt_inv @ (new_mean - mean) / sigma)
            
            # h_sigma condition
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (eval_count - eval_count + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            # Note: The exponent in the denominator should be based on the number of generations, 
            # but since we are in a single run, we approximate. 
            # Actually, the standard formula is:
            # (||p_sigma|| / sqrt(1 - (1-c_sigma)^(2*gen))) < (1.4 + 2/(n+1)) * chi_n
            # We don't track 'gen' explicitly, but eval_count / lambda approximates it.
            # Let's use a simpler check or just the norm.
            # For robustness, let's use the norm check.
            h_sigma = np.linalg.norm(p_sigma) < (1.4 + 2 / (n + 1)) * chi_n
            
            # p_c update
            hsig = 1.0 if h_sigma else 0.0
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_sqrt_inv @ (new_mean - mean) / sigma)
            
            # Covariance update
            rank_one = np.outer(p_c, p_c)
            rank_mu = sum(weights[i] * np.outer(ys[i], ys[i]) for i in range(mu))
            
            C = (1 + c_1 * hsig * (2 - c_c) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure C is positive definite (add small regularization if needed)
            # In practice, CMA-ES maintains PD, but numerical errors can creep in.
            # We can skip explicit check for speed, or do a quick eigenvalue check.
            # For 5D, it's usually stable.
            
            # Update C_sqrt and C_sqrt_inv
            # Use Cholesky or Eigendecomposition. Eigendecomposition is more stable for updates.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            mean = new_mean
            
            # Check convergence
            if np.linalg.norm(mean - best_x) < 1e-8 and abs(f0 - best_f) < 1e-8:
                break
                
            # If sigma is too small, break
            if sigma < 1e-6:
                break

        # Phase 3: Local Polish with Nelder-Mead
        if best_x is not None and eval_count < budget and time_ok():
            x = best_x.copy()
            f = best_f
            # Simple Nelder-Mead implementation
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5
            
            simplex = [x.copy()]
            for j in range(n):
                point = x.copy()
                delta = 0.1 if x[j] + 0.1 <= 5 else -0.1
                point[j] += delta
                point = np.clip(point, -5, 5)
                simplex.append(point)
            
            fvals = [f]
            for pt in simplex[1:]:
                if eval_count >= budget or not time_ok():
                    break
                fx = evaluate(pt)
                if fx is None:
                    break
                fvals.append(fx)
                
            if len(fvals) != len(simplex):
                return best_x, best_f
                
            for _ in range(50): # Max 50 NM steps
                if eval_count >= budget or not time_ok():
                    break
                    
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if abs(fvals[-1] - fvals[0]) < 1e-10:
                    break
                    
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                if eval_count >= budget or not time_ok():
                    break
                fr = evaluate(xr)
                if fr is None:
                    break
                
                if fr < fvals[0]:
                    # Expansion
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    if eval_count >= budget or not time_ok():
                        break
                    fe = evaluate(xe)
                    if fe is None:
                        simplex[-1], fvals[-1] = xr, fr
                        break
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                    else:
                        xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                    if eval_count >= budget or not time_ok():
                        break
                    fc = evaluate(xc)
                    if fc is None:
                        break
                    if fc < min(fr, fvals[-1]):
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = np.clip(simplex[0] + sigma_shrink * (simplex[i] - simplex[0]), -5, 5)
                            if eval_count >= budget or not time_ok():
                                break
                        if eval_count >= budget or not time_ok():
                            break
                        fvals = [f0]
                        for pt in simplex[1:]:
                            fx = evaluate(pt)
                            if fx is None:
                                break
                            fvals.append(fx)
                        if len(fvals) != len(simplex):
                            break

        return best_x, best_f
