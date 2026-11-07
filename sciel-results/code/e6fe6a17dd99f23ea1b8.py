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
        deadline = start_time + 290.0  # Safety margin

        # 1. Initial Exploration with Halton Sequence
        # This ensures we cover the space well early on for AOCC.
        def halton(dim, n):
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2
                f = 1.0
                i = 1
                for _ in range(n):
                    f /= base
                    i += 1
                    while i % base == 0:
                        i //= base
                        f *= base
                    points[_, d] = f
            return points

        n_init = 20
        if budget <= n_init:
            x = np.random.uniform(-5, 5, dim)
            f = func(x)
            return x, f

        H = halton(dim, n_init)
        X_init = H * 10.0 - 5.0
        
        # Add center point
        key_points = [np.zeros(dim)]
        all_init_points = list(X_init) + key_points
        X_init = np.array(all_init_points[:n_init]) # Keep it simple, just use Halton + Center if needed, but Halton is good.
        # Actually, let's just use the first n_init Halton points.
        X_init = H * 10.0 - 5.0

        F_init = np.array([func(x) for x in X_init])
        evals = n_init
        
        best_idx = np.argmin(F_init)
        best_x = X_init[best_idx].copy()
        best_f = F_init[best_idx]

        # 2. CMA-ES Phase
        # CMA-ES is robust to ill-conditioning.
        
        # Parameters
        lambda_pop = 8
        # Initial step size: heuristic based on dimension
        sigma = 1.0 
        
        # Mean of the distribution
        mean = best_x.copy()
        
        # Covariance matrix (initially identity)
        C = np.eye(dim)
        
        # Weights
        mu = lambda_pop // 2
        weights = np.log(lambda_pop / 2 + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Cholesky of C
        try:
            D = np.linalg.cholesky(C)
        except:
            D = np.eye(dim)
        
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            X_pop = np.zeros((lambda_pop, dim))
            for i in range(lambda_pop):
                z = np.random.randn(dim)
                x_i = mean + sigma * D @ z
                # Clip to box? CMA-ES can handle boundaries, but clipping helps.
                # The task says func clips. So we can just evaluate.
                X_pop[i] = x_i
            
            # Evaluate
            F_pop = np.array([func(x) for x in X_pop])
            evals += lambda_pop
            
            # Sort
            order = np.argsort(F_pop)
            X_pop = X_pop[order]
            F_pop = F_pop[order]
            
            # Update best
            if F_pop[0] < best_f:
                best_f = F_pop[0]
                best_x = X_pop[0].copy()
            
            # Update mean
            mean_new = np.zeros(dim)
            for i in range(mu):
                mean_new += weights[i] * X_pop[i]
            
            # Update step size (sigma)
            # Compute norm of p_sigma
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) / sigma * D^T (mean_new - mean) / ||D^T (mean_new - mean)|| ?
            # Actually, the standard update is:
            # p_sigma = (1 - c_sigma) p_sigma + sqrt(c_sigma (2 - c_sigma) mu_eff) * D^{-1} (mean_new - mean) / sigma
            # Wait, D is such that C = D D^T. So D^{-1} is needed.
            # Let's use the standard CMA-ES update rules.
            
            # Vector from old mean to new mean
            v = (mean_new - mean) / sigma
            # Project into the principal axes defined by D
            # v_D = D^{-1} v
            # Since C = D D^T, D is lower triangular.
            try:
                v_D = np.linalg.solve(D, v)
            except:
                v_D = v
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_D
            
            # Hessian trace term for step size adaptation
            # h_sigma = 1 if ||p_sigma||^2 / (1 - (1-c_sigma)^{2gen}) < (1+2/dim) * E||N(0,I)||^2
            # Approximation:
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            if np.linalg.norm(p_sigma)**2 / (1 - (1-c_sigma)**2) < (1 + 2/dim) * chi_n**2:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance matrix
            # p_c = (1 - c_c) p_c + h_sigma sqrt(c_c (2 - c_c) mu_eff) v
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v
            
            # Rank-one update
            C += c_1 * (np.outer(v, v) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            # Rank-mu update
            # C += c_mu * sum w_i (x_i - mean_new)(x_i - mean_new)^T
            # We need D^{-1} (x_i - mean_new)
            X_diff = X_pop[:mu] - mean_new
            # Solve D Y = X_diff^T => Y = D^{-1} X_diff^T
            try:
                Y = np.linalg.solve(D, X_diff.T)
            except:
                Y = X_diff.T
            
            # C += c_mu * D Y Y^T D^T
            # Actually, the update is C = (1 - c_1 - c_mu) C + ...
            # Let's just do the standard update:
            # C = (1 - c_1 - c_mu) C + c_1 (vv^T + ...) + c_mu D Y Y^T D^T
            # This is getting complex to implement correctly from scratch without errors.
            
            # Simpler approach: Use a well-tested minimal CMA-ES implementation logic.
            # Given the risk of implementation errors in CMA-ES, let's use a simpler adaptive method:
            # Nelder-Mead with random restarts?
            # Or just a robust Random Search with Adaptive Step Size?
            
            # Let's stick to a simpler, robust strategy:
            # 1. Halton Init (done)
            # 2. Coordinate Descent with Adaptive Step (Simple and effective for 5D)
            # 3. If stuck, perturb.
            
            # The current program already has Coordinate Descent.
            # The GP part is the weak link.
            # Let's replace GP with a simple "Adaptive Random Search" around the best point.
            # Or better, implement a correct CMA-ES.
            
            # I will implement a simplified CMA-ES that is known to work.
            # See: "CMA-ES: A Tutorial"
            
            # Let's restart the logic with a verified simple CMA-ES.
            break # Break out of the complex loop above and use a cleaner implementation below.

        # --- Clean CMA-ES Implementation ---
        # Reset state for a clean run if the above loop didn't finish or was broken.
        # Actually, let's just replace the whole GP section with a clean CMA-ES.
        
        # Re-initialize CMA-ES variables properly
        mean = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)
        
        lambda_pop = 8
        mu = lambda_pop // 2
        weights = np.log(lambda_pop / 2 + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            X_pop = []
            for i in range(lambda_pop):
                z = np.random.randn(dim)
                # Cholesky decomposition of C
                try:
                    D = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    D = np.eye(dim)
                
                x_i = mean + sigma * D @ z
                X_pop.append(x_i)
            
            X_pop = np.array(X_pop)
            F_pop = np.array([func(x) for x in X_pop])
            evals += lambda_pop
            
            # Sort
            order = np.argsort(F_pop)
            X_pop = X_pop[order]
            F_pop = F_pop[order]
            
            # Update best
            if F_pop[0] < best_f:
                best_f = F_pop[0]
                best_x = X_pop[0].copy()
            
            # Update mean
            mean_new = np.sum(weights[:, None] * X_pop[:mu], axis=0)
            
            # Update step size
            v = (mean_new - mean) / sigma
            try:
                v_D = np.linalg.solve(D, v)
            except:
                v_D = v
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_D
            
            if np.linalg.norm(p_sigma)**2 / (1 - (1-c_sigma)**2) < (1 + 2/dim) * chi_n**2:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update covariance
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v
            
            # Rank-one
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(v, v) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            
            # Rank-mu
            X_diff = X_pop[:mu] - mean_new
            try:
                Y = np.linalg.solve(D, X_diff.T)
            except:
                Y = X_diff.T
            
            C += c_mu * (D @ (Y @ Y.T) @ D.T)
            
            mean = mean_new
            
            # Check for convergence
            if sigma < 1e-8:
                break

        # 3. Local Refinement (Coordinate Descent)
        if evals < budget and time.time() < deadline:
            step = 0.5
            x = best_x.copy()
            f = best_f
            no_improve = 0
            max_no_improve = 10
            
            while evals < budget and time.time() < deadline and no_improve < max_no_improve:
                improved = False
                for i in range(dim):
                    if evals >= budget or time.time() > deadline:
                        break
                    for direction in [-1, 1]:
                        x_trial = x.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        evals += 1
                        if f_trial < f:
                            x = x_trial
                            f = f_trial
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    no_improve += 1
                    if step < 1e-5:
                        break
                else:
                    no_improve = 0
                    step = min(step * 1.2, 1.0)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()

        return best_x, best_f
