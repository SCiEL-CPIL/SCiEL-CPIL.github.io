import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            pt = np.asarray(pt, dtype=np.float64)
            if pt.shape != (dim,):
                pt = pt.reshape(dim)
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Sphere-Specific Model Construction
        # Assume f(x) = a * ||x||^2 + q^T x + c
        # Points: 0, +/- d * e_i
        d = 2.0
        points = [np.zeros(dim)]
        for i in range(dim):
            p1 = np.zeros(dim); p1[i] = d
            p2 = np.zeros(dim); p2[i] = -d
            points.append(p1)
            points.append(p2)
            
        values = np.array([evaluate(p) for p in points])
        
        c_val = values[0]
        a_vals = []
        q_vec = np.zeros(dim)
        
        for i in range(dim):
            val_pos = values[1 + 2*i]
            val_neg = values[2 + 2*i]
            
            # f(d*e_i) = a*d^2 + q_i*d + c
            # f(-d*e_i) = a*d^2 - q_i*d + c
            # Diff: 2*q_i*d = val_pos - val_neg
            # Sum: 2*a*d^2 + 2*c = val_pos + val_neg
            
            q_i = (val_pos - val_neg) / (2 * d)
            a_i = (val_pos + val_neg - 2*c_val) / (2 * d * d)
            
            a_vals.append(a_i)
            q_vec[i] = q_i
            
        a = np.mean(a_vals)
        
        if abs(a) < 1e-12:
            a = 1e-12 * np.sign(a) if a != 0 else 1e-12
            
        # Minimum at x* = -q / (2*a)
        x_init = -q_vec / (2 * a)
        x_init = np.clip(x_init, -5, 5)
        
        # Evaluate the estimated optimum
        f_init = evaluate(x_init)
        
        # Phase 2: CMA-ES Polish
        # Initialize CMA-ES parameters
        x = best_x.copy()
        sigma = 1.0 # Initial step size
        
        # For Sphere, the Hessian is isotropic. 
        # If a is significantly different from 1, scale sigma?
        # f(x) ~ a * ||x - x*||^2. 
        # The "width" of the valley is related to 1/sqrt(a).
        # Let's start with a sigma that is a fraction of the distance from 0 to x_init, or just a fixed small value.
        # Given we are very close, a small sigma is safer to avoid overshooting.
        # However, CMA-ES adapts sigma.
        
        # Population size for CMA-ES
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(np.arange(1, mu+1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # CMA-ES state
        C = np.eye(dim)
        Cs = np.zeros(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Update rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        
        # Damping for step size control (1/5-th rule equivalent in CMA-ES is handled by sigma update)
        # Standard CMA-ES sigma update:
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Precompute Cholesky of C
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)
        
        # Restart logic
        restart_count = 0
        max_restarts = 5
        no_improve = 0
        improve_threshold = 1e-10
        
        while calls_made < budget and time.time() < deadline:
            
            # Generate population
            offspring = []
            offspring_f = []
            
            # We need to generate lambda_ points, but we are limited by budget.
            # We will generate as many as we can in this iteration, or just 1 if we are running low?
            # Actually, CMA-ES needs a full population to update C.
            # If budget is low, we might just do (1+1) or (1+lambda) with fixed C?
            # For simplicity and robustness, if remaining budget < lambda_, we switch to (1+1)-ES using the current C.
            
            remaining = budget - calls_made
            if remaining <= 0:
                break
                
            if remaining < lambda_:
                # Switch to (1+1) mode
                # Generate 1 offspring
                z = np.random.multivariate_normal(np.zeros(dim), C, size=1).flatten()
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                
                if fy < best_f:
                    x = y
                    fx = fy
                    # Simple sigma adaptation for (1+1)
                    # This is a heuristic, not standard CMA-ES (1+1)
                    sigma *= 1.1 
                else:
                    sigma *= 0.9
                    
                # Update C? No, too expensive/risky with 1 point.
                # Just keep C fixed or decay it?
                # Let's just keep C fixed and rely on sigma.
                # Actually, we can update C using the last few points if we stored them, but let's keep it simple.
                
                # Check for stagnation
                if best_f < f_init - improve_threshold:
                    no_improve = 0
                    f_init = best_f
                else:
                    no_improve += 1
                    
                if no_improve > 20 and sigma < 1e-6:
                    break
                continue
                
            # Standard CMA-ES step
            # Generate lambda_ points
            for i in range(lambda_):
                if calls_made >= budget or time.time() > deadline:
                    break
                z = np.random.multivariate_normal(np.zeros(dim), C, size=1).flatten()
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                offspring.append(y)
                offspring_f.append(fy)
                
            if len(offspring) < lambda_:
                # If we ran out of budget mid-population, we can't do a proper CMA-ES update.
                # We could just pick the best one and stop, or do a partial update?
                # Let's just stop to avoid inconsistent state.
                # But we might have made progress.
                # We update best_x/best_f inside evaluate.
                break
                
            # Sort by fitness
            idx = np.argsort(offspring_f)
            offspring = [offspring[i] for i in idx]
            offspring_f = [offspring_f[i] for i in idx]
            
            # Select best mu
            y_mu = np.array(offspring[:mu])
            
            # Weighted mean
            m_new = np.sum(weights[:, None] * y_mu, axis=0)
            
            # Update step size
            # d_new = m_new - x
            d_new = m_new - x
            
            # Update ps
            ps = (1 - c_sigma) * ps + c_sigma * (chi_n / np.linalg.norm(d_new)) * d_new if np.linalg.norm(d_new) > 1e-16 else (1 - c_sigma) * ps
            
            # Update pc
            pc = (1 - c_c) * pc + c_c * (m_new - x)
            
            # Update C
            # hsig = 1 if ||ps|| < (1 + 1/(2*c_sigma)) * chi_n * sqrt(1 - (1-c_sigma)^(2*(gen+1))) else 0
            # We don't track gen, so we approximate.
            # Simpler check:
            hsig = 1.0 if np.linalg.norm(ps) < (1 + 1/(2*c_sigma)) * chi_n else 0.0
            
            C = (1 - c_c * (2 - c_c) * mu_eff + c_sigma * (2 - c_sigma) * mu_eff * hsig) * C \
                + c_sigma * mu_eff * (1 + (2 - c_sigma)/c_sigma * hsig - 1) * np.outer(d_new, d_new) / (sigma**2) \
                - c_sigma * mu_eff * hsig * np.outer(ps, ps) / (chi_n**2)
                
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update x
            x = m_new
            
            # Update sigma
            # Standard CMA-ES sigma update
            # sigma = sigma * exp(c_sigma/chi_n * (||ps||/chi_n - 1))
            sigma = sigma * np.exp(c_sigma/chi_n * (np.linalg.norm(ps)/chi_n - 1))
            
            # Ensure sigma is positive
            if sigma < 1e-12:
                sigma = 1e-12
            
            # Recalculate Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset it
                C = np.eye(dim)
                L = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                
            # Check for stagnation
            if best_f < f_init - improve_threshold:
                no_improve = 0
                f_init = best_f
            else:
                no_improve += 1
                
            if no_improve > 20 and sigma < 1e-6:
                break

        return best_x, best_f
