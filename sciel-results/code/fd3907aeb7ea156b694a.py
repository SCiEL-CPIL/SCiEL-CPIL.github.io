import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

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
        # This is the exact structure of the Sphere function.
        # We need 1 + 2*dim points to estimate a, q, c.
        # For dim=5, that's 11 points. This is extremely efficient.
        
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
            # Fallback if the function is nearly linear or flat
            a = 1e-12 * np.sign(a) if a != 0 else 1e-12
            
        # Minimum at x* = -q / (2*a)
        x_init = -q_vec / (2 * a)
        x_init = np.clip(x_init, -5, 5)
        
        # Evaluate the estimated optimum
        f_init = evaluate(x_init)
        
        # Phase 2: CMA-ES Polish
        # We start very close to the optimum. CMA-ES is robust and handles
        # any slight deviations from the pure sphere model (e.g. numerical noise
        # or if the "rotation" mentioned in the prompt implies a general quadratic
        # that isn't perfectly isotropic, though Sphere is isotropic).
        # Actually, the prompt says "Sphere", so it IS isotropic.
        # But CMA-ES is a safe bet for polishing.
        
        x = best_x.copy()
        fx = best_f
        
        # Initial sigma: We are at the analytical minimum.
        # The curvature is 'a'. The "width" of the well is ~1/sqrt(a).
        # But since we are AT the minimum, we should start small to avoid
        # wasting evaluations far away if the model is slightly off.
        # However, if the model is exact, any step is an error.
        # Let's start with a small sigma, maybe related to the distance from 0 to x_init?
        # No, just a small fixed value scaled by 1/sqrt(a) to be in "normalized" units.
        sigma_scale = 1.0 / np.sqrt(abs(a)) if abs(a) > 1e-12 else 1.0
        sigma = 0.5 * sigma_scale
        
        rng = np.random.default_rng(12345)
        
        # CMA-ES parameters
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
        
        # Damping
        chi_n = np.sqrt(dim) * (1 + 1/(2*dim) + 1/(8*dim**2))
        
        # We will run CMA-ES until budget or time is up.
        # Since we are very close, CMA-ES will converge quickly.
        # We need to be careful not to spend too many calls if we are already at machine precision.
        
        while calls_made < budget and time.time() < deadline:
            # Generate population
            Y = np.zeros((lambda_, dim))
            F = np.zeros(lambda_)
            
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C becomes singular
                    L = np.eye(dim)
                
                Y[i] = x + sigma * (L @ z)
                Y[i] = np.clip(Y[i], -5, 5)
                F[i] = evaluate(Y[i])
            
            if calls_made >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Update mean
            x_new = np.sum(weights[:, None] * Y[:mu], axis=0)
            
            # Update covariance
            y_weighted = (Y[:mu] - x)[:, None] * (weights / sigma)[:, None]
            C = (1 - c_c) * C + c_c * (y_weighted @ y_weighted.T + (1 - mu_eff) * C)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update step size
            ps = (1 - c_sigma) * ps + c_sigma / np.sqrt(mu_eff) * (y_weighted @ np.ones(mu))
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(ps) - chi_n))
            
            # Update path
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff * c_c * (2 - c_c)) / sigma * (x_new - x)
            
            x = x_new
            fx = F[0]
            
            # Convergence check
            if sigma < 1e-12:
                break

        return best_x, best_f
