import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0 # Safety margin
        
        d = self.dim
        # CMA-ES parameters
        # For small d, lambda is often d + 3 + floor(3 log d)
        # For d=5, lambda = 5 + 3 + 3*1 = 11? Let's stick to 10 or 12.
        # Standard recommendation: lambda = 4 + floor(3 log d)
        # d=5 -> 4 + 3*1 = 7.
        # Let's use lambda = 10 for more robustness in 5D.
        lam = 10
        mu = lam // 2 # Selection pressure
        
        # Weights
        # Standard weights: w_i = log(mu + 0.5) - log(i) for i=1..mu
        # Normalized to sum to 1
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Initial state
        # Start at 0,0,0,0,0
        x_mean = np.zeros(d)
        # Initial sigma
        # Box is [-5, 5]. Distance to boundary is 5.
        # Sigma = 2.0 seems reasonable to cover the space quickly.
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Update rates
        # c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        # c_c = (4 + mu_eff/d) / (d + 4 + 2*mu_eff/d)
        c_c = (4 + mu_eff/d) / (d + 4 + 2*mu_eff/d)
        # c_1 = 2 / ((d+1.3)^2 + mu_eff)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        # c_mu = (mu_eff - 2 + 1/mu_eff) / ((d+2)^2 + mu_eff)
        c_mu = (mu_eff - 2 + 1/mu_eff) / ((d + 2)**2 + mu_eff)
        # Damping for ps
        damps = 1 + 2 * max(0, sqrt(mu_eff/d - 1)) + c_sigma
        
        # Eigen decomposition for C is needed for sampling
        # We will update C every generation? Or every 10?
        # For d=5, updating every generation is fine.
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Precompute Cholesky or Eigen for sampling?
        # We will use np.linalg.eigh for C to get B and D
        # x = x_mean + sigma * B @ (D @ z)
        
        for gen in range(self.budget // lam):
            if time.time() > deadline - 5.0:
                break
            
            # Sample points
            B, D = np.linalg.eigh(C)
            # D is diagonal of eigenvalues, B is eigenvectors
            # Covariance matrix C = B diag(D^2) B^T
            # Sampling: z ~ N(0, I)
            # y = B @ (D @ z)
            
            ys = []
            for i in range(lam):
                z = np.random.randn(d)
                y = B @ (D * z)
                x = x_mean + sigma * y
                # Clip to box? 
                # The problem says func(x) clips. 
                # But for CMA-ES, it's better to keep the search within bounds or let it clip.
                # The prompt says "func(x) clips a finite length-5 vector to the box".
                # So we can pass unclipped x, func will handle it.
                # However, if we go far out, we waste evaluations on boundary points.
                # It's generally better to clip x before passing to func to avoid "wasted" evaluations 
                # that are effectively the same as the boundary point.
                # But CMA-ES assumes the function is defined everywhere.
                # Since the evaluator clips, passing out-of-bounds is fine, but might be inefficient.
                # Let's clip x to [-5, 5] to be safe and efficient.
                x = np.clip(x, -5, 5)
                ys.append((x, func(x)))
                
                calls += 1
                if calls >= self.budget:
                    break
            
            if calls >= self.budget:
                break
                
            # Sort by fitness
            ys.sort(key=lambda t: t[1])
            
            # Update best
            if ys[0][1] < best_f:
                best_f = ys[0][1]
                best_x = ys[0][0]
                
            # Selection
            # Take the best mu points
            selected_xs = np.array([y[0] for y in ys[:mu]])
            selected_fs = [y[1] for y in ys[:mu]]
            
            # Update Mean
            old_mean = x_mean.copy()
            x_mean = np.sum(selected_xs * weights[:, np.newaxis], axis=0)
            
            # Update Covariance Matrix
            # Delta rule
            delta = 0
            if np.linalg.norm(x_mean - old_mean) > 0:
                delta = (x_mean - old_mean) / (sigma * np.linalg.norm(x_mean - old_mean))
            
            # Update pc
            pc = (1 - c_c) * pc + c_c * (x_mean - old_mean) / sigma
            # Check for saturation?
            # h_sigma = (np.linalg.norm(ps) / sqrt(1 - (1-c_sigma)**(2*(gen+1))) / np.linalg.norm(B @ (D * np.ones(d)))) < (1.4 + 2/(d+1)) * sqrt((mu_eff+2)/(d+2))
            # This is complex. For d=5 and few generations, we can skip the h_sigma check or simplify.
            # Let's implement the standard h_sigma check.
            ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x_mean - old_mean) / sigma
            # Note: (x_mean - old_mean) / sigma is the step in sigma units.
            
            # h_sigma calculation
            # norm of B @ (D * ones)
            # D is sqrt(eigenvalues). 
            # Actually, the norm of the transformation of the unit vector.
            # Let's use the approximation or just set h_sigma = 1 if we don't want to risk bugs.
            # Given the small budget, saturation is unlikely to be a major issue in the first 50 gens.
            # I will set h_sigma = 1.0 for simplicity and robustness against implementation errors in the check.
            h_sigma = 1.0
            
            pc = (1 - c_c) * pc + c_c * h_sigma * (x_mean - old_mean) / sigma
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc * pc.T + (1 - h_sigma) * (1 - c_c) * c_c * C) + c_mu * sum(wi * (yi - old_mean) * (yi - old_mean).T)
            
            diff = x_mean - old_mean
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - h_sigma) * (1 - c_c) * c_c * C)
            
            for i in range(mu):
                d_i = selected_xs[i] - old_mean
                C += c_mu * weights[i] * np.outer(d_i, d_i)
                
            # Update Sigma
            chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
            sigma *= np.exp(c_sigma / damps * (np.linalg.norm(ps)/chi_n - 1))
            
            # Ensure C is symmetric positive definite
            # Sometimes numerical errors make it slightly asymmetric or indefinite
            C = (C + C.T) / 2
            # Eigen check
            try:
                evals, evecs = np.linalg.eigh(C)
                if np.any(evals < 0):
                    # Add small identity
                    C += np.eye(d) * 1e-6
            except:
                C = np.eye(d)
                
        return best_x, best_f
