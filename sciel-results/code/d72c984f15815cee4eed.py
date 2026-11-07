"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with Hooke-Jeeves polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        
        # Initial point
        x_best = rng.uniform(-5, 5, d)
        f_best = func(x_best)
        calls = 1

        # CMA-ES Parameters
        lambda_ = 6
        mu = lambda_ // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        # Strategy parameters
        c_sigma = (lambda_ / 2 + 1) / (d + 4 + lambda_ / 2)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ / 2 - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 - lambda_ / d + c_sigma) / (d + 4 + 2 * c_sigma)
        c_1 = 2 / ((d + 1.3)**2 + lambda_)
        c_mu = min(1 - c_1, (2 * (mu_eff - 2 + 1 / mu_eff)) / ((d + 2)**2 + lambda_))
        chi_N = np.sqrt(d) * (1 - 1 / (2 * d) + 1 / (12 * d**2))

        # State variables
        x_m = x_best.copy()  # Mean
        sigma = 2.0
        C = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        hsig = 0.0

        es_budget = int(budget * 0.8)
        
        while calls < es_budget and time_ok():
            # Generate offspring
            ys = []
            fs = []
            
            for i in range(lambda_):
                # Sample
                z = rng.standard_normal(d)
                # Transform using Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C becomes singular (rare but possible)
                    C = np.eye(d)
                    L = np.eye(d)
                
                y = x_m + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                
                ys.append(y)
                fs.append(fy)

                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()

            # Sort offspring
            indices = np.argsort(fs)
            ys = [ys[i] for i in indices]
            fs = [fs[i] for i in indices]

            # Update mean
            y_w = np.zeros(d)
            for i in range(mu):
                y_w += w[i] * ys[i]
            
            # Update step size
            y_sigma = (np.linalg.norm(y_w) / (sigma * chi_N)) if sigma > 0 else 0
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_w / sigma if sigma > 0 else 0)
            
            hsig = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - lambda_))) < (1.4 + 2 / (d + 1)) * chi_N
            if hsig:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))

            # Update covariance
            y_c = (x_m + sigma * y_w) - x_m
            x_m = x_m + sigma * y_w
            
            # Clip mean? No, keep mean free, but clip samples. 
            # However, if mean goes out of bounds, we might waste calls. 
            # Let's clip the mean slightly to keep it in a reasonable range, 
            # or just let it be. The samples are clipped.
            
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                v = (ys[i] - x_m) / sigma
                rank_mu += w[i] * np.outer(v, v)
            
            C = (1 + c_1 * (hsig * (np.outer(p_c, p_c) - C) - c_mu * rank_mu) - c_1) * C
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definite (regularization)
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-10:
                    eigvals = np.maximum(eigvals, 1e-10)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            except Exception:
                C = np.eye(d)

        # Phase 2: Hooke-Jeeves Polish
        if time_ok() and calls < budget:
            x = x_best.copy()
            f_x = f_best
            step = 0.5
            min_step = 1e-6
            
            while calls < budget and time_ok() and step > min_step:
                improved = False
                for i in range(d):
                    # Try positive step
                    x_temp = x.copy()
                    x_temp[i] += step
                    x_temp = np.clip(x_temp, -5, 5)
                    f_temp = func(x_temp)
                    calls += 1
                    
                    if f_temp < f_x:
                        x = x_temp
                        f_x = f_temp
                        improved = True
                        # Try to extend in same direction
                        while calls < budget and time_ok():
                            x_next = x.copy()
                            x_next[i] += step
                            x_next = np.clip(x_next, -5, 5)
                            f_next = func(x_next)
                            calls += 1
                            if f_next < f_x:
                                x = x_next
                                f_x = f_next
                            else:
                                break
                    else:
                        # Try negative step
                        x_temp = x.copy()
                        x_temp[i] -= step
                        x_temp = np.clip(x_temp, -5, 5)
                        f_temp = func(x_temp)
                        calls += 1
                        
                        if f_temp < f_x:
                            x = x_temp
                            f_x = f_temp
                            improved = True
                            # Try to extend in same direction
                            while calls < budget and time_ok():
                                x_next = x.copy()
                                x_next[i] -= step
                                x_next = np.clip(x_next, -5, 5)
                                f_next = func(x_next)
                                calls += 1
                                if f_next < f_x:
                                    x = x_next
                                    f_x = f_next
                                else:
                                    break
                        else:
                            # Try smaller step? Or just move to next dimension
                            pass

                if not improved:
                    step *= 0.5
                else:
                    # Keep step or increase slightly? Standard HJ keeps step if improved.
                    # Let's keep step constant to avoid oscillation.
                    pass

                if f_x < f_best:
                    f_best = f_x
                    x_best = x.copy()

        return x_best, f_best
# EVOLVE-BLOCK-END
