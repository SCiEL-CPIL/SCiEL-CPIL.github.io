import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        calls = 0
        deadline = time.time() + 285.0
        
        # Phase 1: CMA-ES
        # Use 80% of budget for CMA-ES
        cma_budget = int(self.budget * 0.8)
        
        # CMA-ES Parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        nu_eff = 1 / np.sum(weights ** 2)
        c_sigma = (nu_eff + 2) / (n + nu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((nu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + nu_eff / n) / (n + 4 + 2 * nu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + nu_eff)
        c_mu = min(1 - c_1, 2 * (nu_eff - 2 + 2 / nu_eff) / ((n + 2) ** 2 + nu_eff))

        # Initialization
        # Sample 5 random points to find a better start
        rng = np.random.default_rng(42)
        x_starts = rng.uniform(-5, 5, (5, n))
        f_starts = [func(x) for x in x_starts]
        calls += 5
        best_idx = np.argmin(f_starts)
        x_mean = x_starts[best_idx].copy()
        best_f = f_starts[best_idx]
        best_x = x_starts[best_idx].copy()
        
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # CMA-ES Loop
        for gen in range(cma_budget - calls):
            if time.time() > deadline:
                break

            # Sample lambda points
            try:
                D = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                D = np.eye(n)
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                sigma = 1.0

            Y = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x_mean + sigma * (D @ z)
                Y.append(y)
            Y = np.array(Y)

            # Evaluate
            F = []
            for y in Y:
                y_clipped = np.clip(y, -5, 5)
                f = func(y_clipped)
                F.append(f)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = y_clipped.copy()

            F = np.array(F)
            order = np.argsort(F)
            Y_sorted = Y[order]
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

            # Update step-size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * nu_eff) * (D @ (x_mean_new - x_mean)) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            delta_hsig = 1 - hsig
            pc = (1 - c_c) * pc + delta_hsig * np.sqrt(c_c * (2 - c_c) * nu_eff) * (x_mean_new - x_mean) / sigma

            C = (1 - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) + delta_hsig * (c_c * (2 - c_c) - 1) * C) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(Y_sorted[:mu] - x_mean, Y_sorted[:mu] - x_mean), axis=0) / sigma ** 2

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e12)
            x_mean = x_mean_new

        # Phase 2: Polish with Nelder-Mead
        # Use remaining budget
        remaining_budget = self.budget - calls
        if remaining_budget > 0 and time.time() < deadline:
            # Define a wrapper to count calls and respect deadline
            nm_calls = [0]
            def nm_func(x):
                if time.time() > deadline:
                    return best_f
                x_clipped = np.clip(x, -5, 5)
                f = func(x_clipped)
                nm_calls[0] += 1
                if f < best_f:
                    best_f = f
                    best_x = x_clipped.copy()
                return f

            # Run Nelder-Mead
            try:
                # Initialize simplex around best_x
                # Simplex: best_x + small perturbations
                initial_simplex = [best_x.copy()]
                for i in range(n):
                    x_pert = best_x.copy()
                    x_pert[i] += 0.1
                    x_pert[i] = np.clip(x_pert[i], -5, 5)
                    initial_simplex.append(x_pert)
                
                # Use minimize with Nelder-Mead
                # We need to limit function evaluations in scipy.optimize.minimize
                # This is tricky. We can just run it and hope it stops, or use a callback to break.
                # Better: implement a simple local search ourselves or limit iterations.
                
                # Let's use a simple coordinate descent or Nelder-Mead with a strict call limit.
                # Since scipy.optimize.minimize doesn't have a strict "max calls" that stops immediately 
                # in all versions easily without callbacks, we can use a callback to raise an exception 
                # or just rely on the deadline check inside nm_func.
                
                # Actually, Nelder-Mead can be slow. Let's do a simple local hill climb.
                # Or, use scipy.optimize.minimize with method='Nelder-Mead' and options={'maxiter': 100}
                # and wrap the function to stop if deadline is reached.
                
                res = minimize(nm_func, best_x, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
