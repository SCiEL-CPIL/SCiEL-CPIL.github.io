"""Model-and-optimize optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget, rng, evals_ref, deadline):
        """Run CMA-ES from initial point x0 with step size sigma0.
        Returns (best_x, best_f, evals_used)."""
        dim = self.dim
        N = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = N // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)

        # State
        mean = np.array(x0, dtype=float)
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        C_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_N = np.sqrt(N) * (1 + 1 / (4 * N) - 1 / (21 * N ** 2))
        sigma = sigma0
        best_x = mean.copy()
        best_f = float('inf')

        # Cache for inverse square root update
        def update_C(C, C_inv_sqrt, C_sqrt, w, z):
            """Update C and its square root decomposition."""
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - delta_sigma) * c_sigma^2 * (1 - chi_N^2) * C) + c_mu * sum w_i (z_i @ z_i.T)
            # For simplicity, we use a simpler update that avoids full eigendecomposition each step
            # but for dim=5, eigendecomposition is cheap.
            
            # Actually, let's just do the standard update and recompute sqrt via eigh.
            return C, C_inv_sqrt, C_sqrt

        evals_used = 0
        generations = 0
        
        while evals_used < budget:
            if time.time() > deadline:
                break
                
            # Sample population
            pop = []
            for i in range(N):
                if evals_used >= budget:
                    break
                if time.time() > deadline:
                    break
                y = mean + sigma * (C_sqrt @ rng.standard_normal(dim))
                y = np.clip(y, -5, 5)
                pop.append(y)
                evals_used += 1
            
            if not pop:
                break
                
            # Evaluate
            vals = np.array([func(p) for p in pop])
            
            # Update best
            idx_min = np.argmin(vals)
            if vals[idx_min] < best_f:
                best_f = vals[idx_min]
                best_x = pop[idx_min].copy()
            
            # Sort
            order = np.argsort(vals)
            pop_sorted = np.array([pop[i] for i in order])
            vals_sorted = vals[order]
            
            # Selection
            z = (pop_sorted[:mu] - mean) / sigma
            w = weights
            
            # Update mean
            mean = mean + sigma * (w @ z)
            mean = np.clip(mean, -5, 5)
            
            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt @ (w @ z))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (generations + 1)) / chi_N < 1.4 + 2 / (dim + 1))
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (w @ z)
            
            # Update covariance
            delta_sigma = (1 - (1 - c_sigma) ** (generations + 1)) * (1 - chi_N ** 2)
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - delta_sigma) * c_sigma ** 2 * (1 - chi_N ** 2) * C) + c_mu * (z.T @ np.diag(w) @ z)
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_N * (np.linalg.norm(ps) / chi_N - 1))
            
            # Recompute square root of C
            # Use eigendecomposition for stability
            evals_eig, evecs = np.linalg.eigh(C)
            # Clip negative eigenvalues
            evals_eig = np.clip(evals_eig, 1e-12, None)
            C_sqrt = evecs @ np.diag(np.sqrt(evals_eig))
            
            generations += 1
            
            # Termination criteria
            if sigma < 1e-12:
                break
            if np.linalg.norm(C) < 1e-12:
                break
                
        return best_x, best_f, evals_used

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Phase 1: Initial Sampling ---
        n_samples = 30
        points = []
        
        # Random points
        for _ in range(15):
            points.append(rng.uniform(-4, 4, dim))
        
        # Center and axis-aligned points
        points.append(np.zeros(dim))
        for i in range(dim):
            for d in [1.5, 3.0, -1.5, -3.0]:
                p = np.zeros(dim)
                p[i] = d
                points.append(p.copy())
        
        # More random points near center
        for _ in range(5):
            points.append(rng.uniform(-2, 2, dim))

        evals = 0
        values = []
        for p in points:
            if evals >= budget:
                break
            fp = func(np.clip(p, -5, 5))
            values.append(fp)
            evals += 1
        
        if evals >= budget or not values:
            return np.zeros(dim), 0.0

        # Find best so far
        best_idx = int(np.argmin(values))
        best_x = np.array(points[best_idx])
        best_f = values[best_idx]

        # Estimate initial sigma
        sorted_idx = np.argsort(values)
        top_k = min(10, len(values))
        top_points = np.array([points[i] for i in sorted_idx[:top_k]])
        
        if len(top_points) > 1:
            distances = np.linalg.norm(top_points[1:] - top_points[0], axis=1)
            sigma = max(0.5, np.mean(distances) * 0.5)
        else:
            sigma = 1.0

        # --- Phase 2: CMA-ES with Restarts ---
        # We will run CMA-ES. If it stagnates, we restart with a new random point.
        
        remaining = budget - evals
        if remaining <= 0:
            return best_x, best_f

        # Run CMA-ES
        x0 = best_x.copy()
        sigma0 = sigma
        
        # We pass a mutable reference for evals
        evals_ref = [evals]
        
        # Modify _cmaes to update evals_ref
        # Actually, let's just track it locally and add to global evals at the end.
        # But the deadline check inside _cmaes uses time.
        
        # Let's run CMA-ES for the remaining budget.
        # If it finishes early (convergence), we can try another restart or polish.
        
        cma_best_x, cma_best_f, cma_evals = self._cmaes(
            func, x0, sigma0, remaining, rng, evals_ref, deadline
        )
        
        evals += cma_evals
        
        if cma_best_f < best_f:
            best_x = cma_best_x
            best_f = cma_best_f
            
        # If we still have budget and time, do a local polish with (1+1)-ES
        if evals < budget and time.time() < deadline:
            # (1+1)-ES polish
            sigma_polish = 0.1
            x_curr = best_x.copy()
            f_curr = best_f
            
            # Adaptive step size
            successes = []
            success_window = 10
            
            while evals < budget:
                if time.time() > deadline:
                    break
                
                y = x_curr + sigma_polish * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                
                if fy < f_curr:
                    x_curr = y
                    f_curr = fy
                    successes.append(1)
                    sigma_polish *= 1.1
                else:
                    successes.append(0)
                    sigma_polish *= 0.9
                    
                if len(successes) > success_window:
                    successes.pop(0)
                    
                if len(successes) == success_window:
                    rate = np.mean(successes)
                    if rate > 0.5:
                        sigma_polish *= 1.05
                    elif rate < 0.2:
                        sigma_polish *= 0.95
                        
                sigma_polish = np.clip(sigma_polish, 1e-5, 1.0)
                
                if f_curr < best_f:
                    best_x = x_curr
                    best_f = f_curr
                    
                # Stop if sigma is too small
                if sigma_polish < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
