"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: CMA-ES for global basin search, then local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, n, dim):
        """Generate n Halton sequence points in [0,1]^dim, mapped to [-5,5]^dim."""
        points = []
        for i in range(1, n + 1):
            point = []
            for d in range(1, dim + 1):
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    i_val, rem = divmod(i_val, d)
                    f /= d
                    r += f * rem
                point.append(r)
            points.append(np.array(point) * 10.0 - 5.0)
        return points

    def _cma_es(self, func, x0, sigma0, budget, rng):
        """Simple CMA-ES implementation."""
        n = self.dim
        x = x0.copy()
        f_x = func(x)
        sigma = sigma0
        C = np.eye(n)
        pc = np.zeros(n)
        cs = np.zeros(n)
        
        lambda_pop = max(4, int(np.ceil(3 + 3 * np.log(n))))
        mu = lambda_pop // 2
        
        cc = (4 + lambda_pop / (n + 4)) / ((n + 2) + 2 * lambda_pop / (n + 4))
        cs = (2 + lambda_pop / (n + 2)) / (n + 2 + 2 + lambda_pop / (n + 2))
        cmu = min(1 - cc, 2 * (sqrt2 - 1)) if n > 2 else min(1 - cc, 0.5)
        
        sigma0 = sigma
        budget_used = 0
        best_x = x.copy()
        best_f = f_x
        
        while budget_used < budget:
            # Generate population
            Y = []
            for i in range(lambda_pop):
                B = np.linalg.cholesky(C)
                z = rng.standard_normal(n)
                y = x + sigma * B @ z
                y = np.clip(y, -5, 5)
                Y.append(y)
            
            # Evaluate
            F = [func(y) for y in Y]
            budget_used += lambda_pop
            
            # Sort by fitness
            idx = np.argsort(F)
            Y = [Y[i] for i in idx]
            F = [F[i] for i in idx]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()
            
            # Mean update
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            x_new = np.sum([w[i] * Y[i] for i in range(mu)], axis=0)
            
            # Update covariance
            d_sigma = (1 - np.exp(-cs * budget_used / (n + 1))) * np.linalg.norm(x_new - x) / sigma
            pc = (1 - cs) * pc + cs * (x_new - x) / sigma
            C = (1 - cs * (2 - cs) - cmu) * C + cmu * np.outer(pc, pc)
            
            # Update step size
            sigma *= np.exp(cs * (d_sigma / np.sqrt(n) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)
            
            x = x_new
            f_x = func(x)
            budget_used += 1
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()
            
            # Check convergence
            if np.linalg.norm(x - x0) < 1e-8 * (1 + np.linalg.norm(x0)):
                break
        
        return best_x, best_f, budget_used

    def _nelder_mead(self, func, x0, budget, rng):
        """Nelder-Mead simplex method."""
        n = self.dim
        # Initialize simplex
        simplex = [x0.copy()]
        for i in range(n):
            x_i = x0.copy()
            x_i[i] += 0.1 if x0[i] < 4.9 else -0.1
            simplex.append(x_i)
        
        F = [func(x) for x in simplex]
        budget_used = n + 1
        best_x = simplex[np.argmin(F)].copy()
        best_f = min(F)
        
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5   # contraction
        sigma = 0.5 # shrink
        
        while budget_used < budget:
            # Sort simplex
            idx = np.argsort(F)
            simplex = [simplex[i] for i in idx]
            F = [F[i] for i in idx]
            
            if F[0] < best_f:
                best_f = F[0]
                best_x = simplex[0].copy()
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            budget_used += 1
            
            if f_r < F[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                budget_used += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    F[-1] = f_e
                else:
                    simplex[-1] = x_r
                    F[-1] = f_r
            elif f_r < F[-2]:
                simplex[-1] = x_r
                F[-1] = f_r
            else:
                # Contraction
                if f_r < F[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                budget_used += 1
                if f_c < min(f_r, F[-1]):
                    simplex[-1] = x_c
                    F[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        F[i] = func(simplex[i])
                        budget_used += 1
                    if budget_used >= budget:
                        break
            
            # Check convergence
            if np.max(F) - np.min(F) < 1e-12:
                break
        
        idx = np.argmin(F)
        if F[idx] < best_f:
            best_f = F[idx]
            best_x = simplex[idx].copy()
        
        return best_x, best_f, budget_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 30s safety margin
        rng = np.random.default_rng(0)
        
        # Phase 1: Halton scan for initial coverage
        n_halton = min(20, self.budget // 5)
        halton_points = self._halton(n_halton, self.dim)
        
        best_x = None
        best_f = np.inf
        for x in halton_points:
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Also try center
        f_center = func(np.zeros(self.dim))
        if f_center < best_f:
            best_f = f_center
            best_x = np.zeros(self.dim)
        
        budget_used = n_halton + 1
        
        # Phase 2: CMA-ES from best point
        if time.time() < deadline and budget_used < self.budget:
            cma_budget = min(self.budget - budget_used - 10, 300)
            x_cma, f_cma, used = self._cma_es(func, best_x, 1.0, cma_budget, rng)
            budget_used += used
            if f_cma < best_f:
                best_f = f_cma
                best_x = x_cma
        
        # Phase 3: Nelder-Mead local refinement
        if time.time() < deadline and budget_used < self.budget:
            nm_budget = self.budget - budget_used
            x_nm, f_nm, used = self._nelder_mead(func, best_x, nm_budget, rng)
            budget_used += used
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm
        
        # Phase 4: Coordinate descent polish
        if time.time() < deadline and budget_used < self.budget:
            x = best_x.copy()
            f = best_f
            step = 0.5
            for _ in range(min(50, self.budget - budget_used)):
                improved = False
                for d in range(self.dim):
                    for sign in [1, -1]:
                        x_try = x.copy()
                        x_try[d] += sign * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        budget_used += 1
                        if f_try < f:
                            x = x_try
                            f = f_try
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break
            if f < best_f:
                best_f = f
                best_x = x
        
        return best_x, best_f
# EVOLVE-BLOCK-END
