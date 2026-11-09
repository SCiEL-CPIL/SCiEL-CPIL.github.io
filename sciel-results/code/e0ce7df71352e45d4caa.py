import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_y = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def call_func(x):
            nonlocal calls_made, best_x, best_y
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5, 5)
            y = func(x)
            calls_made += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        def sobol_sample(n, dim, skip=0):
            # Simple deterministic low-discrepancy sequence approximation using Halton
            # Halton is easier to implement correctly for small dims without scipy dependency issues in some envs
            # But the prompt says scipy is available. Let's use scipy.stats.qmc.Sobol if possible, else fallback.
            # Actually, implementing a simple Halton is safer and fast.
            points = []
            for k in range(skip, skip + n):
                x = []
                for i in range(dim):
                    # Prime numbers for Halton
                    primes = [2, 3, 5, 7, 11]
                    p = primes[i]
                    f = 1.0
                    r = 0.0
                    kk = k + 1
                    while kk > 0:
                        f /= p
                        r += f * (kk % p)
                        kk //= p
                    x.append(r)
                points.append(x)
            return np.array(points)

        # Phase 1: Initial sampling using Halton/Sobol-like sequence for uniform coverage
        # This provides better early progress than random uniform
        n_init = 20
        # Map [0,1] to [-5,5]
        init_points = sobol_sample(n_init, d) * 10.0 - 5.0
        
        # Add center and axis points for robustness
        center = np.zeros(d)
        init_points = np.vstack([init_points, center])
        
        for x in init_points:
            if not time_ok():
                break
            call_func(x)

        if best_x is None:
            # Fallback if something went wrong
            call_func(np.zeros(d))
            if best_x is None:
                return np.zeros(d), np.inf

        # Phase 2: CMA-ES for global/mid-level optimization
        def cma_es(mean, sigma, max_calls=None):
            nonlocal best_x, best_y, calls_made
            
            if max_calls is None:
                max_calls = budget - calls_made
            
            n = d
            lam = int(4 + 3 * np.log(n))
            mu = lam // 2
            
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1)/(n + 1)) - 1) + c_sigma
            
            mean = np.array(mean, dtype=float).copy()
            sigma = float(sigma)
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            def update_sqrt_C():
                nonlocal C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                return eigvecs @ np.diag(np.sqrt(eigvals))
            
            sqrt_C = update_sqrt_C()
            
            gen = 0
            while calls_made < max_calls and time_ok() and gen < 100:
                pop = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    x = mean + sigma * (sqrt_C @ z)
                    pop.append(x)
                
                fvals = []
                for x in pop:
                    y = call_func(x)
                    if y is None:
                        return
                    fvals.append(y)
                
                idx = np.argsort(fvals)
                pop = [pop[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                new_mean = np.sum([weights[i] * pop[i] for i in range(mu)], axis=0)
                
                pc = (1 - c_c) * pc + c_c * (np.linalg.inv(sqrt_C) @ (new_mean - mean) / sigma)
                
                delta_hs = 1.0 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(gen + 1)) < (1.4 + 2/(n+1)) * chi_n else 0.0
                ps = (1 - c_sigma) * ps + delta_hs * np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (np.linalg.inv(sqrt_C) @ (new_mean - mean) / sigma)
                
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) - C)
                C += c_mu * sum(weights[i] * np.outer(pop[i] - mean, pop[i] - mean) for i in range(mu))
                
                C = (C + C.T) / 2
                sqrt_C = update_sqrt_C()
                
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                mean = new_mean
                gen += 1
        
        # Phase 3: Nelder-Mead for local refinement
        def nelder_mead(x0, max_calls):
            nonlocal best_x, best_y, calls_made
            
            n = d
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                # Adaptive step size based on distance to boundary
                dist_to_bound = min(5 - x0[i], x0[i] + 5)
                step = 0.1 * max(0.1, dist_to_bound)
                xi[i] += step
                simplex.append(np.clip(xi, -5, 5))
            
            fvals = []
            for x in simplex:
                fx = call_func(x)
                if fx is None:
                    return
                fvals.append(fx)
            
            max_iter = 30
            for _ in range(max_iter):
                if not time_ok() or calls_made >= max_calls:
                    break
                
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if fvals[0] - fvals[-1] < 1e-10:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = call_func(xr)
                if fxr is None:
                    break
                
                if fxr < fvals[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = call_func(xe)
                    if fxe is None:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        fvals[-1] = fxe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fxr
                elif fxr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fxr
                else:
                    xc = centroid + sigma_nm * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = call_func(xc)
                    if fxc is None:
                        break
                    if fxc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fxi = call_func(simplex[i])
                            if fxi is None:
                                return
                            fvals[i] = fxi
                        break

        # Execute Strategy
        # 1. CMA-ES with moderate sigma
        cma_es(best_x, 1.5)
        
        # 2. Nelder-Mead refinement
        remaining = budget - calls_made
        if remaining > 10:
            nelder_mead(best_x, min(remaining, 50))
            
        # 3. If budget remains, do a second CMA-ES pass with smaller sigma
        remaining = budget - calls_made
        if remaining > 20:
            cma_es(best_x, 0.5, max_calls=min(remaining, 80))
            
        # 4. Final Nelder-Mead polish
        remaining = budget - calls_made
        if remaining > 5:
            nelder_mead(best_x, min(remaining, 30))

        return best_x, best_y
