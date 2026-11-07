import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0
        call_count = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            return func(np.clip(x, -5.0, 5.0))

        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        if lam < 4: lam = 4
        # Number of recombination weights
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        # Variance-effectiveness
        mueff = 1.0 / np.sum(w**2)
        
        # Step size
        sigma = 2.0  # Start large to cover the box [-5, 5]^5
        
        # Mean vector
        m = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Inverse Cholesky decomposition of C
        # We need to sample N(0, C). We can use Cholesky of C.
        # Or sample N(0, I) and multiply by L where C = L L^T.
        
        # Adaptation parameters
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        c1 = (2 - 1 / (dim + 1.7) + 2 / (dim + 1) / mueff) / ((dim + 2) * (dim + 1.7 / mueff - 1))
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Initial Cholesky factor
        L = np.linalg.cholesky(C)
        
        best_x = m.copy()
        best_f = safe_func(best_x)
        
        # Main Loop
        while call_count < budget and time.time() < deadline:
            # Sample population
            pop = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.randn(dim)
                pop[i] = m + sigma * (L @ z)
            
            # Evaluate
            fvals = np.array([safe_func(pop[i]) for i in range(lam)])
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = pop[min_idx].copy()
                
            # Sort population
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(w[:, np.newaxis] * pop[:mu], axis=0)
            
            # Update sigma (CSA)
            x_sigma = np.linalg.solve(L, (m - m_old) / sigma)
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * x_sigma
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(2 * (call_count + 1) / (dim * cs))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mueff + 1) / (dim + 1)))
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))
            
            # Update C (CMA)
            x_c = np.linalg.solve(L, (m - m_old) / sigma)
            p_c = (1 - cc) * p_c + hsig * np.sqrt(cc * (2 - cc) * mueff) * x_c
            
            C = (1 - c1 - cmu) * C + c1 * ((1 - hsig) * (dim + 1 - cc - hsig * cc / (dim + 1)) * np.outer(p_c, p_c) + np.eye(dim))
            C += cmu * np.sum(w[:mu] * np.outer(pop[:mu] - m_old, pop[:mu] - m_old) / (sigma**2), axis=0)
            
            # Regularize C
            C = 0.5 * (C + C.T)
            
            # Update Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset C to identity and restart locally?
                # For now, just reset C to eye and continue, sigma might be too small/large
                C = np.eye(dim)
                L = np.eye(dim)
                sigma = max(sigma, 0.1)

        # Phase 2: Nelder-Mead Polish
        if call_count < budget and time.time() < deadline:
            x, fx = self._nelder_mead(best_x, best_f, safe_func, dim, deadline, budget - call_count)
            if fx < best_f:
                best_x = x
                best_f = fx

        return best_x, best_f

    def _nelder_mead(self, x0, f0, func, dim, deadline, max_calls):
        alpha = 0.05
        simplex = [x0.copy()]
        simplex_f = [f0]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += alpha
            xi = np.clip(xi, -5.0, 5.0)
            fi = func(xi)
            simplex.append(xi)
            simplex_f.append(fi)
            
        if len(simplex) > 1 and max(simplex_f) - min(simplex_f) < 1e-12:
            return x0, f0

        alpha_n = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_n = 0.5

        max_iterations = max_calls - dim
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            order = np.argsort(simplex_f)
            simplex = [simplex[i].copy() for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha_n * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = func(xr)
            
            if fr < simplex_f[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                if fr < simplex_f[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = func(xc)
                
                if fc < simplex_f[-1]:
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_n * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        simplex_f[i] = func(simplex[i])
                        
            if np.max(simplex_f) - np.min(simplex_f) < 1e-12:
                break

        best_idx = int(np.argmin(simplex_f))
        return simplex[best_idx].copy(), simplex_f[best_idx]
