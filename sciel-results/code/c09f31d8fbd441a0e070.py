"""CMA-ES with Sobol seeding for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Sobol-seeded CMA-ES optimizer for continuous black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_sequence(self, n_samples, dim):
        """Generate Sobol sequence in [0,1]^dim using scipy."""
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Need power of 2, so round up
        m = int(np.ceil(np.log2(max(n_samples, 2))))
        n_pow2 = 2 ** m
        points = sampler.random(n_pow2)
        return points[:n_samples]

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        n = dim
        
        # Phase 1: Sobol-based global sampling for early coverage
        # Use ~20% of budget for initial exploration
        n_sobol = min(80, max(20, budget // 5))
        
        # Generate Sobol points in [0,1]^n, map to [-5, 5]^n
        sobol_pts = self._sobol_sequence(n_sobol, n)
        sobol_pts = sobol_pts * 10.0 - 5.0  # map to [-5, 5]
        
        # Evaluate Sobol points
        best_x = sobol_pts[0].copy()
        best_f = float('inf')
        calls = 0
        all_points = []
        all_vals = []
        
        for i in range(n_sobol):
            if time.time() - start_time > deadline:
                break
            x = sobol_pts[i]
            f = func(x)
            calls += 1
            all_points.append(x.copy())
            all_vals.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if calls >= budget or time.time() - start_time > deadline:
            return best_x, best_f
        
        # Phase 2: CMA-ES starting from the best Sobol point
        # Initialize CMA-ES with the best point as mean
        m = best_x.copy()
        
        # Estimate initial sigma from spread of good points
        good_mask = np.array(all_vals) < (best_f + 5.0 * (np.max(all_vals) - best_f + 1e-10))
        if np.sum(good_mask) >= 2:
            good_pts = np.array([all_points[i] for i in range(len(all_points)) if good_mask[i]])
            sigma = np.std(good_pts, axis=0).mean()
            sigma = max(sigma, 0.5)
        else:
            sigma = 1.0
        
        # CMA-ES Parameters
        lam = max(8, 4 + int(3 * np.log(n)))
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 2) / (mueff + 2) - 1) + c_sigma)
        c_c = (4 + mueff / n) / (n + 4 + mueff)
        c_mu = 2 * ((mueff - 2 + 2 / mueff) * c_c - 1) / (n + 2)
        
        # Initial state
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        w_mean = weights
        
        best_x_cma = best_x.copy()
        best_f_cma = best_f
        
        # Track generation for h_sigma calculation
        gen = 0
        
        while calls < budget:
            # Check time
            if time.time() - start_time > deadline:
                break
            
            gen += 1
            
            # Sample new population
            X = np.zeros((lam, n))
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(n)
                C = np.eye(n)
            
            for i in range(lam):
                if calls >= budget:
                    break
                z = np.random.standard_normal(n)
                X[i] = m + sigma * (L @ z)
                calls += 1
            
            if calls >= budget:
                break
            
            # Evaluate remaining population members
            F = np.zeros(lam)
            for i in range(lam):
                if i < lam and calls < budget:
                    F[i] = func(X[i])
                    calls += 1
                else:
                    F[i] = float('inf')
            
            # Track global best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f_cma:
                best_f_cma = F[idx_min]
                best_x_cma = X[idx_min].copy()
            
            # Sort
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(w_mean[:, None] * X_sorted[:mu], axis=0)
            
            # Update evolution paths
            y = (m - m_old) / sigma
            # h_sigma: check if evolution path is consistent with covariance
            chi_n = np.sqrt(n) * (1 + 1.0 / (2 * n) - 1.0 / (8 * n**2))
            # Use a simpler h_sigma approximation
            h_sigma = (np.linalg.norm(y) / chi_n < (1.4 + 2.0 / (n + 1)))
            h_sigma_val = 1.0 if h_sigma else 0.0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma_val * np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y
            p_c = (1 - c_c) * p_c + h_sigma_val * np.sqrt(c_c * (2 - c_c) * mueff) * y
            
            # Update step size
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e6)
            
            # Update covariance
            delta_h = (1 - h_sigma_val) * c_c * (2 - c_c)
            # C = (1 + c_mu * (delta_h - 1)) * C + c_mu * sum(w_i * outer(y_i, y_i))
            C = (1 + c_mu * (delta_h - 1)) * C
            for i in range(mu):
                Y_i = (X_sorted[i] - m_old) / sigma
                C += c_mu * weights[i] * np.outer(Y_i, Y_i)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Ensure C is positive definite
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C = C + 1e-12 * np.eye(n)
            
            # Check convergence: if sigma is very small, stop
            if sigma < 1e-8:
                break
        
        # Phase 3: Final polish with Nelder-Mead on the best point
        if calls < budget and time.time() - start_time < deadline:
            try:
                from scipy.optimize import minimize
                remaining = budget - calls
                if remaining >= 10:
                    # Use a small Nelder-Mead polish
                    def _wrap(x):
                        nonlocal calls
                        if calls >= budget:
                            return float('inf')
                        calls += 1
                        return func(x)
                    
                    # Nelder-Mead with limited iterations
                    res = minimize(
                        _wrap,
                        best_x_cma,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining, 100),
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_f_cma:
                        best_f_cma = res.fun
                        best_x_cma = res.x.copy()
            except Exception:
                pass
        
        return best_x_cma, best_f_cma
# EVOLVE-BLOCK-END
