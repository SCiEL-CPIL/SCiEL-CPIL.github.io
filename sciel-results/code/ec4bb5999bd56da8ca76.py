"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """GP-EI optimizer for Katsuura-style multimodal function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = np.zeros(n)
        best_f = float('inf')
        total_calls = 0

        # GP Implementation
        class GP:
            def __init__(self, alpha=1e-8):
                self.alpha = alpha
                self.X = None
                self.y = None
                self.L = None
                self.beta = None
                self.y_mean = 0
                self.y_std = 1

            def _kernel(self, X1, X2):
                # Matern 5/2 kernel
                d = np.sqrt(np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1))
                sqrt5d = np.sqrt(5.0) * d
                return np.exp(-sqrt5d) * (1.0 + sqrt5d + sqrt5d**2 / 3.0)

            def fit(self, X, y):
                self.X = X
                self.y = y
                self.y_mean = np.mean(y)
                self.y_std = np.std(y)
                if self.y_std < 1e-10:
                    self.y_std = 1e-10
                K = self._kernel(X, X) + self.alpha * np.eye(len(X))
                self.L = np.linalg.cholesky(K)
                self.beta = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y - self.y_mean))

            def predict(self, X_new):
                if self.X is None:
                    return np.zeros(len(X_new)), np.zeros(len(X_new))
                K_s = self._kernel(self.X, X_new)
                K_ss = self._kernel(X_new, X_new)
                mu = self.y_mean + K_s.T @ self.beta
                v = np.linalg.solve(self.L, K_s.T)
                var = np.diag(K_ss) - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-10)
                return mu, np.sqrt(var)

            def ei(self, X_new, f_best, xi=0.01):
                mu, sigma = self.predict(X_new)
                z = (f_best - mu - xi) / sigma
                ei = (f_best - mu - xi) * self._norm_cdf(z) + sigma * self._norm_pdf(z)
                return ei

            def _norm_cdf(self, x):
                return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))

            def _norm_pdf(self, x):
                return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

        gp = GP()
        
        # Phase 1: Initial Sampling (Latin Hypercube)
        # Use 50% of budget for global search via GP
        n_init = min(50, budget // 4)
        if n_init < 10:
            n_init = 10
            
        # Generate LHS points
        X_init = np.zeros((n_init, n))
        for j in range(n):
            X_init[:, j] = np.linspace(-5, 5, n_init, endpoint=False) + (1.0 / n_init) * rng.uniform(size=n_init)
            rng.shuffle(X_init[:, j])
        
        y_init = np.zeros(n_init)
        for i in range(n_init):
            if total_calls >= budget or time.time() > deadline:
                break
            x = X_init[i]
            y = func(x)
            total_calls += 1
            y_init[i] = y
            if y < best_f:
                best_f = y
                best_x = x.copy()

        # Fit initial GP
        X_gp = X_init[:total_calls-1] # adjust for potential early break
        y_gp = y_init[:total_calls-1]
        if len(X_gp) > 1:
            gp.fit(X_gp, y_gp)

        # Phase 2: GP-EI Optimization
        # Reserve some budget for local polish
        polish_budget = min(50, budget // 10)
        gp_budget = budget - total_calls - polish_budget
        
        # If gp_budget is too small, reduce polish
        if gp_budget < 20:
            polish_budget = 10
            gp_budget = budget - total_calls - polish_budget

        # EI parameters
        xi = 0.01
        n_candidates = 2000
        
        for step in range(gp_budget):
            if total_calls >= budget or time.time() > deadline:
                break
            
            # Generate candidate points for EI
            # Mix of random points and points near current best
            X_rand = rng.uniform(-5, 5, (n_candidates, n))
            X_near = best_x + rng.normal(0, 0.5, (n_candidates, n))
            X_near = np.clip(X_near, -5, 5)
            X_cand = np.vstack([X_rand, X_near])
            
            # Compute EI
            e = gp.ei(X_cand, best_f, xi)
            
            # Select best candidate
            idx = np.argmax(e)
            x_next = X_cand[idx]
            
            # Evaluate
            y_next = func(x_next)
            total_calls += 1
            
            # Update best
            if y_next < best_f:
                best_f = y_next
                best_x = x_next.copy()
            
            # Update GP
            X_gp = np.vstack([X_gp, x_next])
            y_gp = np.append(y_gp, y_next)
            
            # Refit GP (can be expensive, so we do it every step but efficiently)
            try:
                gp.fit(X_gp, y_gp)
            except np.linalg.LinAlgError:
                # If GP fails, fall back to random search
                for _ in range(10):
                    if total_calls >= budget or time.time() > deadline:
                        break
                    x_rand = rng.uniform(-5, 5, n)
                    y_rand = func(x_rand)
                    total_calls += 1
                    if y_rand < best_f:
                        best_f = y_rand
                        best_x = x_rand.copy()
                break

        # Phase 3: Local Polish with Nelder-Mead
        remaining = budget - total_calls
        if remaining > 10 and time.time() < deadline:
            # Use Nelder-Mead starting from best_x
            x_nm = best_x.copy()
            y_nm = best_f
            
            # Initial simplex
            simplex = [x_nm.copy()]
            for j in range(n):
                x_v = x_nm.copy()
                x_v[j] += 0.1
                simplex.append(x_v)
            
            y_simplex = [y_nm]
            for i in range(1, n+1):
                if total_calls >= budget or time.time() > deadline:
                    break
                y_i = func(simplex[i])
                total_calls += 1
                y_simplex.append(y_i)
                if y_i < best_f:
                    best_f = y_i
                    best_x = simplex[i].copy()
            
            # Nelder-Mead iterations
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            max_iter = remaining
            for _ in range(max_iter):
                if total_calls >= budget or time.time() > deadline:
                    break
                
                # Sort simplex
                order = np.argsort(y_simplex)
                simplex = [simplex[i] for i in order]
                y_simplex = [y_simplex[i] for i in order]
                
                # Centroid of all but worst
                x_c = np.mean(simplex[:n], axis=0)
                
                # Reflection
                x_r = x_c + alpha * (x_c - simplex[n])
                x_r = np.clip(x_r, -5, 5)
                y_r = func(x_r)
                total_calls += 1
                if y_r < best_f:
                    best_f = y_r
                    best_x = x_r.copy()
                
                if y_simplex[0] <= y_r < y_simplex[n-1]:
                    simplex[n] = x_r
                    y_simplex[n] = y_r
                elif y_r < y_simplex[0]:
                    # Expansion
                    x_e = x_c + gamma * (x_r - x_c)
                    x_e = np.clip(x_e, -5, 5)
                    if total_calls < budget and time.time() < deadline:
                        y_e = func(x_e)
                        total_calls += 1
                        if y_e < best_f:
                            best_f = y_e
                            best_x = x_e.copy()
                        if y_e < y_r:
                            simplex[n] = x_e
                            y_simplex[n] = y_e
                        else:
                            simplex[n] = x_r
                            y_simplex[n] = y_r
                    else:
                        simplex[n] = x_r
                        y_simplex[n] = y_r
                else:
                    # Contraction
                    x_cw = x_c + rho * (simplex[n] - x_c)
                    x_cw = np.clip(x_cw, -5, 5)
                    if total_calls < budget and time.time() < deadline:
                        y_cw = func(x_cw)
                        total_calls += 1
                        if y_cw < best_f:
                            best_f = y_cw
                            best_x = x_cw.copy()
                        if y_cw < y_simplex[n]:
                            simplex[n] = x_cw
                            y_simplex[n] = y_cw
                        else:
                            # Shrink
                            for i in range(1, n+1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                if total_calls < budget and time.time() < deadline:
                                    y_i = func(simplex[i])
                                    total_calls += 1
                                    if y_i < best_f:
                                        best_f = y_i
                                        best_x = simplex[i].copy()
                                    y_simplex[i] = y_i
                    else:
                        simplex[n] = x_cw
                        y_simplex[n] = y_cw

        return best_x, best_f
# EVOLVE-BLOCK-END
