import scipy
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        # Parameters
        n = self.dim
        budget = self.budget
        
        # Reserve 50 evaluations for Nelder-Mead polish
        polish_budget = 50
        cma_budget = budget - polish_budget
        
        # Initialize CMA-ES
        # Start at 0, which is usually a good guess for BBOB
        x0 = np.zeros(n)
        fx0 = func(x0)
        best_x = x0.copy()
        best_f = fx0
        
        lambda_ = 9  # Population size for dim 5
        mu = lambda_ // 2
        weights = np.log(lambda_ / 2 + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(n)
        # Pre-compute eigen decomposition for sampling? 
        # For n=5, Cholesky is fast enough.
        
        # Counters
        cs = 0.0
        ps = np.zeros(n)
        c1 = (2 / ((n + 1)**2 + 10)) / mueff
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        cs = 1.0 / (5 * n + 5)
        damps = 0.5 + 2 / (n + 1) + cmu / 2
        
        # CMA-ES Loop
        calls_made = 1
        gen = 0
        
        while calls_made < cma_budget and (time.time() - start_time) < deadline:
            # Sample
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset C
                C = np.eye(n)
                L = np.eye(n)
                
            z = np.random.randn(lambda_, n)
            ys = np.array([z[i] @ L.T for i in range(lambda_)])
            xs = np.array([x0 + sigma * y for y in ys])
            
            # Evaluate
            fs = np.array([func(x) for x in xs])
            calls_made += lambda_
            
            # Sort
            order = np.argsort(fs)
            xs = xs[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
            
            # Update Mean
            y_mean = np.zeros(n)
            for i in range(mu):
                y_mean += weights[i] * (xs[i] - x0) / sigma
            
            # Update ps (for sigma adaptation)
            hsig = (1 - (1 - cs)**(2 * (gen + 1)))**0.5
            norm_c_sqrt_det_C = np.linalg.norm(L) # Approximation, ideally log det
            # Correct formula: hsig = 1 - (1-cs)^(2*(gen+1))
            # And check ||ps|| / sqrt(1 - (1-cs)^(2*(gen+1))) < (1.4 + 2/(n+1)) * E||N(0, I)||
            # E||N(0, I)|| = sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
            en = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            ps = (1 - cs) * ps + hsig * np.sqrt(mueff) * y_mean
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * en:
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / en - 1))
            
            # Update C
            # y_diff = xs[:mu] - x0
            # Delta = sum(w_i * outer(y_diff_i, y_diff_i))
            # C = (1 - c1 - cmu) * C + c1 * (outer(ps, ps) + (1 - hsig) * cs * (1 - cs) * C) + cmu * Delta
            # Note: The term (1-hsig)*cs*(1-cs)*C is often simplified or omitted in basic implementations, 
            # but for correctness:
            delta = np.zeros((n, n))
            for i in range(mu):
                y_diff = (xs[i] - x0) / sigma
                delta += weights[i] * np.outer(y_diff, y_diff)
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(ps, ps) + (1 - hsig) * cs * (1 - cs) * C) + cmu * delta
            
            # Update Mean
            x0 = x0 + sigma * y_mean
            
            gen += 1
            
            # Check time
            if (time.time() - start_time) > deadline - 10:
                break

        # Nelder-Mead Polish
        # Use scipy.optimize.minimize with Nelder-Mead
        # We have `polish_budget` calls left.
        # NM typically uses n+1 points per iteration.
        # Max iterations = polish_budget // (n + 1)
        
        if calls_made < budget and (time.time() - start_time) < deadline:
            # Define a counter to limit function calls
            call_count = [0]
            def func_counted(x):
                call_count[0] += 1
                return func(x)
                
            # Initial simplex around best_x
            # Use a small step size
            step = 0.1 * np.ones(n)
            simplex = [best_x + step]
            for i in range(n):
                pt = best_x.copy()
                pt[i] += step[i]
                simplex.append(pt)
            
            # Evaluate initial simplex
            fvals = [func_counted(pt) for pt in simplex]
            calls_made += len(simplex)
            
            # Manual Nelder-Mead to control budget strictly? 
            # Or use scipy if it respects a maxiter? 
            # scipy.optimize.minimize does not accept a 'budget' directly, but maxiter.
            # maxiter = polish_budget // (n + 1)
            
            try:
                res = scipy.optimize.minimize(
                    func_counted, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': polish_budget // (n + 1), 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
