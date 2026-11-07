import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        x = np.zeros(self.dim)
        best_x = x.copy()
        best_f = float('inf')
        
        calls_made = 0
        
        def evaluate(pt):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return best_f
            if time.time() > deadline:
                return best_f
            val = func(pt)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = pt.copy()
            return val

        # Phase 1: Estimate gradient and Hessian using finite differences
        # For sphere function f(x) = (x-x*)^T A (x-x*) + c, we can estimate A and gradient
        
        # First, get a good initial point using a few random samples
        rng = np.random.default_rng(42)
        
        # Sample a few points to find a reasonable starting location
        n_init = 10
        candidates = [np.zeros(self.dim)]
        for _ in range(n_init - 1):
            candidates.append(rng.uniform(-4, 4, self.dim))
        
        f_vals = [evaluate(c) for c in candidates]
        min_idx = int(np.argmin(f_vals))
        x = candidates[min_idx].copy()
        fx = f_vals[min_idx]
        
        # If we haven't used too many calls, estimate gradient and Hessian
        if calls_made < self.budget - 50:
            # Estimate gradient using central differences
            h = 0.1
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                fp = evaluate(xp)
                fm = evaluate(xm)
                grad[i] = (fp - fm) / (2 * h)
            
            # Estimate Hessian using second-order central differences
            # For sphere, Hessian is constant (2*A)
            hess = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(i, self.dim):
                    xpp = x.copy()
                    xpm = x.copy()
                    xmp = x.copy()
                    xmm = x.copy()
                    xpp[i] += h
                    xpp[j] += h
                    xpm[i] += h
                    xpm[j] -= h
                    xmp[i] -= h
                    xmp[j] += h
                    xmm[i] -= h
                    xmm[j] -= h
                    
                    fpp = evaluate(xpp)
                    fpm = evaluate(xpm)
                    fmp = evaluate(xmp)
                    fmm = evaluate(xmm)
                    
                    hess_val = (fpp - fpm - fmp + fmm) / (4 * h * h)
                    hess[i, j] = hess_val
                    hess[j, i] = hess_val
            
            # Solve for minimizer: x* = x - H^{-1} g
            # For sphere, this should give the exact minimizer
            try:
                # Add small regularization for stability
                reg = 1e-8 * np.eye(self.dim)
                x_opt = x - np.linalg.solve(hess + reg, grad)
                x_opt = np.clip(x_opt, -5, 5)
                f_opt = evaluate(x_opt)
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse
                try:
                    x_opt = x - np.linalg.pinv(hess) @ grad
                    x_opt = np.clip(x_opt, -5, 5)
                    f_opt = evaluate(x_opt)
                except Exception:
                    pass
        
        # Phase 2: Polish with adaptive step size search
        x = best_x.copy()
        fx = best_f
        
        # Use a more sophisticated local search
        sigma = 0.5
        rng2 = np.random.default_rng(0)
        
        while calls_made < self.budget and time.time() < deadline:
            # Generate multiple candidates with varying step sizes
            candidates = []
            for scale in [sigma, sigma * 0.5, sigma * 2]:
                step = scale * rng2.standard_normal(self.dim)
                candidates.append(np.clip(x + step, -5, 5))
            
            # Evaluate all candidates
            best_candidate = None
            best_candidate_f = fx
            
            for cand in candidates:
                f_cand = evaluate(cand)
                if f_cand < best_candidate_f:
                    best_candidate_f = f_cand
                    best_candidate = cand
            
            if best_candidate is not None:
                x = best_candidate
                fx = best_candidate_f
                sigma *= 1.3  # Increase on success
            else:
                sigma *= 0.7  # Decrease on failure
            
            if sigma < 1e-10:
                break
        
        return best_x, best_f
