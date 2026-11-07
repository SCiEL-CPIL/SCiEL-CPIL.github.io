import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0
        d = self.dim
        budget = self.budget
        
        def _check_time():
            return time.time() - start_time < timeout
        
        # CMA-ES Parameters
        # Lambda: population size
        lam = 4 + int(3 * np.log(d))
        # For d=5, lam = 4 + 3*1.609 = 8.8 -> 8? 
        # Inspiration 1 used 12. 
        # 4 + 3 * ln(5) = 4 + 4.82 = 8.82. int(8.82) = 8.
        # Wait, inspiration 1 code: lam = 4 + int(3 * np.log(d)).
        # np.log(5) is natural log. 3 * 1.609 = 4.827. int(4.827) = 4.
        # 4 + 4 = 8.
        # But the comment in Inspiration 1 said "Lambda 10 is a good choice" in Current, but Inspiration 1 code calculates it.
        # Let's check the score of Inspiration 1. It was 5.24.
        # The current program uses lam=10.
        # Let's use lam=10 or 12?
        # Standard CMA-ES suggests lam = 4 + 3 ln d.
        # For d=5, lam=8 or 9.
        # However, larger lambda can help with exploration.
        # Let's stick to lam=10 as in the current program which scored 5.0, 
        # but maybe the sigma was the issue.
        # Let's try lam=10, sigma=1.0.
        
        lam = 10
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff)))
        
        # Initialize
        x_mean = np.zeros(d)
        sigma = 1.0  # Reduced from 2.0 to 1.0
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        
        # PC and PS for path evolution
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        best_x = None
        best_f = np.inf
        
        n_evals = 0
        it = 0
        
        # Phase 1: CMA-ES
        # Use 70% of budget for CMA-ES
        cma_budget = int(0.7 * budget)
        
        while n_evals + lam <= cma_budget and _check_time():
            # Sample
            Z = np.random.randn(lam, d)
            Y = x_mean + sigma * (Z * D) @ B.T
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            n_evals += lam
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = Y[min_idx].copy()
                
            # Sort
            sorted_idx = np.argsort(F)
            Y_sorted = Y[sorted_idx]
            
            # Update mean
            y_old = x_mean.copy()
            x_mean = np.dot(w, Y_sorted[:mu])
            
            # Path evolution sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B.T @ (D * (x_mean - y_old)))
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (it + 1)))) < (1 + 2 / (d + 1)) * cc
            it += 1
            
            # Path evolution covariance
            if hsig:
                pc = (1 - cc) * pc + cc / sigma * (B.T @ (D * (x_mean - y_old)))
            
            # Update covariance
            delta_hebbian = c1 * (np.outer(ps, ps) + (1 - hsig) * (cc * (2 - cc) * np.outer(pc, pc)))
            
            for i in range(mu):
                delta_hebbian += (cmu * w[i]) * np.outer(Y_sorted[i] - y_old, Y_sorted[i] - y_old) / (sigma * sigma)
                
            C = (1 - c1 - cmu) * C + delta_hebbian
            C = (C + C.T) / 2
            
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D2, 1e-20, None))
            
            # Step size adaptation
            chiN = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Convergence check
            if sigma < 1e-5:
                break

        # Phase 2: Local Refinement
        if best_x is not None and n_evals < budget and _check_time():
            x = best_x.copy()
            f_cur = best_f
            
            # Use a shrinking step size approach
            # Start with a step size based on current sigma, but bounded
            initial_ss = max(sigma, 0.05)
            
            # Generate a list of step sizes to try
            # We want to cover from coarse to fine
            step_sizes = []
            ss = initial_ss
            while ss > 1e-4:
                step_sizes.append(ss)
                ss *= 0.5
            
            # Ensure we have a few very small steps at the end
            step_sizes.extend([1e-4, 1e-5, 1e-6])
            
            for ss in step_sizes:
                if n_evals >= budget or not _check_time():
                    break
                improved = True
                while improved and n_evals < budget and _check_time():
                    improved = False
                    for i in range(d):
                        for direction in [-1, 1]:
                            if n_evals >= budget or not _check_time():
                                break
                            x_try = x.copy()
                            x_try[i] += direction * ss
                            x_try = np.clip(x_try, -5, 5)
                            f_try = func(x_try)
                            n_evals += 1
                            if f_try < f_cur:
                                f_cur = f_try
                                x = x_try.copy()
                                improved = True
                                if f_cur < best_f:
                                    best_f = f_cur
                                    best_x = x.copy()
                                break # Restart loop for this ss if improved
                        if n_evals >= budget or not _check_time():
                            break
                    if n_evals >= budget or not _check_time():
                        break

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            n_evals += 1
            
        return best_x, best_f
