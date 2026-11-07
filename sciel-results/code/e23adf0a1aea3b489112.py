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
        sigma = 2.0
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
        chiN = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        # Main CMA-ES Loop
        while n_evals + lam <= budget and _check_time():
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
            
            # Update covariance using rank-mu via mean shift (more stable)
            delta_hebbian = c1 * (np.outer(ps, ps) + (1 - hsig) * (cc * (2 - cc) * np.outer(pc, pc)))
            
            # Rank-mu update: use the weighted mean of (Y_sorted[:mu] - y_old)
            # This is equivalent to sum_i w_i * outer(y_i - y_old, y_i - y_old) / sigma^2
            # but computed more efficiently
            mu_shift = np.zeros(d)
            for i in range(mu):
                mu_shift += w[i] * (Y_sorted[i] - y_old)
            # delta_hebbian += cmu/sigma^2 * outer(mu_shift, mu_shift) + correction
            # Actually, the standard rank-mu uses:
            # sum_i w_i * outer(y_i - y_old, y_i - y_old)
            # Let's compute it properly but efficiently
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                diff = (Y_sorted[i] - y_old) / sigma
                rank_mu += w[i] * np.outer(diff, diff)
            delta_hebbian += cmu * rank_mu
            
            C = (1 - c1 - cmu) * C + delta_hebbian
            C = (C + C.T) / 2
            
            # Eigen decomposition
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D2, 1e-20, None))
            
            # Step size adaptation
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Convergence check
            if sigma < 1e-5:
                break

        # Phase 2: Local Refinement with adaptive step sizes
        if best_x is not None and n_evals < budget and _check_time():
            x = best_x.copy()
            f_cur = best_f
            
            # Adaptive coordinate descent with multiple scales
            # Start with larger steps and shrink
            step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            
            for ss in step_sizes:
                if n_evals >= budget or not _check_time():
                    break
                improved = True
                while improved and n_evals < budget and _check_time():
                    improved = False
                    for i in range(d):
                        if n_evals >= budget or not _check_time():
                            break
                        # Try both directions
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
                                break  # Restart coordinate loop if improved
        
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            n_evals += 1
            
        return best_x, best_f
