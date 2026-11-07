"""Optimizer for BBOB-style f17 (Schaffers F7) using CMA-ES."""
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: kill at 280s, limit is 300s (evaluator) / 330s (hard kill)
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # lambda: population size
        lam = 4 + int(3 * np.log(dim))
        # mu: number of best solutions used for recombination
        mu = lam // 2
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        # Evolution path lengths
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Step size
        sigma = 2.0 # Initial step size for [-5, 5]^5
        
        # Initial mean
        mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Pre-compute C^{1/2}
        def get_C_half():
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            return eigvecs @ np.diag(np.sqrt(eigvals))
            
        C_half = get_C_half()
        
        count = 0
        best_x = mean.copy()
        best_f = float('inf')
        
        # Initial evaluation
        try:
            f = func(mean)
            count += 1
            best_f = f
            best_x = mean.copy()
        except Exception:
            pass

        if count >= budget or time.time() > deadline:
            return best_x, best_f

        # Main loop
        while count < budget:
            if time.time() > deadline:
                break
            
            # 1. Sample population
            Z = np.random.randn(lam, dim)
            X = (sigma * (C_half @ Z.T)).T + mean
            
            # 2. Evaluate
            F = np.zeros(lam)
            valid_count = 0
            for i in range(lam):
                if time.time() > deadline:
                    break
                try:
                    F[i] = func(X[i])
                    count += 1
                    valid_count += 1
                    if F[i] < best_f:
                        best_f = F[i]
                        best_x = X[i].copy()
                except Exception:
                    # If an exception occurs, we stop
                    return best_x, best_f
            
            if time.time() > deadline:
                break
                
            # If we didn't get all evaluations, we might have issues with sorting
            # But usually we get them all unless time runs out.
            # If time runs out, we break.
            
            # 3. Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            
            # 4. Update mean
            mean_old = mean.copy()
            mean = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # 5. Update sigma (CSA)
            # p_sigma = (1-cs) * p_sigma + sqrt(cs * (2-cs) * mueff) / sigma * (C^{-1/2} * (mean - mean_old))
            # C^{-1/2} is the inverse of C_half
            # We can compute C_inv_half = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
            # Or just solve C_half @ v = (mean - mean_old)
            try:
                delta = mean - mean_old
                # Solve C_half @ v = delta / sigma
                # v = C_half^{-1} @ delta / sigma
                # Since C_half is symmetric, C_half^{-1} = C_half.T^{-1}
                # We can use np.linalg.solve
                v = np.linalg.solve(C_half, delta / sigma)
            except np.linalg.LinAlgError:
                v = delta / sigma # Fallback
            
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * v
            
            # h_sigma = (||p_sigma|| / sqrt(1 - (1-cs)^{2*(count_gen+1)})) < c_h * (1 + cs)
            # Approximation: use threshold
            # count_gen is not tracked strictly, but we can use a large number for the denominator
            # Or simply check if ||p_sigma|| is too large
            # Standard implementation:
            # h_sigma = int( (np.linalg.norm(p_sigma) / np.sqrt(1.0 - (1.0-cs)**(2*(count/lam+1)))) < (1.4 + 2.0/(dim+1.0)) * (1.0 + cs) )
            # Let's use a simpler heuristic or the standard formula with generation count
            gen = count // lam
            denom = np.sqrt(1.0 - (1.0 - cs) ** (2 * (gen + 1)))
            if denom < 1e-12:
                denom = 1e-12
            h_sigma = 1 if (np.linalg.norm(p_sigma) / denom < (1.4 + 2.0 / (dim + 1.0)) * (1.0 + cs)) else 0
            
            # 6. Update p_c
            # p_c = (1-cc) * p_c + h_sigma * sqrt(cc * (2-cc) * mueff) * C^{-1/2} * (mean - mean_old)
            # We already have v = C^{-1/2} * (mean - mean_old) / sigma
            # So C^{-1/2} * (mean - mean_old) = v * sigma
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff) * (v * sigma)
            
            # 7. Update C
            # C = (1 - mu * (c1 + cmu)) * C + c1 * (p_c * p_c.T + (1 - h_sigma) * cc * (2 - cc) * C) + cmu * sum(w_i * (y_i * y_i.T))
            # where y_i = (x_i - mean_old) / sigma
            # Note: The term (1 - h_sigma) * cc * (2 - cc) * C is sometimes written differently.
            # Standard: C = (1 - c1 - cmu) * C + c1 * (p_c p_c.T + (1-h_sigma) cc (2-cc) C) + cmu * sum(w_i y_i y_i.T)
            
            # Calculate y_i
            Y = (X_sorted[:mu] - mean_old) / sigma
            # sum(w_i * y_i * y_i.T)
            # Y is (mu, dim), w is (mu,)
            # sum = (Y.T * w) @ Y
            rank_mu = (Y.T * w) @ Y
            
            # Update C
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * cc * (2 - cc) * C) + cmu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # 8. Update sigma
            # sigma = sigma * exp(cs / (2*lam) * (||p_sigma||^2 / dim - 1))
            sigma = sigma * np.exp((cs / (2 * lam)) * (np.linalg.norm(p_sigma) ** 2 / dim - 1))
            
            # 9. Update C_half
            C_half = get_C_half()
            
            # Optional: Restart if sigma becomes too small or too large?
            # With deadline guard, we might not need complex restarts.
            
        return best_x, best_f
