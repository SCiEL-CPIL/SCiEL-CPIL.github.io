import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        N = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(N))  # Population size, e.g., 9 for N=5
        mu = lambda_ // 2                  # Number of parents, e.g., 4
        
        # Log-uniform weights for the top mu parents
        # w_i corresponds to the i-th best solution (1-indexed rank)
        # Rank 1 (best) gets log(lambda), Rank mu gets log(lambda - mu + 1)
        w = np.log(np.arange(lambda_, lambda_ - mu, -1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        # Strategy parameters
        cc = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N)
        cs = (mu_eff + 2) / (N + mu_eff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (N + 1)) - 1) + cs
        cmu = min(1 - cc, 2 * (mu_eff - 2 + 1 / mu_eff) / ((N + 2)**2 + mu_eff))
        
        # State
        m = np.zeros(N)  # Mean, start at center
        sigma = 1.5      # Initial step size
        C = np.eye(N)    # Covariance
        pc = np.zeros(N) # Evolution path for covariance
        ps = np.zeros(N) # Evolution path for step size
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Precompute chiN
        chiN = np.sqrt(N) * (1 - 1 / (2 * N**2) + 1 / (8 * N**4))
        
        for gen in range(int(budget / lambda_)):
            if time.time() > deadline:
                break
            
            # Compute Cholesky of current C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(N)
                L = np.eye(N)
                
            xs = np.zeros((lambda_, N))
            fs = np.zeros(lambda_)
            
            # Generate and evaluate offspring
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = np.random.standard_normal(N)
                y = L @ z
                x = m + sigma * y
                x = np.clip(x, -5, 5)
                
                fx = func(x)
                calls += 1
                xs[i] = x
                fs[i] = fx
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            if calls >= budget:
                break
            
            # Sort offspring by fitness
            idx = np.argsort(fs)
            xs = xs[idx]
            fs = fs[idx]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
            
            # Update Mean
            m_old = m.copy()
            m = np.sum(w[:, np.newaxis] * xs[:mu], axis=0)
            
            # Update Step Size (CSA)
            C_old = C.copy()
            
            # Calculate C^{-1/2} (m - m_old)
            v = m - m_old
            # Solve L w_temp = v, then L^T u = w_temp
            w_temp = np.linalg.solve(L, v)
            u = np.linalg.solve(L.T, w_temp)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * u
            
            # hsig condition
            norm_ps = np.linalg.norm(ps)
            denom = np.sqrt(1 - (1 - cs)**(2 * (gen + 1)))
            if denom < 1e-10:
                hsig = 0
            else:
                hsig = 1 if norm_ps / denom < (1.4 + 2 / (N + 1)) * chiN else 0
            
            sigma *= np.exp(cs / damps * (norm_ps / chiN - 1))
            
            # Update Covariance
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * u
            
            # Rank-1 and Rank-mu update
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            
            # Rank-mu term
            C_update_term = np.zeros((N, N))
            for i in range(mu):
                diff = xs[i] - m_old
                w_i_vec = np.linalg.solve(L, diff)
                v_i_vec = np.linalg.solve(L.T, w_i_vec)
                C_update_term += w[i] * np.outer(v_i_vec, v_i_vec)
            
            C = (1 + delta_hsig - cmu) * C_old + cmu * C_update_term + (1 + delta_hsig - cmu) * np.outer(pc, pc)
            # Wait, the formula is:
            # C <- (1 + delta_hsig - cmu) C + cmu * sum(...) + (1 + delta_hsig - cmu) * outer(pc, pc) ?
            # No, the standard update is:
            # C <- (1 + delta_hsig - cmu) C + cmu * sum_{i=1}^mu w_i (y_i - y_bar)(y_i - y_bar)^T
            # where y_i = C^{-1/2}(x_i - m_old)
            # AND the rank-1 update is applied to C BEFORE the rank-mu?
            # Actually, the combined update is:
            # C_new = (1 + delta_hsig - cmu) * C_old + cmu * C_mu_term + (1 + delta_hsig - cmu) * outer(pc, pc) ??
            
            # Let's check the standard CMA-ES paper (Hansen 2016).
            # C <- (1 + delta_hsig - cmu) C + cmu * sum w_i (y_i - y_bar)(y_i - y_bar)^T
            # The term (1 + delta_hsig) * outer(pc, pc) is part of the update?
            # Actually, the rank-1 update is:
            # C <- (1 - cc) C + cc * outer(pc, pc) ?? No.
            
            # Correct update:
            # C <- (1 + delta_hsig - cmu) C + cmu * C_mu + (1 + delta_hsig - cmu) * outer(pc, pc)
            # IS INCORRECT.
            
            # The correct sequential update is:
            # 1. C <- (1 - cc) C + cc * outer(pc, pc)  [Rank-1]
            # 2. C <- (1 - cmu) C + cmu * C_mu        [Rank-mu]
            # Combined:
            # C_new = (1 - cc)(1 - cmu) C_old + (1 - cmu) cc outer(pc, pc) + cmu C_mu
            # Note that delta_hsig is related to the hsig condition in the rank-1 update.
            # If hsig=0, cc is effectively 0? No.
            # The formula with delta_hsig is:
            # C_new = (1 + delta_hsig - cmu) C_old + (1 + delta_hsig - cmu) outer(pc, pc) + cmu C_mu
            # Wait, if hsig=1, delta=0. C_new = (1-cmu)C + outer(pc,pc) + cmu C_mu.
            # If hsig=0, delta=cc(2-cc). C_new = (1+delta-cmu)C + (1+delta-cmu)outer(pc,pc) + cmu C_mu.
            
            # Let's use the explicit two-step logic to be safe, or the combined formula from the paper.
            # Paper: C <- (1 + delta_hsig - cmu) C + cmu * sum ... + (1 + delta_hsig - cmu) * outer(pc, pc)
            # This formula assumes the rank-1 part is added with weight (1 + delta_hsig - cmu)?
            # No, the rank-1 update is C <- (1-cc)C + cc*outer(pc,pc).
            # If we combine:
            # C_rank1 = (1-cc)C + cc*outer(pc,pc)
            # C_new = (1-cmu)C_rank1 + cmu*C_mu
            # C_new = (1-cmu)(1-cc)C + (1-cmu)cc*outer(pc,pc) + cmu*C_mu
            
            # The delta_hsig formulation is an optimization.
            # Let's stick to the simpler, robust two-step update to avoid algebra errors.
            
            C_temp = (1 - cc) * C_old + cc * np.outer(pc, pc)
            C = (1 - cmu) * C_temp + cmu * C_update_term
            
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Regularize if necessary
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-6 * np.eye(N)

        if best_x is None:
            best_x = np.zeros(N)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
