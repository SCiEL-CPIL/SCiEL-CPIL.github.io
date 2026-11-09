import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        budget = self.budget
        d = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                raise StopIteration()
            f = float(func(np.asarray(x, dtype=np.float64)))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=np.float64).copy()
            return f

        # Initialize at origin
        x0 = np.zeros(d)
        try:
            f_x0 = safe_func(x0)
        except StopIteration:
            return best_x, best_f

        # CMA-ES Parameters
        # lambda = 4 + 3*ln(d) is standard. For d=5, lambda ~ 11.
        lam = 11 
        # Weights
        w = np.log(lam / 2 + 1) - np.log(np.arange(1, lam + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step size adaptation
        c_sigma = (lam / 2 + 1) / (d + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (d + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + lam / d) / (d + 4 + 2 * lam / d)
        c_1 = 2 / ((d + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * (lam / 2 - 2 + 1 / lam) / ((d + 2) ** 2 + lam))
        
        # Initial state
        mean = x0.copy()
        sigma = 2.0 # Initial step size
        C = np.eye(d)
        C_sqrt = np.eye(d)
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Precompute constants
        chi_lam = np.sqrt(d) * (1 - 1 / (2 * d * d) + 1 / (8 * d**4))
        
        def update_chol(C):
            # Add small regularization to ensure PD
            L = np.linalg.cholesky(C + 1e-12 * np.eye(d))
            return L

        try:
            C_sqrt = update_chol(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(d)

        # Main CMA-ES Loop
        while calls < budget and time.time() < deadline:
            # Check if we have enough calls for a full generation
            if calls + lam > budget:
                break
            
            # Generate population
            pop = []
            for i in range(lam):
                z = rng.normal(0, 1, d)
                y = C_sqrt @ z
                x_i = mean + sigma * y
                pop.append(x_i)
            
            # Evaluate population
            f_vals = []
            for x_i in pop:
                try:
                    f_i = safe_func(x_i)
                    f_vals.append(f_i)
                except StopIteration:
                    break
            
            if len(f_vals) < lam:
                break
                
            # Sort
            idx = np.argsort(f_vals)
            pop = [pop[i] for i in idx]
            f_vals = [f_vals[i] for i in idx]
            
            # Update Mean
            mean_old = mean.copy()
            mean = np.sum(w[:, np.newaxis] * np.array(pop), axis=0)
            
            # Update p_sigma
            y_k = (mean - mean_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_k
            
            # Check h_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (calls // lam + 1)))) < (1.4 + 2 / (d + 1)) * chi_lam else 0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_k
            
            # Update Covariance Matrix
            y_sel = np.array([(pop[i] - mean_old) / sigma for i in range(lam)])
            y_w = np.sum(w[:, np.newaxis] * y_sel, axis=0)
            
            rank_mu = np.zeros((d, d))
            for i in range(lam):
                diff = y_sel[i] - y_w
                rank_mu += w[i] * np.outer(diff, diff)
                
            # Rank-one update
            # Note: The term (1 - h_sigma) * (1 + (1 - c_c)*(1 - c_c) - mu_eff) * C
            # is often simplified in practice to just ensure stability.
            # A common stable update is:
            C_new = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * (1 + (1 - c_c) * (1 - c_c) - mu_eff) * C) + c_mu * rank_mu
            
            # Symmetrize
            C_new = (C_new + C_new.T) / 2
            
            # Ensure Positive Definite
            try:
                C_sqrt = update_chol(C_new)
                C = C_new
            except np.linalg.LinAlgError:
                # If not PD, restart from best with identity covariance
                mean = best_x.copy()
                sigma *= 0.5
                C = np.eye(d)
                C_sqrt = np.eye(d)
                p_sigma = np.zeros(d)
                p_c = np.zeros(d)
                continue
            
            # Update Step Size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_lam - 1))
            
            # Convergence check
            if sigma < 1e-12:
                break

        # Final Polish with (1+1)-ES
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.05 # Smaller initial sigma for polish
            while calls < budget and time.time() < deadline:
                step = rng.normal(0, 1, d)
                x_candidate = x + sigma * step
                try:
                    f_candidate = safe_func(x_candidate)
                except StopIteration:
                    break
                if f_candidate < f_x:
                    x = x_candidate
                    f_x = f_candidate
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                    if sigma < 1e-8:
                        break

        return best_x, best_f
