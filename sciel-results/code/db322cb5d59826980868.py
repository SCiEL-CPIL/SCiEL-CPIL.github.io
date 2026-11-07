"""Optimizer for BBOB-style f17 (Schaffers F7) using a two-phase approach:
Phase 1: Random restarts with local Nelder-Mead simplex refinement
Phase 2: Global CMA-ES with adaptive restarts
"""
import numpy as np
import time

class Optimizer:
    """Two-phase optimizer: random restarts + local refinement, then CMA-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        count = 0
        
        # Helper to safely evaluate
        def safe_eval(x):
            nonlocal count, best_f, best_x
            if count >= budget or time.time() > deadline:
                return None
            try:
                f = func(x)
                count += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return None
        
        # Phase 1: Random restarts with local refinement (use ~60% of budget)
        # This helps find a good basin early for Schaffers F7 which has many local minima
        phase1_budget = int(budget * 0.6)
        
        # Initial point at origin
        f0 = safe_eval(np.zeros(dim))
        if f0 is not None:
            best_f = f0
            best_x = np.zeros(dim).copy()
        
        # Random restarts
        n_restarts = 8
        points_per_restart = max(10, (phase1_budget - count) // n_restarts)
        
        for r in range(n_restarts):
            if count >= phase1_budget or time.time() > deadline:
                break
            
            # Random starting point
            x0 = np.random.uniform(-4, 4, dim)
            f_start = safe_eval(x0)
            if f_start is None:
                continue
            
            # Local Nelder-Mead refinement
            # Simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5
                simplex.append(xi)
            
            f_simplex = []
            for i in range(dim + 1):
                f_val = safe_eval(simplex[i])
                if f_val is None:
                    break
                f_simplex.append(f_val)
            else:
                # Run Nelder-Mead
                max_iter = 50
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                
                for it in range(max_iter):
                    if count >= phase1_budget or time.time() > deadline:
                        break
                    
                    # Sort simplex
                    order = np.argsort(f_simplex)
                    simplex = [simplex[i].copy() for i in order]
                    f_simplex = [f_simplex[i] for i in order]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_ref = centroid + alpha * (centroid - simplex[-1])
                    f_ref = safe_eval(x_ref)
                    if f_ref is None:
                        break
                    
                    if f_ref < f_simplex[0]:
                        # Expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        f_exp = safe_eval(x_exp)
                        if f_exp is None:
                            break
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            f_simplex[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            f_simplex[-1] = f_ref
                    elif f_ref < f_simplex[-2]:
                        # Accept reflection
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                    else:
                        # Contraction
                        x_con = centroid + rho * (simplex[-1] - centroid)
                        f_con = safe_eval(x_con)
                        if f_con is None:
                            break
                        if f_con < f_ref:
                            simplex[-1] = x_con
                            f_simplex[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                f_shrink = safe_eval(simplex[i])
                                if f_shrink is None:
                                    break
                                f_simplex[i] = f_shrink
                            else:
                                continue
                            break
        
        # Phase 2: CMA-ES from the best point found
        if count < budget and time.time() < deadline:
            # Initialize CMA-ES
            lam = 4 + int(3 * np.log(dim))
            mu = lam // 2
            w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mueff = 1 / np.sum(w**2)
            
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            
            sigma = 1.0  # Smaller step size since we're already in a good region
            mean = best_x.copy()
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            def get_C_half():
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                return eigvecs @ np.diag(np.sqrt(eigvals))
            
            C_half = get_C_half()
            
            while count < budget:
                if time.time() > deadline:
                    break
                
                # Sample population
                Z = np.random.randn(lam, dim)
                X = (sigma * (C_half @ Z.T)).T + mean
                
                # Evaluate
                F = np.zeros(lam)
                valid = 0
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    f_val = safe_eval(X[i])
                    if f_val is None:
                        break
                    F[i] = f_val
                    valid += 1
                
                if valid < 2 or time.time() > deadline:
                    break
                
                # Sort by fitness
                order = np.argsort(F)
                X_sorted = X[order]
                F_sorted = F[order]
                
                # Update mean
                mean_old = mean.copy()
                mean = np.sum(w[:valid] / np.sum(w[:valid])[:, np.newaxis] * X_sorted[:min(mu, valid)], axis=0)
                
                # Update sigma (CSA)
                delta = mean - mean_old
                try:
                    v = np.linalg.solve(C_half, delta / sigma)
                except np.linalg.LinAlgError:
                    v = delta / sigma
                
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * v
                
                gen = count // lam
                denom = np.sqrt(1.0 - (1.0 - cs) ** (2 * (gen + 1)))
                if denom < 1e-12:
                    denom = 1e-12
                h_sigma = 1 if (np.linalg.norm(p_sigma) / denom < (1.4 + 2.0 / (dim + 1.0)) * (1.0 + cs)) else 0
                
                # Update p_c
                p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff) * (v * sigma)
                
                # Update C
                Y = (X_sorted[:min(mu, valid)] - mean_old) / sigma
                w_norm = w[:min(mu, valid)] / np.sum(w[:min(mu, valid)])
                rank_mu = (Y.T * w_norm) @ Y
                
                C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * cc * (2 - cc) * C) + cmu * rank_mu
                C = (C + C.T) / 2
                
                # Update sigma
                sigma = sigma * np.exp((cs / (2 * lam)) * (np.linalg.norm(p_sigma) ** 2 / dim - 1))
                
                # Update C_half
                C_half = get_C_half()
                
                # Check for convergence
                if sigma < 1e-10:
                    break
        
        return best_x, best_f
