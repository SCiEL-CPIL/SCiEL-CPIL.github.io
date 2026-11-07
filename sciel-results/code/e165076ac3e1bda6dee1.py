"""Optimized program for BBOB f21 (Gallagher 101 peaks) with CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """CMA-ES optimizer for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size lambda
        # For dim=5, standard is 4 + int(3*log(5)) = 8, but larger is better for multimodal.
        # Inspiration suggests 30.
        lam = 30
        
        # Initial mean
        m = np.zeros(dim)
        # Initial sigma
        # The search space is [-5, 5]^5. 
        # A reasonable initial step size is 1.0 or 2.0.
        # Given f21 has peaks, a larger initial sigma might help escape local minima.
        # 1.5 is a good compromise for [-5,5] box.
        sigma = 1.5
        
        # CMA-ES weights
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        dcor = 1.0 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        
        # Precompute square root of C (initially identity)
        C_eigvals = np.ones(dim)
        C_eigvecs = np.eye(dim)
        
        def update_eig():
            nonlocal C, C_eigvals, C_eigvecs
            C_eigvals, C_eigvecs = np.linalg.eigh(C)
            C_eigvals = np.maximum(C_eigvals, 1e-12)
            
        # Initial evaluation
        n_calls = 0
        best_f = float('inf')
        best_x = None
        
        def wrapped_func(x):
            nonlocal n_calls, best_f, best_x
            if n_calls >= budget:
                return 1e18
            if time.time() > deadline - 1:
                return 1e18
            n_calls += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Evaluate initial mean
        wrapped_func(m)
        
        # Main loop
        while n_calls < budget and time.time() < deadline - 5:
            # 1. Sample population
            sqrt_C_eigvals = np.sqrt(C_eigvals)
            C_sqrt = C_eigvecs @ np.diag(sqrt_C_eigvals)
            
            Z = np.random.default_rng(42 + n_calls).standard_normal((lam, dim))
            
            Y = m[None, :] + sigma * (C_sqrt @ Z.T).T
            Y = np.clip(Y, -5.0, 5.0)
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                if n_calls >= budget or time.time() > deadline - 5:
                    F[i] = 1e18
                    continue
                F[i] = wrapped_func(Y[i])
            
            if n_calls >= budget or time.time() > deadline - 5:
                break
                
            # Sort by fitness
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()
                
            # 2. Update mean
            m_old = m.copy()
            m = np.sum(w[:, None] * Y_sorted[:mu], axis=0)
            
            # 3. Update sigma, C, pc, ps
            y = (m - m_old) / sigma
            
            # Update ps
            hs = (math.sqrt(1 - math.exp(-2 * (cs + 1))) * (dim + 2) / 2 + cs)
            if np.linalg.norm(y) < (1 + 5.0 / (dim + 1)) * hs * math.sqrt(dim) * sigma:
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mu_eff) * y
            else:
                pass # ps = (1-cs)*ps implicitly by not adding, but standard is to just not update?
                # Actually standard is ps = (1-cs)*ps. Let's do that.
                ps = (1 - cs) * ps
            
            # Update pc
            # The condition for pc update involves the norm of ps relative to expected value.
            # Simplified check: if ps is not too large.
            if np.linalg.norm(ps) < (1 + 5.0 / (dim + 1)) * hs * sigma:
                 pc = (1 - cc) * pc + math.sqrt(cc * (2 - cc) * mu_eff) * ps / sigma
            else:
                 pc = (1 - cc) * pc
            
            # Update C
            norm_ps_sq = np.dot(ps, ps)
            delta_h = (1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs) * (norm_ps_sq / dim - 1)
            
            rank1 = np.outer(y, y)
            
            Y_selected = Y_sorted[:mu]
            Y_steps = (Y_selected - m_old[None, :]) / sigma
            rank_mu = np.sum(w[:, None, None] * np.einsum('ij,ik->ijk', Y_steps, Y_steps), axis=0)
            
            C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * rank1 + cmu * rank_mu
            
            # Update sigma
            sigma = sigma * math.exp(cs / dcor * (np.linalg.norm(ps) / math.sqrt(dim) - 1))
            
            # Numerical stabilization
            C = (C + C.T) / 2.0
            update_eig()
            
            # Convergence check
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
