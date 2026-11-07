"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lightweight CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)
        
        # Initialize CMA-ES state
        x = rng.uniform(-5, 5, self.dim)
        sigma = 2.0
        C = np.eye(self.dim)
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        
        # Parameters
        mu = max(1, self.dim // 2)  # Number of selected individuals
        # Weights for the selected individuals (standard CMA-ES weights)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Adaptation parameters
        cs = (mu_eff + 2) / (self.dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + cs
        c1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        
        # Number of offspring
        lam = 4 + int(3 * np.log(self.dim))
        
        # Initial evaluation
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1
        gen = 0
        
        # Main loop
        while calls_made + lam <= self.budget:
            if time.time() > deadline:
                break
                
            # Decompose C
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D2, 1e-12))
            invB = np.linalg.inv(B)
            
            # Generate offspring
            z = rng.standard_normal((lam, self.dim))
            y = x[None, :] + sigma * (z @ (B / D[None, :]))
            y = np.clip(y, -5, 5)
            
            # Evaluate
            fy = np.array([func(yi) for yi in y])
            calls_made += lam
            gen += 1
            
            # Track best
            idx_min = np.argmin(fy)
            if fy[idx_min] < best_fx:
                best_fx = fy[idx_min]
                best_x = y[idx_min].copy()
            
            # Sort by fitness
            order = np.argsort(fy)
            y_sel = y[order[:mu]]
            
            # Weighted mean
            m = (w[:, None] * y_sel).sum(axis=0)
            
            # Update path accumulators
            y_m_sigma = (m - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invB @ y_m_sigma)
            
            # hsig check
            # Norm of ps
            ps_norm = np.linalg.norm(ps)
            # Expected norm under null hypothesis (approx)
            # Theoretical expectation: sqrt(1 - (1-cs)^(2*gen)) * sqrt(1 - (1-cs)^2) ... 
            # Standard check: ps_norm / sqrt(1 - (1-cs)^(2*gen)) < (1 + 2/dim) * sqrt(cs * (2 - cs))
            # Note: gen starts at 1 here.
            denom = np.sqrt(1 - (1 - cs) ** (2 * gen))
            if denom > 1e-10:
                ratio = ps_norm / denom
            else:
                ratio = 0
            hsig = 1.0 if ratio < (1 + 2 / self.dim) * np.sqrt(cs * (2 - cs)) else 0.0
            
            # Update pc
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * (invB @ y_m_sigma)
            
            # Update C
            # Rank-1 update
            rank1 = np.outer(pc, pc)
            # Rank-mu update
            y_sel_sigma = (y_sel - x) / sigma
            # Transform to standard normal space for C update? 
            # Standard CMA-ES updates C in the transformed space.
            # C_new = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - hsig) * cs * (2 - cs) * C) 
            #        + cmu * sum(w_i * (z_i @ z_i.T))
            # Where z_i are the standard normal vectors of the selected individuals.
            # We have y_sel. We need the corresponding z_sel.
            # y = x + sigma * B * D^-1 * z  =>  z = D * B^-1 * (y - x) / sigma
            # So z_sel = D[None, :] * (invB @ y_sel_sigma.T).T
            z_sel = (D[None, :] * (invB @ y_sel_sigma.T).T)
            
            rank_mu = np.sum([w[i] * np.outer(z_sel[i], z_sel[i]) for i in range(mu)], axis=0)
            
            C = (1 - c1 - cmu) * C + c1 * (rank1 + (1 - hsig) * cs * (2 - cs) * C) + cmu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small diagonal for stability if needed
            # C += 1e-12 * np.eye(self.dim)
            
            # Update x
            x = m
            
            # Update sigma
            # log(sigma) update
            # log_sigma = log_sigma + cs / ds * (||ps|| / sqrt(1 - (1-cs)^(2*gen)) - chi_n)
            # chi_n = sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
            chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
            if denom > 1e-10:
                log_sigma_update = (ps_norm / denom - chi_n)
            else:
                log_sigma_update = 0
            sigma *= np.exp(cs / ds * log_sigma_update)
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # If sigma is too small, we might be stuck. 
            # For now, just continue.
            
        return best_x, best_fx
# EVOLVE-BLOCK-END
