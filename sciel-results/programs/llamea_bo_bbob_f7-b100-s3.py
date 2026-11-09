import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_lo = -5.0
        self.bound_hi = 5.0
        self.start_time = None
        self.deadline = 325.0  # 5s margin

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        lo = self.bound_lo
        hi = self.bound_hi

        rng = np.random.default_rng(42)

        # Phase 1: Enhanced Scout Phase
        # Use stratified sampling to cover the space better than pure random.
        # We will use 20 points.
        scout_budget = 20
        best_x = None
        best_f = np.inf
        scout_points = []
        
        # Stratified sampling in 5D: divide each dimension into 2 intervals (4^5 = 1024 cells, too many).
        # Instead, let's just use a mix of random and a few structured points (center, corners, etc.)
        # Or simply random with a fixed seed is robust enough if we increase the count slightly.
        # Let's stick to random but ensure we check the center.
        
        points = []
        # Center
        points.append(np.zeros(d))
        # Random uniform
        for _ in range(scout_budget - 1):
            points.append(rng.uniform(lo, hi, d))
            
        for x in points:
            if time.time() - self.start_time > self.deadline:
                break
            fx = func(x)
            scout_points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)

        # Phase 2: Multi-Start CMA-ES
        # We will run CMA-ES for a portion of the budget.
        # If stagnation occurs, we restart from a new random point or the best found so far.
        
        lam = 4 + int(3 * np.log(d))
        if lam < 6:
            lam = 6
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        chi_n = np.sqrt(d) * (1 - 1.0/(4*d) + 1.0/(21*d*d))

        # We will use 80% of the budget for the first CMA-ES run.
        # If it stagnates, we use the remaining 20% for a restart.
        # Actually, let's just run one CMA-ES but with a stagnation check that triggers a "jolt" (random restart of sigma/mean) rather than a full restart, to save complexity and keep code compact.
        
        m = best_x.copy()
        # Initialize sigma based on scout spread
        if len(scout_points) > 2:
            scout_points.sort(key=lambda t: t[1])
            top_half = np.array([t[0] for t in scout_points[:max(1, len(scout_points)//2)]])
            sigma = np.linalg.norm(np.std(top_half, axis=0), axis=0) * 0.5
            if sigma < 0.1:
                sigma = 0.1
            elif sigma > 2.0:
                sigma = 2.0
        else:
            sigma = 1.0

        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        remaining_budget = budget - scout_budget
        best_f_cma = best_f
        best_x_cma = best_x.copy()
        
        stagnation_count = 0
        stagnation_threshold = 15 # Generations without improvement
        
        while remaining_budget >= lam:
            if time.time() - self.start_time > self.deadline:
                break
                
            # Sample candidates
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(d)
                L = np.eye(d)
                
            Z = rng.standard_normal((lam, d))
            Y = Z @ L.T
            X = m[None, :] + sigma * Y
            X = np.clip(X, lo, hi)
            
            F = np.array([func(x) for x in X])
            
            min_idx = np.argmin(F)
            if F[min_idx] < best_f_cma:
                best_f_cma = F[min_idx]
                best_x_cma = X[min_idx].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Stagnation Handling:
            # If we haven't improved for a while, we might be stuck in a local minimum or plateau.
            # For Step Ellipsoidal, a plateau is likely. We can try to increase sigma to jump out,
            # or restart from a new random point if we are really stuck.
            if stagnation_count > stagnation_threshold:
                # Jolt: Increase sigma and maybe shift mean slightly
                sigma *= 1.5
                if sigma > 5.0:
                    # Full restart from a new random point
                    new_x = rng.uniform(lo, hi, d)
                    new_f = func(new_x)
                    if new_f < best_f_cma:
                        best_f_cma = new_f
                        best_x_cma = new_x.copy()
                    m = new_x.copy()
                    sigma = 1.0
                    C = np.eye(d)
                    pc = np.zeros(d)
                    ps = np.zeros(d)
                    stagnation_count = 0
                    remaining_budget -= 1 # Account for the extra call? No, let's just reset state.
                    # Actually, the budget accounting is tricky here if we add extra calls.
                    # Let's stick to just increasing sigma and resetting paths, which is cheaper.
                    C = np.eye(d)
                    pc = np.zeros(d)
                    ps = np.zeros(d)
                    # Do NOT reset m, keep it at the best found so far? No, if we are stuck, m is likely in the basin.
                    # Let's just reset the adaptive variables.
                    pass

            sort_idx = np.argsort(F)
            X_sorted = X[sort_idx]
            F_sorted = F[sort_idx]
            
            m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-16, None)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            step = (m_new - m) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ step)
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            
            h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2 / (d + 1)) * chi_n else 0
            
            C = (1 + c_1 * (h_sigma - 1) - c_1 - c_mu) * C \
                + c_1 * np.outer(pc, pc)
                
            Y_delta = X_sorted[:mu] - m
            Y_delta = Y_delta / sigma
            rank_mu_term = np.sum(weights[:, None, None] * Y_delta[:, :, None] * Y_delta[:, None, :], axis=0)
            C += c_mu * rank_mu_term
            
            C = (C + C.T) / 2.0
            if not np.all(np.linalg.eigvalsh(C) > 0):
                C = C + 1e-4 * np.eye(d)
                
            m = m_new
            remaining_budget -= lam
            
        return best_x_cma, best_f_cma
