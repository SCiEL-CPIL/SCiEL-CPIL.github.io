"""Portfolio optimizer for BBOB-style f7 (Step ellipsoidal), dim=5, budget=500."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Broad sampling + Axis scans + Robust CMA-ES + (1+1)-ES Polish.
    
    Strategy:
    1. Phase 0: Coarse global sampling to identify the basin.
    2. Phase 1: Axis-aligned scans to exploit 1-D structure and find step boundaries.
    3. Phase 2: CMA-ES to handle rotation and scale differences (the core of f7).
    4. Phase 3: (1+1)-ES for fine-grained local polishing.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # 300s limit, 15s margin
        B = self.budget
        d = self.dim
        LO, HI = -5.0, 5.0
        SPAN = HI - LO

        # ---- bookkeeping -------------------------------------------------
        used = 0

        def f(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), LO, HI)
            used += 1
            return float(func(x))

        # ---- Phase 0: cheap initial samples -----------------------------
        # Center + axis extremes to give a broad picture in ~11 calls.
        xs = [np.full(d, 0.0)]
        for i in range(d):
            p = np.full(d, 0.0); p[i] = -4.0; xs.append(p.copy())
            p = np.full(d, 0.0); p[i] =  4.0; xs.append(p.copy())

        best_x = None
        best_f = np.inf
        for x in xs:
            if used >= B or time.time() > deadline:
                break
            v = f(x)
            if v < best_f:
                best_f, best_x = v, x.copy()

        # ---- Phase 1: coordinate-axis scans -----------------------------
        # For f7, the "step" nature means large changes in value along axes 
        # corresponding to large coefficients. Scanning axes helps find 
        # the "flat" regions or the minimum if rotation is small.
        # Use fewer points but focus on finding the basin more efficiently.
        n_axis_pts = 7
        for i in range(d):
            if used >= B or time.time() > deadline:
                break
            lo = max(LO, best_x[i] - 2.5)
            hi = min(HI, best_x[i] + 2.5)
            for t in np.linspace(lo, hi, n_axis_pts):
                if used >= B or time.time() > deadline:
                    break
                cand = best_x.copy(); cand[i] = t
                v = f(cand)
                if v < best_f:
                    best_f, best_x = v, cand.copy()

        # ---- Phase 2: Robust CMA-ES -------------------------------------
        # CMA-ES is superior for rotated ellipsoids (f7 has hidden rotation).
        # We use a standard implementation with careful parameter tuning.
        rng = np.random.default_rng(12345)
        
        # Parameters
        lam = max(4, int(4 + 3 * np.log(d)))  # ~10-12 for d=5
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        x_mean = best_x.copy()
        # Start with a smaller sigma to be more conservative
        sigma = 1.0
        C = np.eye(d)
        C_inv_sqrt = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        gen = 0
        stagnation_count = 0
        max_stagnation = 40 # Generations without improvement before switching to polish

        while used < B:
            if time.time() > deadline:
                break
            # Reserve calls for polish phase
            if B - used < lam + 20:
                break

            pop = np.empty((lam, d))
            fvals = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = C_inv_sqrt @ rng.standard_normal(d)
                pop[i] = np.clip(x_mean + sigma * z, LO, HI)
                fvals[i] = f(pop[i])
            
            if time.time() > deadline:
                break

            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]

            if fvals[0] < best_f:
                best_f, best_x = float(fvals[0]), pop[0].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
                if stagnation_count > max_stagnation:
                    break
            
            # Adaptive sigma: if we're making progress, allow sigma to grow slightly
            # If stuck, shrink faster
            if stagnation_count > 15:
                sigma *= 0.9  # Shrink faster when stuck

            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

            step = x_mean - x_mean_old
            step_norm = np.linalg.norm(step)
            
            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * step / (sigma * step_norm + 1e-15)
            
            # Hessian check for sigma update
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (d + 1)) * chi_n
            sigma *= np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = float(np.clip(sigma, 1e-4, SPAN))

            # Update pc
            if h_sigma:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * step / sigma
            else:
                pc = (1 - c_c) * pc

            # Update C
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = ((1 - c_1 - c_mu) * C
                 + c_1 * ((1 + delta_h) * np.outer(pc, pc) - C)
                 + c_mu * np.sum(weights[i] * np.outer(pop[i] - x_mean_old, pop[i] - x_mean_old) for i in range(mu)))

            # Ensure C is positive semi-definite and update inverse sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            gen += 1

        # ---- Phase 3: (1+1)-ES Polish -----------------------------------
        # Use remaining budget for a simple, robust (1+1)-ES.
        # This is effective for final convergence where CMA-ES might be 
        # too noisy or if the landscape is very flat.
        if used < B and time.time() < deadline:
            rng2 = np.random.default_rng(999)
            sigma_es = 0.3  # Start smaller for finer polish
            fails = 0
            max_fails = 15
            
            while used < B and time.time() < deadline:
                y = np.clip(best_x + sigma_es * rng2.standard_normal(d), LO, HI)
                fy = f(y)
                if fy < best_f:
                    best_f, best_x = fy, y.copy()
                    sigma_es = min(sigma_es * 1.15, SPAN)
                    fails = 0
                else:
                    sigma_es *= 0.85
                    fails += 1
                    if fails >= max_fails:
                        break
                # If sigma is very small, we are likely at the precision limit
                if sigma_es < 1e-4:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
