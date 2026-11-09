"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 30s margin

        # CMA-ES Parameters
        # lambda = 4 is a good balance for dim=5 with budget=500
        lam = 4
        mu = lam // 2  # 2
        # Weights for the top mu individuals
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Step size
        # Initial sigma should be large enough to explore the box
        sigma = 1.0 
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim**2))
        
        # Initial mean
        # Start at origin, as it's the center of the box and often close to the shifted minimum
        # Or random start? Origin is usually safe for centered landscapes.
        xmean = np.zeros(dim)
        C = np.eye(dim)
        Cinv = np.eye(dim)
        Cinvsqrt = np.eye(dim)
        
        # State variables
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Track best
        best_x = xmean.copy()
        best_f = np.inf
        calls = 0
        
        def update_chol(C, Cinv):
            # Compute Cholesky of C and inverse Cholesky of Cinv
            # We need Cinvsqrt such that Cinvsqrt @ Cinvsqrt.T = Cinv
            # And Csqrt such that Csqrt.T @ Csqrt = C
            # Standard CMA-ES uses Cinvsqrt for sampling: y = Cinvsqrt @ z
            # Wait, standard notation: x = xmean + sigma * Cinvsqrt @ z ? No.
            # x = xmean + sigma * Csqrt @ z.
            # Cinvsqrt is used for ps update: ps = (1-cs)*ps + cs/sigma * Cinvsqrt @ (xnew - xold)
            
            try:
                Csqrt = np.linalg.cholesky(C)
                Cinv = np.linalg.inv(C)
                Cinvsqrt = np.linalg.cholesky(Cinv).T
            except np.linalg.LinAlgError:
                # Fallback if matrices become ill-conditioned
                C = np.eye(dim)
                Cinv = np.eye(dim)
                Cinvsqrt = np.eye(dim)
            return Csqrt, Cinv, Cinvsqrt

        Csqrt, Cinv, Cinvsqrt = update_chol(C, Cinv)

        # Main Loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate population
            pop_x = np.zeros((lam, dim))
            pop_f = np.zeros(lam)
            
            for i in range(lam):
                z = np.random.randn(dim)
                x = xmean + sigma * (Csqrt.T @ z)
                # Clip to box? The problem says func clips. 
                # It is generally better NOT to clip x before passing to func if func handles it,
                # but for CMA-ES geometry, clipping can distort the distribution.
                # However, the contract says "func(x) clips... and returns a float".
                # So we pass unclipped x.
                pop_x[i] = x
                pop_f[i] = func(x)
                calls += 1
                if time.time() > deadline:
                    break
            
            if calls >= budget or time.time() > deadline:
                # Update best if we just made a call
                idx_min = np.argmin(pop_f[:calls])
                if pop_f[idx_min] < best_f:
                    best_f = pop_f[idx_min]
                    best_x = pop_x[idx_min]
                break

            # Sort population
            idx_sorted = np.argsort(pop_f)
            x_sorted = pop_x[idx_sorted]
            f_sorted = pop_f[idx_sorted]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0]
                
            # Weighted mean
            xnew = np.sum(w[:, None] * x_sorted[:mu], axis=0)
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (Cinvsqrt @ (xnew - xmean))
            
            # Update pc
            if np.linalg.norm(ps) / chi_n < (1 + cs) * hsig:
                # hsig needs to be calculated
                pass # We need hsig before this. Let's restructure.
            
            # Recalculate hsig properly
            # hsig = ||ps|| / chi_n < (1 + cs) * sqrt((1 - (1-cs)^(2*(gen+1))) / (1 - (1-cs)^2))
            # We don't track gen explicitly in a simple loop, but we can approximate or just use the standard condition
            # For simplicity and robustness in short runs, we can use a simplified hsig check or always update pc if ps is small.
            # Standard condition:
            hsig = (np.linalg.norm(ps) / chi_n) < (1 + cs) * np.sqrt(1 - (1 - cs)**(2 * (calls // lam + 1)))
            
            if hsig:
                pc = (1 - cs) * pc + cs * (xnew - xmean) / sigma
                delta_hsig = (1 - hsig) * (1 + cs) * (1 + 2 * (mueff - 1) / dim)
                C = (1 - c1 - cmu + cs * (1 - cs) + delta_hsig * c1) * C + \
                    c1 * (pc @ pc.T + (1 - cs) * (1 - 2 * cs) * C) + \
                    cmu * np.sum(w[:, None, None] * np.outer(x_sorted[:mu] - xmean, x_sorted[:mu] - xmean) / sigma**2, axis=0)
            else:
                C = (1 - c1 - cmu + cs * (1 - cs)) * C + \
                    c1 * (1 - cs) * (1 - 2 * cs) * C + \
                    cmu * np.sum(w[:, None, None] * np.outer(x_sorted[:mu] - xmean, x_sorted[:mu] - xmean) / sigma**2, axis=0)
            
            # Update sigma
            sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update xmean
            xmean = xnew
            
            # Update C matrices
            Csqrt, Cinv, Cinvsqrt = update_chol(C, Cinv)
            
            # Ensure C is symmetric and positive definite (numerical stability)
            C = (C + C.T) / 2
            # Regularize if eigenvalues are bad
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += (1e-12 - np.min(eigvals)) * np.eye(dim)
                Csqrt, Cinv, Cinvsqrt = update_chol(C, Cinv)

        return best_x, best_f
# EVOLVE-BLOCK-END
