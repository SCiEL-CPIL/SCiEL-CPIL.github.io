import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        calls = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f, calls
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def alive():
            return calls < budget and time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Initialization
        # Evaluate origin
        evaluate(np.zeros(dim))
        
        # Axis sweep to find the "needle" direction and approximate center
        # Discus f11 is sensitive to the max component.
        # We probe each axis at a few radii to see which one drops the value most.
        radii = [1.0, 2.0, 3.0, 4.0]
        for axis in range(dim):
            for r in radii:
                if not alive(): break
                x_pos = np.zeros(dim)
                x_pos[axis] = r
                evaluate(x_pos)
                if not alive(): break
                x_neg = np.zeros(dim)
                x_neg[axis] = -r
                evaluate(x_neg)
            if not alive(): break

        # Phase 2: CMA-ES
        # Compact CMA-ES implementation
        n = dim
        lam = max(4, int(4 + 3 * np.log(n)))  # Population size, e.g., 12 for n=5
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = (2.0 / ((n + 1.8) ** 2 + mueff)) * (1.0 - cc)
        cmu = min(1.0 - c1, (2.0 * (mueff - 2.0 + 1.0 / mueff)) / ((n + 2.0) ** 2 + mueff))
        
        xmean = best_x.copy() if best_x is not None else np.zeros(n)
        # Initial sigma: heuristic based on box size
        sigma = 1.0 
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
        
        C = np.eye(n)
        
        stagnation = 0
        best_f_prev = best_f
        
        while alive():
            # Generate population
            ys = rng.standard_normal((lam, n))
            # Transform ys using B and D
            # y = sigma * (B @ D * y)
            # Note: B @ D is a matrix, but D is a vector scaling columns of B
            # In standard CMA-ES, step = sigma * (B @ (D * y))
            # Let's compute the matrix M = B @ np.diag(D)
            M = B * D  # Broadcasting scales columns of B
            
            xs = np.array([xmean + sigma * (M @ ys[i]) for i in range(lam)])
            xs = np.clip(xs, -5.0, 5.0)
            
            # Evaluate
            fs = np.array([evaluate(xs[i]) for i in range(lam)])
            
            if not alive():
                break
                
            # Sort
            idx = np.argsort(fs)
            xs = xs[idx]
            fs = fs[idx]
            ys = ys[idx]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
                
            # Convergence check
            if fs[0] - fs[-1] < 1e-15 * max(1.0, abs(fs[0])):
                break
            if np.std(fs[:mu]) < 1e-15 * max(1.0, abs(fs[0])):
                break
                
            # Weighted recombination
            y_rec = np.sum(weights[:, None] * ys[:mu], axis=0)
            
            # Update evolution paths
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) / (D @ B.T) @ y_rec
            # Note: (D @ B).T is B.T @ D? No.
            # The inverse of B @ diag(D) is diag(1/D) @ B.T
            # So we need (B @ diag(D))^{-1} @ y_rec
            # Let's use the explicit inverse calculation or just update C carefully.
            # Standard update: ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * (B D)^-1 y_rec
            
            # Calculate (B D)^-1
            # Since B is orthogonal (initially identity, updated via eigen-decomp which preserves orthogonality in CMA-ES context? No, B is from eigendecomp of C, so B is orthogonal).
            # If B is orthogonal, (B D)^-1 = D^-1 B^T.
            inv_BD = (1.0 / D)[:, None] * B.T
            
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (inv_BD @ y_rec)
            
            if np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls / lam + 1) / (n + 1))) < (1.4 + 1.0 / n) * chi_n:
                pc = (1.0 - cc) * pc + cc * y_rec
                rank_pc = np.outer(pc, pc)
            else:
                rank_pc = (1.0 + 2.0 * (cs / (1.0 - cs) - 1.0)) * np.outer(pc, pc)
            
            rank_mu = np.sum(weights[:, None] * (ys[:mu] * ys[:mu].T), axis=0)
            
            # Update Covariance Matrix
            C = (1.0 + c1 * cc * (1.0 - cc) / (1.0 - cc) + c1 - c1) * C # This term is usually 1 + c1*cc*(1-cc)/(1-cc) ?
            # Correct formula: C = (1 - c1 - cmu + c1*cc*(1-cc)/(1-cc)) C + c1(rank_pc - C) + cmu rank_mu
            # The term (1 - c1 - cmu + c1*cc*(1-cc)/(1-cc)) is the decay factor.
            # Let's simplify: The standard update is:
            # C = (1 - c1 - cmu) C + c1 (rank_pc - C) + cmu rank_mu
            # Wait, the standard CMA-ES update is:
            # C = (1 - c1 - cmu + c1 * cc * (1 - cc) / (1 - cc)) * C + c1 * (rank_pc - C) + cmu * rank_mu
            # Actually, the term is just (1 - c1 - cmu) if we use the standard rank-1 and rank-mu updates relative to the identity?
            # No, the standard update is:
            # C <- (1 - c1 - cmu) C + c1 (pc pc^T - C) + cmu sum w_i (y_i y_i^T)
            # This assumes C is the covariance.
            
            C = (1.0 - c1 - cmu) * C + c1 * (rank_pc - C) + cmu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Eigen-decomposition to update B and D
            # We need to ensure C is positive definite.
            try:
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-10))
                B = B.T # eigh returns vectors in columns, we need rows for M = B @ D?
                # Standard: y = B @ (D * y)
                # If eigh returns V such that C = V D V^T, then B = V.T?
                # Let's check: B @ diag(D) @ (B @ diag(D))^T = B diag(D) diag(D) B^T = B diag(D^2) B^T.
                # If C = V diag(eig) V^T, and we set B = V.T, D = sqrt(eig),
                # then B diag(D^2) B^T = V.T diag(eig) V = V diag(eig) V^T = C. Correct.
                # So B should be V.T (rows are eigenvectors).
                # np.linalg.eigh returns eigenvectors in columns.
                # So V = eigenvectors. B = V.T.
            except np.linalg.LinAlgError:
                # Fallback: restart or break
                break
                
            # Update xmean
            xmean = xmean + sigma * (B @ (D * y_rec))
            
            # Adapt sigma
            sigma *= np.exp(cs / (n + 2.0) * (np.linalg.norm(ps) / chi_n - 1.0))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)

        # Phase 3: Polish with Nelder-Mead if budget remains
        if alive() and best_x is not None:
            from scipy.optimize import minimize
            try:
                res = minimize(func, best_x, method='Nelder-Mead',
                               options={'maxiter': 100, 'maxfev': max(10, budget - calls - 5),
                                        'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
