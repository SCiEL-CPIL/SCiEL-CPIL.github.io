"""Black-box continuous optimizer on BBOB f11 (Discus) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f11 (Discus) using CMA-ES.
    
    CMA-ES adapts its search distribution (mean, covariance, step size)
    based on the observed landscape. For the Discus function with its
    long, narrow valley, CMA-ES's covariance adaptation aligns the search
    with the valley direction, making it far more efficient than
    fixed-shape methods like Nelder-Mead.
    
    Strategy:
    - Brief Sobol exploration to locate the basin
    - CMA-ES with population size tuned to budget
    - Tight deadline guard
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s/330s limit

        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)
        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used, best_f, best_x
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Brief Sobol exploration (16 points) ---
        n_sobol = 16
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_pts = sampler.random(n_sobol) * 10.0 - 5.0

        evals = []
        for x in sobol_pts:
            if calls_used >= budget - 10:
                break
            fx = safe_call(x)
            evals.append((fx, x.copy()))

        if not evals:
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            evals.append((fx, x.copy()))

        evals.sort(key=lambda t: t[0])

        # --- Phase 2: CMA-ES from the best starting point ---
        # CMA-ES hyperparameters
        # Population size: lambda = 4 + floor(3 * ln(n)) for dim=5 -> 4+3*1.61=8
        # But we need to fit within budget. Each generation costs lambda calls.
        # Total generations = (budget - calls_used) / lambda
        # For dim=5, lambda=8, remaining ~480 -> 60 generations. Good.
        
        remaining = budget - calls_used
        if remaining < 20:
            # Not enough budget for CMA-ES, just return best
            return best_x, best_f

        # Choose population size
        # Standard: lambda = 4 + floor(3*ln(n))
        lam = 4 + int(3 * np.log(dim))
        lam = max(4, min(lam, remaining // 5))  # Ensure at least a few generations

        # Starting point: best from Sobol
        x0 = evals[0][1].copy()
        f0 = evals[0][0]

        # Initial step size: for Discus, the valley is narrow.
        # Use a moderate initial sigma. The box is [-5,5]^5.
        # CMA-ES will adapt sigma quickly.
        sigma0 = 1.0  # Will be adapted

        # CMA-ES state
        mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)  # 1/sqrt(eigenvalues)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        cc = (4 + dim / (dim + 4)) / ((dim + 1.3) ** 2 + 4)
        cs = (2 + 2.0 / (dim + 1.3)) / (dim + 2 + 2.0 / (dim + 1.3))
        cmu = min(1 - 2 * cc, 2 / ((dim + 1.5) ** 2 + dim))

        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        def update_covariance(C, B, D, z, weights, cmu, mueff, dim):
            """Update covariance matrix C = B diag(1/D^2) B^T"""
            # Compute outer products of top mu individuals
            # z: (mu, dim) array of (y_i - mean) / sigma
            # y_i = B * D * z_i
            # We need sum w_i * y_i * y_i^T
            # y_i = B @ (D * z_i)
            Y = (B * D) @ z.T  # (dim, mu)
            # C_new = C + cmu/mueff * (sum w_i y_i y_i^T - C)
            # But we work with B, D representation
            # Simpler: recompute eigen decomposition periodically
            
            # Actually, let's just maintain C directly for simplicity
            # C = sum w_i y_i y_i^T + (1 - cmu/mueff) * C
            # But this is expensive for large dim. For dim=5 it's fine.
            pass

        def eigendecomp(C):
            """Eigen decomposition of C = B diag(D^2) B^T"""
            evals_eig, evecs = np.linalg.eigh(C)
            # Ensure positive
            evals_eig = np.maximum(evals_eig, 1e-20)
            D_eig = np.sqrt(evals_eig)
            B_eig = evecs
            return B_eig, D_eig, evals_eig

        # CMA-ES main loop
        generation = 0
        while calls_used < budget - 5 and time.time() < deadline - 10:
            if time.time() > deadline - 15:
                break

            # Check if we've run out of budget for a full generation
            if calls_used + lam > budget - 5:
                # Do a partial generation or stop
                break

            # Sample lambda points
            ys = []
            xs = []
            fs = []

            for i in range(lam):
                if calls_used + 1 > budget - 5:
                    break
                if time.time() > deadline - 15:
                    break

                # Sample z ~ N(0, I)
                z = rng.standard_normal(dim)

                # Transform to y = B * D * z
                y = (B * D) @ z

                # x = mean + sigma * y
                x = mean + sigma * y
                x = np.clip(x, -5.0, 5.0)

                fx = safe_call(x)
                ys.append(y)
                xs.append(x.copy())
                fs.append(fx)

            if not fs:
                break

            # Sort by function value
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]

            # Use top mu for update
            mu_actual = min(mu, len(ys))
            ys_top = np.array(ys[:mu_actual])  # (mu, dim)
            xs_top = np.array(xs[:mu_actual])
            fs_top = np.array(fs[:mu_actual])

            # Update mean
            new_mean = np.sum(weights[:mu_actual, None] * xs_top, axis=0)

            # Update step size (sigma)
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * C_inv^(1/2) * (mean - old_mean)/sigma
            # C_inv^(1/2) = B * D * B^T ... actually C_inv = B diag(1/D^2) B^T
            # C_inv^(1/2) = B diag(1/D) B^T
            diff = (new_mean - mean) / sigma
            # C_inv_sqrt @ diff = B @ (D_inv @ (B.T @ diff))
            Bt_diff = B.T @ diff
            D_inv_Bt_diff = Bt_diff / D
            C_inv_sqrt_diff = B @ D_inv_Bt_diff
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt_diff

            sigma *= np.exp(cs / (dim + 2) * (np.linalg.norm(ps) / np.sqrt(mueff) - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Update evolution path pc
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * diff

            # Update covariance matrix
            # C = (1 - cmu/mueff - 2*cc*(1-cc))*C 
            #     + cmu/mueff * sum w_i y_i y_i^T
            #     + cc*(2-cc) * pc * pc^T
            # where y_i = (x_i - mean)/sigma
            
            # Recompute C from scratch for stability (dim=5 is cheap)
            # C = sum w_i y_i y_i^T
            # But we need to use the y's in the rotated space
            # Actually, the standard CMA-ES update is:
            # C_new = (1 - cmu/mueff) * C + cmu/mueff * sum w_i (x_i - mean) (x_i - mean)^T / sigma^2
            # Plus the pc term
            
            # Let's use the direct formula
            C_new = (1 - cmu / mueff) * C
            for i in range(mu_actual):
                wi = weights[i]
                yi = (xs_top[i] - mean) / sigma
                C_new += wi * np.outer(yi, yi)
            
            # Add pc contribution
            C_new += cc * (2 - cc) * np.outer(pc, pc)

            # Ensure C is symmetric positive definite
            C_new = (C_new + C_new.T) / 2
            # Add small regularization
            C_new += 1e-10 * np.eye(dim)

            C = C_new

            # Update eigen decomposition
            B, D, _ = eigendecomp(C)

            # Update mean
            mean = new_mean

            generation += 1

            # Termination checks
            if sigma < 1e-12:
                break
            if np.max(np.abs(D)) / np.min(np.abs(D)) > 1e12:
                break
            if np.linalg.norm(mean) > 1e8:
                break

        # --- Phase 3: Final local polish with Nelder-Mead if budget remains ---
        if best_x is not None and calls_used < budget - 10 and time.time() < deadline - 10:
            from scipy.optimize import minimize
            remaining = budget - calls_used
            max_iter = max(10, remaining // 3)

            def nm_func(x):
                return safe_call(x)

            try:
                res = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            if calls_used < budget:
                best_f = safe_call(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
