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
        deadline = start_time + 290  # Safety margin before 300s limit

        # CMA-ES parameters tuned for dim=5, budget=500
        lam = 15
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        chi_lambda = np.sqrt(dim * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2)))

        best_x = np.zeros(dim)
        best_f = np.inf
        evals = 0

        # Helper to evaluate and update global best
        def do_eval(x):
            nonlocal best_x, best_f, evals
            f = float(func(x))
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Deterministic Space-Filling Exploration (Sobol/Halton-like)
        # This ensures we cover the search space broadly and quickly,
        # avoiding the initial slow convergence of stochastic methods.
        # We use a simple low-discrepancy sequence approach.
        
        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        # Generate a small set of deterministic points to start
        # This is cheap and helps find a good initial sigma/center
        n_init = 10
        init_points = []
        for i in range(1, n_init + 1):
            p = np.array([halton_sequence(i, b) for b in range(2, 2 + dim)])
            # Map [0,1] to [-5,5]
            p = p * 10.0 - 5.0
            init_points.append(p)
        
        # Evaluate these points
        for p in init_points:
            if evals >= budget or time.time() > deadline:
                break
            do_eval(p)

        # Phase 2: CMA-ES with Adaptive Restart and Local Refinement
        # Start CMA-ES from the best point found in Phase 1
        
        def run_cma_es(x0, sigma0, max_evals):
            nonlocal best_x, best_f, evals

            m = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            # Initial evaluation of mean
            do_eval(m)

            no_improve_count = 0
            stagnation_threshold = 10  # Generations without improvement
            gen = 0
            
            while evals + lam <= max_evals and (time.time() - start_time) < deadline:
                gen += 1

                # Sample lambda candidates
                Z = np.random.randn(lam, dim)
                X = m + sigma * (Z @ L.T)
                # Clip to bounds to avoid wasted evaluations (though func clips, it's cleaner)
                X = np.clip(X, -5, 5)

                # Evaluate
                F = np.array([do_eval(x) for x in X])

                # Sort
                order = np.argsort(F)
                F_sorted = F[order]
                X_sorted = X[order]

                # Update best tracking for stagnation
                if F_sorted[0] < best_f:
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Check for stagnation - restart with perturbation
                if no_improve_count > stagnation_threshold and gen > 5:
                    # Restart from best with larger sigma
                    break

                # Update mean
                m_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

                # Compute C^{-1/2} v
                v = (m_new - m) / sigma
                try:
                    y = np.linalg.solve(L, v)
                    v_inv_half = y
                except np.linalg.LinAlgError:
                    v_inv_half = v

                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + \
                          np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v_inv_half

                # Update sigma
                sigma_norm = np.linalg.norm(p_sigma) / chi_lambda
                sigma = sigma * np.exp(c_sigma / chi_lambda * (sigma_norm - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)

                # h_sigma
                h_sigma_thresh = (1 + 1.4 / mu_eff) * np.sqrt((1 - c_c) * (2 - c_c))
                h_sigma = 1.0 if sigma_norm < h_sigma_thresh else 0.0

                # Update p_c
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * v_inv_half

                # Update C
                Y = (X_sorted[:mu] - m) / sigma
                M2 = (Y.T * weights) @ Y
                C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * M2
                C = (C + C.T) / 2

                # Update Cholesky
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-10 * np.eye(dim)
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(dim)
                        C = np.eye(dim)
                        p_sigma = np.zeros(dim)
                        p_c = np.zeros(dim)
                        sigma = 1.0
                        m = best_x.copy()

                m = m_new

            return no_improve_count > stagnation_threshold

        # Run CMA-ES
        x0 = best_x.copy()
        sigma0 = 2.0
        run_cma_es(x0, sigma0, budget)

        # Phase 3: Local Refinement / Coordinate Descent
        # If we have budget left, do a simple coordinate descent to polish the solution.
        # This exploits the structure of Schaffers F7 which has ill-conditioned axes.
        if evals < budget and (time.time() - start_time) < deadline:
            step = 0.5
            for _ in range(50):  # Max 50 rounds of coordinate descent
                if evals >= budget or time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    # Try positive and negative steps
                    for direction in [1, -1]:
                        x_try = best_x.copy()
                        x_try[i] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = do_eval(x_try)
                        if f_try < best_f:
                            improved = True
                            # If improved, try to take a larger step? 
                            # For simplicity, just keep the best found.
                            # We can adapt step size
                            step = step * 1.2
                    # Reset step size if no improvement in this coordinate
                    if not improved:
                        step = max(0.1, step * 0.5)
                if not improved:
                    break

        return best_x, best_f
