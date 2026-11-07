import numpy as np
import time
import math


class Optimizer:
    """Bayesian Optimization with GP surrogate + CMA-ES hybrid for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf
        eval_count = 0

        # Storage for GP surrogate
        X_obs = []
        Y_obs = []

        def time_left():
            return deadline - time.time()

        def do_eval(x):
            nonlocal eval_count, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            X_obs.append(x.copy())
            Y_obs.append(fx)
            return fx

        # --- Gaussian Process Surrogate ---
        def gp_predict(X_new, X_train, Y_train, length_scale=1.0, variance=1.0, noise=1e-6):
            """
            Simple GP prediction with RBF kernel.
            Returns: mean, std
            """
            if len(X_train) == 0:
                return np.zeros(len(X_new)), np.ones(len(X_new))

            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            X_new = np.array(X_new)

            # RBF kernel: k(x, x') = variance * exp(-||x-x'||^2 / (2 * length_scale^2))
            def kernel(A, B):
                # A: (n, dim), B: (m, dim)
                A2 = np.sum(A**2, axis=1).reshape(-1, 1)
                B2 = np.sum(B**2, axis=1).reshape(1, -1)
                dist2 = A2 + B2 - 2 * A @ B.T
                dist2 = np.maximum(dist2, 0)
                return variance * np.exp(-dist2 / (2 * length_scale**2))

            K = kernel(X_train, X_train) + noise * np.eye(len(X_train))
            K_s = kernel(X_train, X_new)
            K_ss = kernel(X_new, X_new)

            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                v = np.linalg.solve(L, K_s.T)
            except np.linalg.LinAlgError:
                # Fallback if not PD
                K = K + 1e-6 * np.eye(len(X_train))
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                v = np.linalg.solve(L, K_s.T)

            mean = K_s.T @ alpha
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
            std = np.sqrt(np.maximum(var, 1e-12))
            return mean, std

        # --- Acquisition Function: Expected Improvement (EI) ---
        def acquisition_ei(X_new, X_train, Y_train, f_best, length_scale, variance):
            mean, std = gp_predict(X_new, X_train, Y_train, length_scale, variance)
            # Normalize for stability
            Y_arr = np.array(Y_train)
            f_min = f_best
            # EI = (f_best - mean) * Phi(Z) + std * phi(Z)
            # where Z = (f_best - mean) / std
            # But we want to minimize, so EI = (f_best - mean) * Phi((f_best - mean)/std) + std * pdf((f_best - mean)/std)
            # Actually, standard EI for minimization:
            # Let mu = mean, sigma = std
            # Z = (f_best - mu) / sigma
            # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
            # If sigma is 0, EI = max(0, f_best - mu)
            
            Z = (f_best - mean) / np.maximum(std, 1e-8)
            # Phi and phi
            Phi = 0.5 * (1 + np.vectorize(math.erf)(Z / math.sqrt(2)))
            phi = np.exp(-0.5 * Z**2) / np.sqrt(2 * math.pi)
            EI = (f_best - mean) * Phi + std * phi
            EI = np.maximum(EI, 0)
            return EI

        # --- CMA-ES Implementation (Streamlined) ---
        def run_cmaes(mean_init, sigma_init, max_evals):
            nonlocal eval_count, best_x, best_f
            if time_left() < 5.0 or max_evals <= 0:
                return

            mean = np.clip(mean_init.copy(), -5, 5)
            sigma = sigma_init
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            local_evals = 0
            no_improve = 0
            best_local_f = best_f

            do_eval(mean)
            local_evals += 1

            lam = max(4, int(4 + 2 * math.log(dim)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / weights.sum()
            mueff = 1.0 / weights.sum()

            c1 = (2 * (mueff - 2 + 2 / dim)) / ((dim + 2) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / dim)) / ((dim + 2) ** 2 + mueff)
            cs = (lam / 2 + 1) / (dim + lam / 2 + 3)
            ds = 1 + 2 * max(0, math.sqrt((lam / dim - 1) / (dim + 1)) - 1) + cs

            while local_evals < max_evals and time_left() > 5.0:
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    x = mean + sigma * z
                    x = np.clip(x, -5, 5)
                    offspring.append(x)

                values = []
                for x in offspring:
                    if local_evals >= max_evals or time_left() < 2.0:
                        break
                    fx = do_eval(x)
                    values.append(fx)
                    local_evals += 1

                if len(values) < mu:
                    break

                idx = np.argsort(values)
                x_sorted = [offspring[i] for i in idx]
                f_sorted = [values[i] for i in idx]

                if f_sorted[0] < best_local_f - 1e-10:
                    best_local_f = f_sorted[0]
                    no_improve = 0
                else:
                    no_improve += 1

                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * (x_sorted[i] - mean) / sigma
                old_mean = mean.copy()
                mean = mean + sigma * y_w
                mean = np.clip(mean, -5, 5)

                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * y_w
                chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (local_evals / (cs * (dim + 1))))) < \
                       (1.4 + 2 / (dim + 1)) * chi_n
                sigma = sigma * math.exp((cs / ds) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-15, 50.0)

                delta_hsig = (1 - hsig) * cs * (2 - cs)
                pc = (1 - cmu) * pc + delta_hsig * (y_w + delta_hsig * ps)
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    zi = (x_sorted[i] - old_mean) / sigma
                    rankmu += weights[i] * np.outer(zi, zi)
                C = (1 + c1 * (2 - c1 - delta_hsig) + cmu) * C - c1 * rank1 - cmu * rankmu

                eigvals, eigvecs = np.linalg.eigh(C)
                min_eig = np.min(eigvals)
                if min_eig < 1e-12:
                    C = C + (1e-12 - min_eig) * np.eye(dim)
                    eigvals, eigvecs = np.linalg.eigh(C)
                C = eigvecs @ np.diag(np.clip(eigvals, 1e-12, None)) @ eigvecs.T

                if sigma < 1e-12 or no_improve > 30:
                    break

        # --- Phase 1: Initial Random Sampling to seed GP ---
        n_init = 20
        for i in range(n_init):
            if eval_count >= budget or time_left() < 10:
                break
            x = rng.uniform(-5, 5, dim)
            do_eval(x)

        # --- Phase 2: Bayesian Optimization Loop ---
        # Use GP to guide search. We'll alternate between GP-guided steps and CMA-ES restarts.
        # GP parameters: length_scale and variance are fixed for simplicity, or adapted.
        # For Büche-Rastrigin, length_scale ~ 1.0 is reasonable.
        
        gp_length_scale = 1.0
        gp_variance = 1.0
        
        # We will run BO for a while, then switch to CMA-ES for local refinement.
        # Or interleave. Let's do BO first to find a good region, then CMA-ES.
        
        bo_evals_budget = 150
        bo_count = 0
        
        while bo_count < bo_evals_budget and eval_count < budget and time_left() > 20:
            # Generate candidate points
            # 1. Random points
            n_rand = 10
            X_rand = rng.uniform(-5, 5, (n_rand, dim))
            
            # 2. GP-guided points
            # Sample many candidates and pick the one with highest EI
            n_candidates = 100
            X_cand = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Combine
            X_pool = np.vstack([X_rand, X_cand])
            
            # Compute EI
            ei_vals = acquisition_ei(X_pool, X_obs, Y_obs, best_f, gp_length_scale, gp_variance)
            
            # Pick best EI
            idx_best = np.argmax(ei_vals)
            x_next = X_pool[idx_best]
            
            # Evaluate
            fx = do_eval(x_next)
            bo_count += 1
            
            # Adaptive length scale: shrink if we are improving fast, expand if stuck
            # Simple heuristic: if best_f improved in last 5 evals, shrink LS slightly
            if len(Y_obs) > 5:
                recent_improvement = Y_obs[-5] > Y_obs[-1]
                if recent_improvement:
                    gp_length_scale = max(0.5, gp_length_scale * 0.9)
                else:
                    gp_length_scale = min(2.0, gp_length_scale * 1.05)

        # --- Phase 3: CMA-ES Refinement around best point ---
        # Use remaining budget for CMA-ES
        remaining = budget - eval_count
        if remaining > 20 and time_left() > 10:
            run_cmaes(best_x.copy(), 1.0, min(remaining, 100))
            
        if time_left() > 10 and eval_count < budget:
            run_cmaes(best_x.copy(), 0.3, min(budget - eval_count, 80))
            
        if time_left() > 5 and eval_count < budget:
            run_cmaes(best_x.copy(), 0.05, min(budget - eval_count, 50))

        # --- Phase 4: Final GP polish if budget remains ---
        if eval_count < budget and time_left() > 5:
            # Do a few more BO steps
            for _ in range(20):
                if eval_count >= budget or time_left() < 2:
                    break
                X_cand = rng.uniform(-5, 5, (50, dim))
                ei_vals = acquisition_ei(X_cand, X_obs, Y_obs, best_f, gp_length_scale, gp_variance)
                idx_best = np.argmax(ei_vals)
                x_next = X_cand[idx_best]
                do_eval(x_next)

        return best_x, best_f
