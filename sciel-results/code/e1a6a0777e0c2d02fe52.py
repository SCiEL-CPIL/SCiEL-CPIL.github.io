"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Improved CMA-ES with better initialization and adaptive restarts for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin of 50s
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        total_calls = 0

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 3:
                return None
            x = np.clip(x, -5, 5)
            f = func(x)
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Adaptive random search to find good starting points
        n_random_phase1 = min(20, budget // 10)
        
        # Try center and a few structured points
        structured_points = [
            np.zeros(n),
            np.ones(n) * 3,
            -np.ones(n) * 3,
        ]
        for p in structured_points:
            if total_calls >= budget:
                break
            try_eval(p)
        
        # Random samples with decreasing scale
        for i in range(n_random_phase1):
            if total_calls >= budget:
                break
            if time.time() > deadline - 5:
                break
            scale = 3.5 - 2.0 * (i / max(1, n_random_phase1))
            x_try = rng.normal(0, scale, n)
            try_eval(x_try)

        # Phase 2: Run CMA-ES from the best point found
        def cma_es(x0, sigma0, max_calls):
            """CMA-ES that respects global total_calls and deadline."""
            nonlocal total_calls, best_x, best_f
            
            x = x0.copy()
            sigma = sigma0
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            
            # CMA-ES parameters tuned for 5D
            lambda_ = min(max(4 * n + 2, 12), max(6, max_calls // 4))
            mu = max(1, lambda_ // 2)
            
            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (2 - cc)
            cmu = min(1 - c1, (2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff)) * (2 - cc))

            hsig = (1 - (1 - cc) ** (1 / cs)) / np.sqrt(1 - (1 - cc) ** (2 / cs))

            best_x_local = x0.copy()
            best_f_local = best_f  # Start with global best
            
            no_improve = 0
            patience = max(15, lambda_ * 4)

            while total_calls < budget:
                if time.time() > deadline - 5:
                    break
                
                # Generate population
                ys = []
                fs = []
                for i in range(lambda_):
                    if total_calls >= budget:
                        break
                    if time.time() > deadline - 3:
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * z
                    y = np.clip(y, -5, 5)
                    fy = try_eval(y)
                    if fy is None:
                        break
                    ys.append(y)
                    fs.append(fy)
                    if fy < best_f_local:
                        best_f_local = fy
                        best_x_local = y.copy()
                        no_improve = 0
                    else:
                        no_improve += 1

                if len(ys) < mu:
                    break

                # Sort by fitness
                sorted_idx = np.argsort(fs)
                ys_selected = np.array([ys[i] for i in sorted_idx[:mu]])
                fs_selected = [fs[i] for i in sorted_idx[:mu]]

                # Update mean
                x_old = x.copy()
                x = np.dot(weights, ys_selected)
                x = np.clip(x, -5, 5)

                # Compute C_sqrt_inv using eigendecomposition (efficient for 5D)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-14)
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                d = x - x_old
                C_sqrt_inv_d = C_sqrt_inv @ d
                norm_d = np.linalg.norm(C_sqrt_inv_d)

                # Update sigma
                if norm_d > 1e-12:
                    p_sigma = (1 - cs) * p_sigma + cs * (norm_d / np.sqrt(2) - 1) * C_sqrt_inv_d / norm_d
                    sigma *= np.exp(cs / (n + 2) * (norm_d / np.sqrt(2) - 1))
                    sigma = np.clip(sigma, 1e-12, 5.0)

                # Update C (rank-1)
                C = (1 + c1 * cc) * C + c1 * np.outer(d, d)

                # Rank-mu update
                y_diffs = ys_selected - x_old
                y_diffs_transformed = np.array([C_sqrt_inv @ yd for yd in y_diffs])
                h = np.dot(weights, y_diffs_transformed)
                C += cmu * np.outer(h, h)

                # Update p_c
                if norm_d > 1e-12:
                    p_c = (1 - cc) * p_c + cc * hsig * C_sqrt_inv_d / np.sqrt(2)
                # Update C with p_c
                C = (1 - c1 * cc * (2 - cc)) * C + c1 * (1 + cc * (2 - cc)) * np.outer(p_c, p_c)

                # Ensure C is positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-12:
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T

                # Check for convergence/stagnation
                if no_improve >= patience and sigma < 1e-5:
                    break
                if sigma < 1e-8:
                    break

            return best_x_local, best_f_local

        # Run CMA-ES from best found point with smaller initial sigma
        if budget - total_calls > 15 and best_x is not None:
            x_opt, f_opt = cma_es(best_x, 1.0, budget - total_calls)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # Phase 3: One restart from a different location if budget remains
        if total_calls < budget - 20 and time.time() < deadline - 60:
            x_try = rng.uniform(-4, 4, n)
            f_try = try_eval(x_try)
            if f_try is not None and f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
            
            if budget - total_calls > 15:
                x_opt, f_opt = cma_es(best_x, 0.8, budget - total_calls)
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()

        # Final high-precision polish: coordinate descent + small perturbations
        if total_calls < budget and best_x is not None and time.time() < deadline - 10:
            # Coordinate descent polish
            step = 0.05
            for _ in range(3):
                if total_calls >= budget or time.time() > deadline - 5:
                    break
                for coord in range(n):
                    if total_calls >= budget or time.time() > deadline - 5:
                        break
                    for direction in [-1, 1]:
                        x_try = best_x.copy()
                        x_try[coord] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = try_eval(x_try)
                        if f_try is None:
                            break
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                step *= 0.5
            
            # Final random perturbations with very small scale
            n_final = min(10, budget - total_calls)
            for i in range(n_final):
                if total_calls >= budget:
                    break
                if time.time() > deadline - 3:
                    break
                perturbation = rng.normal(0, 0.01, n)
                x_try = best_x + perturbation
                f_try = try_eval(x_try)

        return best_x, best_f
# EVOLVE-BLOCK-END
