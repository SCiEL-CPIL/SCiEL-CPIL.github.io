import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= total_budget:
                return np.inf
            calls_made += 1
            try:
                fx = func(np.asarray(x, dtype=float))
                if np.isfinite(fx):
                    return float(fx)
                return np.inf
            except Exception:
                return np.inf

        rng = np.random.default_rng(42)

        # Initial exploration
        n_init = min(10, total_budget // 10)
        init_points = [np.zeros(dim)]
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-5, 5, dim))

        for p in init_points:
            if calls_made >= total_budget or time.time() > deadline - 10:
                break
            fx = safe_call(p)
            if fx < best_f:
                best_f = fx
                best_x = p.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (2 + mueff / dim) / (dim + 2 + mueff / dim)
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (1 - cc - cs)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        hsig_threshold = (1.4 + 2 / (dim + 1)) * np.sqrt(dim * (1 - 1 / dim ** 2))

        def run_cmaes_restart(x_start, sigma_start, max_evals, restart_threshold=1e-6):
            nonlocal calls_made, best_f, best_x, deadline

            if max_evals <= 0 or time.time() > deadline - 5:
                return x_start, best_f

            x_mean = x_start.copy()
            sigma = sigma_start
            C = np.eye(dim)
            pc = np.zeros(dim)
            p_sigma = np.zeros(dim)

            best_x_local = x_start.copy()
            best_f_local = best_f
            evals_in_run = 0
            gen = 0
            last_best_f = best_f
            stagnation_counter = 0

            while evals_in_run < max_evals:
                if time.time() > deadline - 5:
                    break
                if calls_made + lambda_ > total_budget:
                    break

                # Check for stagnation
                if best_f_local < last_best_f - restart_threshold:
                    last_best_f = best_f_local
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    if stagnation_counter > 5:
                        break # Restart

                # Generate offspring
                try:
                    B = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    C = (C + C.T) / 2
                    B = np.linalg.cholesky(C + 1e-10 * np.eye(dim))

                z = rng.standard_normal((lambda_, dim))
                Y = z @ B.T
                Offspring = x_mean[None, :] + sigma * Y
                Offspring = np.clip(Offspring, -5, 5)

                # Evaluate offspring
                Offspring_F = []
                for i in range(lambda_):
                    if calls_made >= total_budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    fi = safe_call(Offspring[i])
                    Offspring_F.append(fi)
                    evals_in_run += 1
                    if fi < best_f_local:
                        best_f_local = fi
                        best_x_local = Offspring[i].copy()

                if len(Offspring_F) < lambda_:
                    break

                Offspring_F = np.array(Offspring_F)
                order = np.argsort(Offspring_F)
                Offspring = Offspring[order]
                Offspring_F = Offspring_F[order]

                if Offspring_F[0] < best_f:
                    best_f = Offspring_F[0]
                    best_x = Offspring[0].copy()

                gen += 1

                # Update mean
                x_mean_new = np.sum(weights[:, np.newaxis] * Offspring[:mu], axis=0)

                # Update paths
                y_diff = (x_mean_new - x_mean) / (sigma + 1e-300)
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * y_diff
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_diff

                # Sigma adaptation
                ps_norm = np.linalg.norm(p_sigma)
                h_sigma = 1.0 if ps_norm / np.sqrt(1 - (1 - cs) ** (2 * gen)) < hsig_threshold else 0.0
                sigma *= np.exp((cs / 5) * (ps_norm / np.sqrt(1 - (1 - cs) ** (2 * gen)) - hsig_threshold))

                # Update covariance
                delta_h = 1 + 2 * max(0, ps_norm / np.sqrt(1 - (1 - cs) ** (2 * gen)) - 1) * h_sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (cc * (2 - cc)) * delta_h * np.eye(dim))

                # Weighted outer products
                Y_diff = (Offspring[:mu] - x_mean[None, :]) / (sigma + 1e-300)
                C += cmu * np.sum(weights[:, np.newaxis, np.newaxis] * Y_diff[:, :, np.newaxis] * Y_diff[:, np.newaxis, :], axis=0)

                # Symmetrize and ensure positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.any(eigvals < 1e-12):
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                x_mean = x_mean_new

            return best_x_local, best_f_local

        # Multi-restart CMA-ES
        # We divide the remaining budget into chunks for restarts
        remaining_budget = total_budget - calls_made
        num_restarts = 5
        budget_per_restart = remaining_budget // num_restarts
        
        current_x = best_x.copy()
        current_sigma = 1.0

        for i in range(num_restarts):
            if calls_made >= total_budget or time.time() > deadline - 10:
                break
            
            # Decrease sigma slightly for each restart to focus on exploitation
            # But keep it large enough to escape local minima if the previous one failed
            # For f19, starting with 1.0 is reasonable, then maybe 0.5, 0.2, etc.
            # However, if we restart, we should probably keep the sigma from the previous run 
            # or reset it to a value proportional to the spread of the population.
            # Here we just use a decaying schedule for the initial sigma of the restart.
            current_sigma = max(0.1, current_sigma * 0.5)

            new_best_x, new_best_f = run_cmaes_restart(current_x, current_sigma, budget_per_restart)
            
            if new_best_f < best_f:
                best_f = new_best_f
                best_x = new_best_x.copy()
                current_x = new_best_x.copy()
            else:
                # If no improvement, keep the best found so far as the center
                current_x = best_x.copy()
                # Maybe increase sigma slightly if we are stuck? 
                # For now, just continue with the decaying schedule.

        # Final polish with Nelder-Mead
        if calls_made < total_budget and time.time() < deadline - 5:
            # Nelder-Mead implementation
            def nelder_mead(func, x0, max_iter=100, tol=1e-10):
                nonlocal calls_made, best_f, best_x
                dim = len(x0)
                # Initialize simplex
                simplex = [x0.copy() for _ in range(dim + 1)]
                simplex[0] = x0
                for i in range(dim):
                    simplex[i+1] = x0.copy()
                    simplex[i+1][i] += 0.05 if x0[i] < 4.95 else -0.05
                    
                fx = [safe_call(s) for s in simplex]
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_nm = 0.5
                
                for _ in range(max_iter):
                    if calls_made >= total_budget:
                        break
                    if time.time() > deadline - 1:
                        break
                    
                    # Sort simplex
                    indices = np.argsort(fx)
                    simplex = [simplex[i] for i in indices]
                    fx = [fx[i] for i in indices]
                    
                    if fx[0] < best_f:
                        best_f = fx[0]
                        best_x = simplex[0].copy()
                        
                    # Check convergence
                    if max(fx) - min(fx) < tol:
                        break
                        
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5, 5)
                    f_r = safe_call(x_r)
                    
                    if f_r < fx[0]:
                        # Expansion
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = safe_call(x_e)
                        if f_e < f_r:
                            simplex[-1] = x_e
                            fx[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            fx[-1] = f_r
                    elif f_r < fx[-2]:
                        simplex[-1] = x_r
                        fx[-1] = f_r
                    else:
                        # Contraction
                        if f_r < fx[-1]:
                            # Outside contraction
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            # Inside contraction
                            x_c = centroid - rho * (simplex[-1] - centroid)
                        
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_call(x_c)
                        
                        if f_c < min(f_r, fx[-1]):
                            simplex[-1] = x_c
                            fx[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fx[i] = safe_call(simplex[i])
                            
                            if calls_made >= total_budget:
                                break
                            
                            # Re-sort
                            indices = np.argsort(fx)
                            simplex = [simplex[i] for i in indices]
                            fx = [fx[i] for i in indices]

                return simplex[0], fx[0]

            nelder_mead(func, best_x)

        return best_x, best_f
