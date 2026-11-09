"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart for Rastrigin on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_left():
            return deadline - time.time()

        # Global best tracking
        global_best_x = None
        global_best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, global_best_x, global_best_f
            if calls_used >= budget:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            calls_used += 1
            if f_val < global_best_f:
                global_best_f = f_val
                global_best_x = x_clipped.copy()
            return f_val

        # Phase 1: Broad random exploration to find multiple good basins
        explore_budget = min(50, int(budget * 0.1))
        explore_points = []
        for i in range(explore_budget):
            if time_left() < 5:
                break
            x = rng.uniform(-5, 5, n)
            fx = evaluate(x)
            if fx is None:
                break
            explore_points.append((fx, x.copy()))

        if global_best_x is None:
            # Fallback: evaluate origin
            fx = evaluate(np.zeros(n))
            if fx is not None:
                global_best_f = fx
                global_best_x = np.zeros(n)
            else:
                return np.zeros(n), np.inf

        # Sort exploration points by fitness
        explore_points.sort(key=lambda t: t[0])

        # Select diverse starting points for CMA-ES restarts
        # Take the best point, then greedily add points that are far from existing selected
        start_points = [explore_points[0][1].copy()]
        start_fits = [explore_points[0][0]]
        min_dist = 1.5  # Minimum distance between selected starts

        for fx, x in explore_points[1:]:
            if len(start_points) >= 4:  # Max 4 restart points
                break
            diverse = True
            for sp in start_points:
                if np.linalg.norm(x - sp) < min_dist:
                    diverse = False
                    break
            if diverse:
                start_points.append(x.copy())
                start_fits.append(fx)

        # Phase 2: CMA-ES with restarts from diverse starting points
        # CMA-ES parameters (tuned for n=5)
        lam = 10
        mu = lam // 2

        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = 2 * (mueff - 2 + np.sum(w[:mu] ** 2) / np.sum(w) ** 2) / ((n + 2) ** 2 + mueff)
        cmu = min(cmu, 2 * (1 - c1))
        cs = (mueff + 2) / (n + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        c_c = (4 - mueff + cs) / (n + 4.3)

        chi_n = np.sqrt(n) * (1 + 1 / (4 * n) - 1 / (21 * n ** 2))

        for restart_idx, (sp, sf) in enumerate(start_points):
            if calls_used >= budget or time_left() < 15:
                break

            # Initialize CMA-ES state
            mean = sp.copy()
            sigma = 1.5 / (restart_idx + 1)  # Decreasing sigma for later restarts
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            gen = 0

            # Local best for this restart
            local_best_f = sf
            local_stagnant = 0
            best_local_f = sf

            while calls_used < budget and time_left() > 10:
                gen += 1

                # Generate population
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    L = np.eye(n)

                Z = rng.standard_normal((lam, n))
                offspring = mean + sigma * (Z @ L.T)
                offspring = np.clip(offspring, -5, 5)

                # Evaluate offspring
                offspring_f = np.empty(lam)
                eval_count = 0
                for i in range(lam):
                    if calls_used >= budget or time_left() < 5:
                        break
                    offspring_f[i] = func(offspring[i])
                    calls_used += 1
                    eval_count += 1
                    if offspring_f[i] < global_best_f:
                        global_best_f = offspring_f[i]
                        global_best_x = offspring[i].copy()

                if eval_count == 0:
                    break
                if eval_count < lam:
                    offspring = offspring[:eval_count]
                    offspring_f = offspring_f[:eval_count]
                    lam_actual = eval_count
                    mu_actual = min(mu, lam_actual)
                else:
                    lam_actual = lam
                    mu_actual = mu

                # Sort by fitness
                order = np.argsort(offspring_f)
                selected_y = offspring[order[:mu_actual]]
                selected_f = offspring_f[order[:mu_actual]]

                # Update mean
                m_old = mean.copy()
                mean_new = np.sum([w[i] * selected_y[i] for i in range(mu_actual)], axis=0)

                # Update evolution paths
                try:
                    C_inv_sqrt = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(n)

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (mean_new - m_old) / sigma

                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) < (1.4 + 2 / (n + 1)) * chi_n

                if hsig:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (mean_new - m_old) / sigma

                # Update covariance matrix
                delta_sig = 1.0 if hsig else 0.0
                C = (1 - c1 - cmu) * C
                C += c1 * (1 - delta_sig) * ps @ ps.T + c1 * delta_sig * (ps @ ps.T + (1 - (1 - cs) ** 2) * pc @ pc.T)
                C += cmu * np.sum([w[i] * np.outer(selected_y[i] - m_old, selected_y[i] - m_old) for i in range(mu_actual)], axis=0)

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += 1e-12 * np.eye(n)

                mean = mean_new

                # Adapt sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(1e-10, min(sigma, 10.0))

                # Check convergence
                if np.linalg.norm(mean_new - m_old) < 1e-9 * max(1, np.linalg.norm(mean_new)):
                    local_stagnant += 1
                else:
                    local_stagnant = 0

                if local_stagnant > 20:
                    break

            # If we made significant progress, continue to next restart
            if global_best_f < sf - 1e-4:
                continue
            else:
                # No progress from this restart, try next
                continue

        # Phase 3: Final polish with coordinate descent
        if time_left() > 20 and calls_used < budget and global_best_x is not None:
            polish_budget = min(80, budget - calls_used)
            x_cur = global_best_x.copy()
            f_cur = global_best_f
            step = 0.1

            for polish_iter in range(polish_budget):
                if calls_used >= budget or time_left() < 5:
                    break
                improved = False
                for dim_i in range(n):
                    if calls_used >= budget or time_left() < 5:
                        break
                    for direction in [1, -1]:
                        if calls_used >= budget or time_left() < 5:
                            break
                        x_trial = x_cur.copy()
                        x_trial[dim_i] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        calls_used += 1
                        if f_trial < f_cur:
                            x_cur = x_trial
                            f_cur = f_trial
                            improved = True
                            if f_trial < global_best_f:
                                global_best_f = f_trial
                                global_best_x = x_cur.copy()
                    if improved:
                        step *= 0.5
                    else:
                        step *= 0.7
                if step < 1e-6:
                    break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
