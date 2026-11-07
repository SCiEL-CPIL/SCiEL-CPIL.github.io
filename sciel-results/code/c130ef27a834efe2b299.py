"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Nelder-Mead with CMA-ES polish for Büche-Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_deadline():
            return time.time() > deadline

        def eval_point(x):
            nonlocal calls_used
            if calls_used >= budget:
                return None
            fx = func(x)
            calls_used += 1
            return fx

        def nelder_mead(x0, max_calls, initial_step=1.0):
            nonlocal calls_used, best_x, best_f

            if max_calls < 10:
                return

            x0 = np.array(x0, dtype=float)
            n = len(x0)

            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += initial_step
                simplex.append(np.clip(xi, -5, 5))

            # Evaluate all simplex vertices
            values = []
            for i, vertex in enumerate(simplex):
                if calls_used >= budget or check_deadline():
                    return
                fv = eval_point(vertex)
                if fv is None:
                    return
                values.append(fv)

            local_best_idx = 0
            for i in range(len(values)):
                if values[i] < values[local_best_idx]:
                    local_best_idx = i

            if values[local_best_idx] < best_f:
                best_f = values[local_best_idx]
                best_x = simplex[local_best_idx].copy()

            # Nelder-Mead iterations
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            local_calls = n + 1  # initial evaluations
            max_iter = max(1, (max_calls - local_calls) // 2)

            for it in range(max_iter):
                if check_deadline() or local_calls >= max_calls:
                    break

                # Sort simplex by function value
                order = np.argsort(values)
                simplex = [simplex[i] for i in order]
                values = [values[i] for i in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_val = values[-1]

                # Reflection
                x_ref = centroid + alpha * (centroid - worst)
                x_ref = np.clip(x_ref, -5, 5)
                if local_calls >= max_calls or check_deadline():
                    break
                f_ref = eval_point(x_ref)
                if f_ref is None:
                    break
                local_calls += 1

                if f_ref < values[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    if local_calls >= max_calls or check_deadline():
                        break
                    f_exp = eval_point(x_exp)
                    if f_exp is None:
                        break
                    local_calls += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        values[-1] = f_ref
                elif f_ref < values[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    values[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (worst - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    if local_calls >= max_calls or check_deadline():
                        break
                    f_con = eval_point(x_con)
                    if f_con is None:
                        break
                    local_calls += 1

                    if f_con <= f_ref:
                        simplex[-1] = x_con
                        values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if local_calls >= max_calls or check_deadline():
                                break
                            fv = eval_point(simplex[i])
                            if fv is None:
                                break
                            local_calls += 1
                            values[i] = fv

                # Update global best
                min_idx = np.argmin(values)
                if values[min_idx] < best_f:
                    best_f = values[min_idx]
                    best_x = simplex[min_idx].copy()

        def cma_es(x0, sigma0, max_calls):
            nonlocal calls_used, best_x, best_f

            if max_calls < 10:
                return

            x0 = np.array(x0, dtype=float)
            n = len(x0)

            # CMA-ES parameters
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            x = x0.copy()
            sigma = sigma0
            ps = np.zeros(n)
            pc = np.zeros(n)
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)

            # Evaluate initial point
            fx = eval_point(x)
            if fx is None:
                return
            local_best_x = x.copy()
            local_best_f = fx

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            local_calls = 1
            max_iter = max(1, max_calls // lambda_)

            for it in range(max_iter):
                if check_deadline() or local_calls >= max_calls:
                    break

                # Generate population
                offspring = []
                for i in range(lambda_):
                    if local_calls >= max_calls or check_deadline():
                        break
                    z = np.random.standard_normal(n)
                    y = x + sigma * (B @ z)
                    y = np.clip(y, -5, 5)
                    fy = eval_point(y)
                    if fy is None:
                        break
                    local_calls += 1
                    offspring.append((fy, y))

                if len(offspring) < mu:
                    break

                # Sort
                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < local_best_f:
                    local_best_f = offspring[0][0]
                    local_best_x = offspring[0][1].copy()
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()

                # Weighted mean
                xnew = np.zeros(n)
                for i in range(mu):
                    xnew += weights[i] * offspring[i][1]

                # Step size control
                ydiff = (xnew - x) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / D * ydiff
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1))) < \
                       (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
                hs = float(hsig)
                pc = (1 - cc) * pc + hs * cc / D * ydiff

                # Covariance matrix adaptation
                rank1 = np.outer(pc, pc)
                delta_h = (1 - hs) * cc * (1 - cc)
                rankmu = np.zeros((n, n))
                for i in range(mu):
                    zi = (offspring[i][1] - x) / sigma
                    rankmu += np.outer(weights[i] * zi, weights[i] * zi)
                C = (1 + c1 * delta_h - c1) * C + c1 * rank1 + cmu * rankmu

                # Eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs

                # Update
                x = xnew
                sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(ydiff) / np.sqrt(2 * n) - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

        # Strategy: Multi-start Nelder-Mead for exploration, CMA-ES for exploitation
        # Büche-Rastrigin is multimodal; NM handles local minima well with diverse starts

        # Phase 1: Multi-start Nelder-Mead with diverse initializations
        # Use about 60% of budget for exploration
        explore_budget = int(budget * 0.6)
        n_starts = 6

        for r in range(n_starts):
            if check_deadline() or calls_used >= budget:
                break

            # Different starting points with varying scales
            if r == 0:
                x0 = np.zeros(n)
                step = 1.5
            elif r == 1:
                x0 = np.random.uniform(-4, 4, n)
                step = 1.0
            elif r == 2:
                x0 = np.random.uniform(-3, 3, n)
                step = 1.2
            elif r == 3:
                x0 = np.random.uniform(-4, 4, n) * 0.7
                step = 0.8
            elif r == 4:
                x0 = np.random.uniform(-5, 5, n)
                step = 1.5
            else:
                x0 = np.random.uniform(-4, 4, n)
                step = 1.0

            alloc = explore_budget // n_starts
            nelder_mead(x0, alloc, initial_step=step)

        # Phase 2: CMA-ES exploitation around best found point
        if best_x is not None and not check_deadline() and calls_used < budget:
            remaining = budget - calls_used
            if remaining > 20:
                cma_es(best_x, 0.3, remaining)

        # Phase 3: Fine-tuning with small-step NM if budget remains
        if best_x is not None and not check_deadline() and calls_used < budget:
            remaining = budget - calls_used
            if remaining > 10:
                nelder_mead(best_x, remaining, initial_step=0.1)

        if best_x is None:
            # Should not happen, but just in case
            best_x = np.zeros(n)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
