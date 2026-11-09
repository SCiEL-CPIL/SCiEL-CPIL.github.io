"""Improved BBOB f22 optimizer: CMA-ES + DE + Nelder-Mead + (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-strategy optimizer for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_time():
            return (time.time() - start_time) < time_limit

        def safe_func(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        rng = np.random.default_rng(7)

        # Phase 1: Random exploration to find good basins
        n_random = min(40, max(10, self.budget // 10))
        random_points = []
        for i in range(n_random):
            if not check_time() or calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = safe_func(x)
            random_points.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        random_points.sort(key=lambda t: t[0])
        
        # Additional local refinements from top 3 random points using (1+1)-ES
        top_k = min(3, len(random_points))
        for i in range(top_k):
            if not check_time() or (self.budget - calls_made) <= 0:
                break
            x_start = random_points[i][1].copy()
            f_start = random_points[i][0]
            sigma_local = 0.5
            x = x_start.copy()
            fx = f_start
            for _ in range(min(20, self.budget - calls_made)):
                if not check_time() or (self.budget - calls_made) <= 0:
                    break
                y = np.clip(x + sigma_local * rng.standard_normal(self.dim), -5, 5)
                fy = safe_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma_local *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma_local *= 0.8

        # CMA-ES implementation
        def cma_es(x0, f0, budget_remaining, max_steps):
            nonlocal best_x, best_f, calls_made

            if budget_remaining <= 0 or not check_time():
                return

            dim = self.dim
            lam = max(4, int(4 + 3 * np.log(dim)))
            sigma = 0.5
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)

            cs = (lam // 2 + 1) / (dim + lam // 2 + 5)
            ds = 1 + 2 * max(0, (dim - 1) / (dim + 2) - 1) + cs
            c1 = (2 + 2 / (dim + 1.7)) / ((dim + 2) ** 2 + lam)
            cmu = min(2 / ((dim + 1.3) ** 2 + lam), (1 - c1) * 0.5)
            cc = (4 + dim / lam) / (dim + 4 + 2 * lam / dim)

            mu = lam // 2
            weights = np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights ** 2)

            x_mean = x0.copy()

            for step in range(max_steps):
                if not check_time() or (self.budget - calls_made) <= lam:
                    break

                offspring = []
                for i in range(lam):
                    if not check_time() or (self.budget - calls_made) <= lam - i:
                        break
                    z = rng.standard_normal(dim)
                    L = np.linalg.cholesky(C)
                    y = x_mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    fy = safe_func(y)
                    offspring.append((fy, y))

                if len(offspring) == 0:
                    break

                offspring.sort(key=lambda t: t[0])

                if offspring[0][0] < best_f:
                    best_f = offspring[0][0]
                    best_x = offspring[0][1].copy()

                mu_actual = min(mu, len(offspring))
                y_w = np.zeros(dim)
                for i in range(mu_actual):
                    y_w += weights[i] * offspring[i][1]
                y_w /= np.sum(weights[:mu_actual])

                x_mean = y_w

                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (y_w - x_mean)) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (step + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))

                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (y_w - x_mean) / sigma

                delta_d = (1 + 2 * (hsig - 1) * cc * (2 - cc)) * c1
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu_actual):
                    zi = (offspring[i][1] - x_mean) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)

                C = (1 + c1 * (delta_d - 1) + cmu * (mueff * (c1 + cmu) / c1 - 1)) * C + cmu * rank_mu - c1 * rank_one

                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += (1e-12 - np.min(eigvals) + 1e-9) * np.eye(dim)

                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt((dim + 1) / (dim - 1)) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                if sigma < 1e-10 or np.max(np.abs(C)) < 1e-12:
                    break

        # Nelder-Mead local search
        def nelder_mead(x0, f0, budget_remaining, max_steps=100):
            nonlocal best_x, best_f, calls_made

            if budget_remaining <= 0 or not check_time():
                return

            dim = self.dim
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 2.0

            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.05 * (1 if xi[i] < 0 else -1)
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            simplex_f = [f0]
            for i in range(1, dim + 1):
                if not check_time() or (self.budget - calls_made) <= 0:
                    return
                fx = safe_func(simplex[i])
                simplex_f.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = simplex[i].copy()

            for step in range(max_steps):
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                if fxr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    if not check_time() or (self.budget - calls_made) <= 0:
                        break
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_func(xc)
                    if not check_time() or (self.budget - calls_made) <= 0:
                        break
                    if fxc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = safe_func(simplex[i])
                            if not check_time() or (self.budget - calls_made) <= 0:
                                return

                # Convergence check
                if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                    break

        # Differential Evolution
        def differential_evolution(x0, f0, budget_remaining, max_steps=50):
            nonlocal best_x, best_f, calls_made

            if budget_remaining <= 0 or not check_time():
                return

            dim = self.dim
            pop_size = max(8, min(20, budget_remaining // 3))
            F = 0.8
            CR = 0.5

            # Initialize population around x0
            pop = [x0.copy() + rng.standard_normal(dim) * 0.5 for _ in range(pop_size)]
            pop_f = []
            for i in range(pop_size):
                pop[i] = np.clip(pop[i], -5, 5)
                if not check_time() or (self.budget - calls_made) <= 0:
                    return
                fx = safe_func(pop[i])
                pop_f.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = pop[i].copy()

            for step in range(max_steps):
                if not check_time() or (self.budget - calls_made) <= 0:
                    break

                for i in range(pop_size):
                    if not check_time() or (self.budget - calls_made) <= 0:
                        return

                    a, b, c = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                    trial = pop[a] + F * (pop[b] - pop[c])
                    for j in range(dim):
                        if rng.random() < CR:
                            trial[j] = pop[a][j]
                    trial = np.clip(trial, -5, 5)
                    ftrial = safe_func(trial)
                    if ftrial <= pop_f[i]:
                        pop[i] = trial
                        pop_f[i] = ftrial
                        if ftrial < best_f:
                            best_f = ftrial
                            best_x = trial.copy()

        # Phase 2: CMA-ES from best random point
        if random_points and best_x is not None:
            budget_for_cma = self.budget - calls_made
            lam = max(4, int(4 + 3 * np.log(self.dim)))
            max_steps = max(1, budget_for_cma // lam)
            cma_es(best_x, best_f, budget_for_cma, max_steps)
            
            # Phase 2b: If CMA-ES didn't improve much, try from 2nd best
            if len(random_points) > 1 and (self.budget - calls_made) > 20 and check_time():
                x2 = random_points[1][1].copy()
                f2 = random_points[1][0]
                if f2 < best_f + 5: # Only if 2nd best is reasonably close
                    budget_for_cma2 = self.budget - calls_made
                    lam2 = max(4, int(4 + 3 * np.log(self.dim)))
                    max_steps2 = max(1, budget_for_cma2 // lam2)
                    cma_es(x2, f2, budget_for_cma2, max_steps2)

        # Phase 3: Nelder-Mead polish on current best
        remaining = self.budget - calls_made
        if remaining > 20 and best_x is not None and check_time():
            nelder_mead(best_x, best_f, remaining, max_steps=50)

        # Phase 4: DE from a different starting point if budget allows
        remaining = self.budget - calls_made
        if remaining > 30 and len(random_points) > 1 and check_time():
            x_start = random_points[1][1].copy()
            f_start = random_points[1][0]
            differential_evolution(x_start, f_start, remaining, max_steps=30)

        # Phase 5: Another Nelder-Mead pass
        remaining = self.budget - calls_made
        if remaining > 20 and best_x is not None and check_time():
            nelder_mead(best_x, best_f, remaining, max_steps=40)

        # Phase 6: (1+1)-ES final polish
        remaining = self.budget - calls_made
        if remaining > 0 and best_x is not None and check_time():
            sigma_local = 0.3
            x = best_x.copy()
            fx = best_f
            for _ in range(min(remaining, 100)):
                if not check_time() or (self.budget - calls_made) <= 0:
                    break
                y = np.clip(x + sigma_local * rng.standard_normal(self.dim), -5, 5)
                fy = safe_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma_local *= 1.4
                else:
                    sigma_local *= 0.85
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
