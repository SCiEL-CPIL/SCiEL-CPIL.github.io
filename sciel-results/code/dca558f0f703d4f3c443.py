"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Portfolio optimizer combining CMA-ES with local refinement for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 45s safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        lb, ub = self.bounds

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = [0]

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def check_deadline():
            return time.time() >= deadline or call_count[0] >= budget

        def clip(x):
            return np.clip(x, lb, ub)

        # Phase 1: Coarse exploration with multiple random restarts
        # Use a mix of uniform sampling and golden-ratio spaced points
        n_explore = min(50, budget // 10)
        rng = np.random.default_rng(12345)

        # Add center point
        center = np.zeros(dim)
        f_center = func(center)
        call_count[0] += 1
        update_best(center, f_center)

        # Uniform random samples
        for _ in range(n_explore - 1):
            if check_deadline():
                break
            x = rng.uniform(lb, ub, dim)
            fx = func(x)
            call_count[0] += 1
            update_best(x, fx)

        # Phase 2: CMA-ES with adaptive parameters
        # Start from the best point found in exploration
        x0 = best_x.copy()
        sigma0 = 2.0  # Initial step size

        # CMA-ES parameters
        pop_size = max(4, int(np.ceil(4 + 3 * np.log(dim))))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        c_sigma = (mu_eff + 2) / ((dim + 2) * (mu_eff + 0.5))
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (mu_eff - 2 + 2 / (dim + mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        mu_eff = 1 / sum(weights ** 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # CMA-ES state
        mean = x0.copy()
        sigma = sigma0
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Track progress for restarts
        best_f_cma = best_f
        stagnation = 0
        max_stagnation = 15

        while call_count[0] < budget and time.time() < deadline:
            # Generate population
            offspring = []
            for i in range(pop_size):
                if check_deadline():
                    break
                z = rng.standard_normal(dim)
                y = B @ (D * z)
                x = mean + sigma * y
                x = clip(x)
                offspring.append((x, i))

            if not offspring:
                break

            # Evaluate offspring
            offspring_f = []
            for x, i in offspring:
                fx = func(x)
                call_count[0] += 1
                offspring_f.append((fx, x, i))
                update_best(x, fx)

            # Sort by fitness
            offspring_f.sort(key=lambda t: t[0])

            # Update mean with weighted recombination
            new_mean = np.zeros(dim)
            for i in range(min(mu, len(offspring_f))):
                new_mean += weights[i] * offspring_f[i][1]
            mean = new_mean

            # Update covariance matrix
            # Calculate evolution path
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / D @ (B.T @ (mean - x0) / sigma)
            ps_norm = np.linalg.norm(ps) / chi_n

            # Update C
            delta_sig = 1 if ps_norm < (1.4 + 1 / dim) * (1 - c_c) else 0
            C = (1 + c_1 * delta_sig - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * sum(
                weights[i] * np.outer(offspring_f[i][1] - x0, offspring_f[i][1] - x0) for i in range(min(mu, len(offspring_f)))
            ) / sigma ** 2

            # Update pc
            pc = (1 - c_c) * pc + c_c * (mean - x0) / sigma

            # Eigen-decompose for B and D
            # Use SVD for numerical stability
            U, s, Vt = np.linalg.svd(C)
            D = np.sqrt(s)
            B = U

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (ps_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Check for stagnation
            if offspring_f[0][0] < best_f_cma - 1e-10:
                best_f_cma = offspring_f[0][0]
                stagnation = 0
            else:
                stagnation += 1

            # Restart if stagnant too long
            if stagnation >= max_stagnation:
                # Try a local refinement around the best point
                break

        # Phase 3: Local refinement with Nelder-Mead style simplex
        if call_count[0] < budget and time.time() < deadline:
            # Create initial simplex
            simplex = [best_x.copy()]
            for i in range(dim):
                perturbation = best_x.copy()
                perturbation[i] += 0.1 * sigma
                simplex.append(clip(perturbation))

            simplex_f = [func(x) for x in simplex]
            call_count[0] += dim
            update_best(best_x, best_f)

            # Nelder-Mead iterations
            for _ in range(50):
                if check_deadline():
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = clip(x_ref)
                f_ref = func(x_ref)
                call_count[0] += 1
                update_best(x_ref, f_ref)

                if simplex_f[0] <= f_ref < simplex_f[-2]:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                elif f_ref < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = clip(x_exp)
                    f_exp = func(x_exp)
                    call_count[0] += 1
                    update_best(x_exp, f_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - simplex[-1])
                    x_con = clip(x_con)
                    f_con = func(x_con)
                    call_count[0] += 1
                    update_best(x_con, f_con)

                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = clip(simplex[i])
                            simplex_f[i] = func(simplex[i])
                            call_count[0] += 1
                            update_best(simplex[i], simplex_f[i])

        return best_x, best_f
# EVOLVE-BLOCK-END
