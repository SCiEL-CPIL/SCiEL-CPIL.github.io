import numpy as np
import time
import math
import random

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # Safety margin
        dim = self.dim
        budget = self.budget

        # Global best tracking
        self.best_x = np.zeros(dim)
        self.best_f = float('inf')
        self.calls = 0

        def evaluate(x):
            if time.time() > deadline:
                raise TimeoutError("Time Limit Exceeded")
            if self.calls >= budget:
                raise Exception("Budget Exceeded")
            self.calls += 1
            x = np.asarray(x, dtype=float).flatten()
            if x.shape != (dim,):
                try:
                    x = x.reshape(dim)
                except:
                    x = np.zeros(dim)
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = float('inf')
            except Exception:
                f = float('inf')
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        def time_left():
            return deadline - time.time()

        def calls_left():
            return budget - self.calls

        # CMA-ES implementation with careful budget management
        def run_cma_es(x0, sigma0, max_calls, pop_size=None):
            if max_calls <= 0:
                return self.best_x.copy(), self.best_f

            local_best_x = self.best_x.copy()
            local_best_f = self.best_f
            local_calls = 0

            # Evaluate initial point
            f0 = evaluate(x0)
            local_calls += 1
            if f0 < local_best_f:
                local_best_f = f0
                local_best_x = x0.copy()

            if local_calls >= max_calls:
                return local_best_x, local_best_f

            if pop_size is None:
                lam = max(4, int(4 + 3 * math.log(dim)))
            else:
                lam = pop_size

            mu = max(1, lam // 2)

            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w**2)

            c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
            d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
            c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
            c_1 = 2.0 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1.0 - c_1, 2.0 * ((mu_eff + 2.0) / (dim + 2.0) - 1.0))

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            chi_n = math.sqrt(dim) * (1.0 - 1.0/(4.0*dim) + 1.0/(21.0*dim**2))

            while local_calls < max_calls and time_left() > 2.0:
                n_off = min(lam, max_calls - local_calls)
                if n_off <= 0:
                    break

                samples = []
                try:
                    L = np.linalg.cholesky(C)
                    for _ in range(n_off):
                        if local_calls >= max_calls or time_left() < 2.0:
                            break
                        z = np.random.randn(dim)
                        y = x_mean + sigma * (L @ z)
                        y = np.clip(y, -5.0, 5.0)
                        fy = evaluate(y)
                        local_calls += 1
                        samples.append((y, fy))
                        if fy < local_best_f:
                            local_best_f = fy
                            local_best_x = y.copy()
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    std_eig = np.sqrt(eigvals)
                    for _ in range(n_off):
                        if local_calls >= max_calls or time_left() < 2.0:
                            break
                        z = np.random.randn(dim)
                        y = x_mean + sigma * ((eigvecs * std_eig) @ z)
                        y = np.clip(y, -5.0, 5.0)
                        fy = evaluate(y)
                        local_calls += 1
                        samples.append((y, fy))
                        if fy < local_best_f:
                            local_best_f = fy
                            local_best_x = y.copy()

                if len(samples) < mu or time_left() < 2.0 or local_calls >= max_calls:
                    break

                samples.sort(key=lambda x: x[1])

                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += w[i] * samples[i][0]
                x_new_mean = y_w

                diff = x_new_mean - x_mean
                norm_diff = np.linalg.norm(diff)

                if norm_diff < 1e-14:
                    break

                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
                except:
                    C_inv_sqrt = np.eye(dim)

                p_sigma = (1 - c_sigma) * p_sigma + \
                          math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ diff) / sigma

                norm_p_sigma = np.linalg.norm(p_sigma)

                h_sigma = 1.0
                if norm_p_sigma > 1.4 * chi_n:
                    h_sigma = 0.0

                p_c = (1 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ diff) / sigma

                sigma = sigma * math.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)

                c1_term = c_1 * np.outer(p_c, p_c)

                cmu_term = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = samples[i][0]
                    diff_i = y_i - x_new_mean
                    cmu_term += w[i] * np.outer(diff_i, diff_i)

                C = (1 - c_1 - c_mu + mu_eff * c_mu) * C + c1_term + c_mu * cmu_term
                C = (C + C.T) / 2.0

                x_mean = x_new_mean

            return local_best_x, local_best_f

        # Nelder-Mead implementation
        def run_nelder_mead(x0, max_calls):
            if max_calls <= 0:
                return self.best_x.copy(), self.best_f

            local_best_x = self.best_x.copy()
            local_best_f = self.best_f
            local_calls = 0

            f0 = evaluate(x0)
            local_calls += 1
            if f0 < local_best_f:
                local_best_f = f0
                local_best_x = x0.copy()

            if local_calls >= max_calls:
                return local_best_x, local_best_f

            simplex = [x0.copy()]
            for i in range(dim):
                if local_calls >= max_calls or time_left() < 2.0:
                    break
                xi = x0.copy()
                xi[i] += 0.1
                simplex.append(xi)
                fi = evaluate(xi)
                local_calls += 1
                if fi < local_best_f:
                    local_best_f = fi
                    local_best_x = xi.copy()

            if len(simplex) < 2:
                return local_best_x, local_best_f

            f_vals = []
            for i in range(len(simplex)):
                if i == 0:
                    f_vals.append(f0)
                else:
                    # Already evaluated during simplex construction
                    # We need to track these - let me restructure
                    pass

            # Actually, let me redo this more carefully
            simplex = [x0.copy()]
            f_vals = [f0]
            for i in range(dim):
                if local_calls >= max_calls or time_left() < 2.0:
                    break
                xi = x0.copy()
                xi[i] += 0.1
                simplex.append(xi)
                fi = evaluate(xi)
                local_calls += 1
                f_vals.append(fi)
                if fi < local_best_f:
                    local_best_f = fi
                    local_best_x = xi.copy()

            if len(simplex) < 2:
                return local_best_x, local_best_f

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 2.0

            while local_calls < max_calls and time_left() > 2.0:
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

                x_centroid = np.mean(simplex[:-1], axis=0)

                if local_calls >= max_calls or time_left() < 2.0:
                    break
                x_reflect = x_centroid + alpha * (x_centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                f_reflect = evaluate(x_reflect)
                local_calls += 1
                if f_reflect < local_best_f:
                    local_best_f = f_reflect
                    local_best_x = x_reflect.copy()

                if f_reflect < f_vals[0]:
                    if local_calls >= max_calls or time_left() < 2.0:
                        break
                    x_expand = x_centroid + gamma * (x_reflect - x_centroid)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    f_expand = evaluate(x_expand)
                    local_calls += 1
                    if f_expand < local_best_f:
                        local_best_f = f_expand
                        local_best_x = x_expand.copy()
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        f_vals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        f_vals[-1] = f_reflect
                elif f_reflect < f_vals[-2]:
                    simplex[-1] = x_reflect
                    f_vals[-1] = f_reflect
                else:
                    if local_calls >= max_calls or time_left() < 2.0:
                        break
                    x_contract = x_centroid + rho * (simplex[-1] - x_centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    f_contract = evaluate(x_contract)
                    local_calls += 1
                    if f_contract < local_best_f:
                        local_best_f = f_contract
                        local_best_x = x_contract.copy()
                    if f_contract < f_reflect:
                        simplex[-1] = x_contract
                        f_vals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if local_calls >= max_calls or time_left() < 2.0:
                                break
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_i = evaluate(simplex[i])
                            local_calls += 1
                            if f_i < local_best_f:
                                local_best_f = f_i
                                local_best_x = simplex[i].copy()
                            f_vals[i] = f_i
                        break

                spread = max(np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, len(simplex)))
                if spread < 1e-8:
                    break

            return local_best_x, local_best_f

        # Differential Evolution
        def run_de(x0, max_calls, pop_size=20, F=0.8, CR=0.9):
            if max_calls <= 0:
                return self.best_x.copy(), self.best_f

            local_best_x = self.best_x.copy()
            local_best_f = self.best_f
            local_calls = 0

            # Initialize population
            pop = []
            pop_f = []
            pop.append(x0.copy())
            f0 = evaluate(x0)
            local_calls += 1
            pop_f.append(f0)
            if f0 < local_best_f:
                local_best_f = f0
                local_best_x = x0.copy()

            for _ in range(pop_size - 1):
                if local_calls >= max_calls or time_left() < 2.0:
                    break
                x = np.random.uniform(-4.5, 4.5, dim)
                x = np.clip(x, -5.0, 5.0)
                f = evaluate(x)
                local_calls += 1
                pop.append(x.copy())
                pop_f.append(f)
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x.copy()

            if len(pop) < 3:
                return local_best_x, local_best_f

            # DE loop
            while local_calls < max_calls and time_left() > 2.0:
                for i in range(len(pop)):
                    if local_calls >= max_calls or time_left() < 2.0:
                        break

                    # Select three distinct individuals
                    indices = list(range(len(pop)))
                    random.shuffle(indices)
                    a, b, c = indices[0], indices[1], indices[2]

                    v = pop[a] + F * (pop[b] - pop[c])
                    v = np.clip(v, -5.0, 5.0)

                    u = np.zeros(dim)
                    j_rand = random.randint(0, dim - 1)
                    for j in range(dim):
                        if random.random() < CR or j == j_rand:
                            u[j] = v[j]
                        else:
                            u[j] = pop[i][j]

                    f_u = evaluate(u)
                    local_calls += 1

                    if f_u <= pop_f[i]:
                        pop[i] = u.copy()
                        pop_f[i] = f_u
                        if f_u < local_best_f:
                            local_best_f = f_u
                            local_best_x = u.copy()

            return local_best_x, local_best_f

        # Main strategy
        # Phase 1: Global exploration with DE - use about 40% of budget
        de_budget = min(180, max(50, int(budget * 0.4)))
        if calls_left() > de_budget and time_left() > 10.0:
            run_de(np.zeros(dim), de_budget, pop_size=15)

        # Phase 2: CMA-ES from the best point found - use about 40% of budget
        cma_budget = min(180, max(50, int(budget * 0.4)))
        if calls_left() > cma_budget and time_left() > 5.0:
            run_cma_es(self.best_x.copy(), 0.5, cma_budget)

        # Phase 3: Nelder-Mead fine-tuning - use about 10% of budget
        nm_budget = min(50, max(20, int(budget * 0.1)))
        if calls_left() > nm_budget and time_left() > 3.0:
            run_nelder_mead(self.best_x.copy(), nm_budget)

        # Phase 4: Final CMA-ES polish with small sigma
        if calls_left() > 10 and time_left() > 2.0:
            run_cma_es(self.best_x.copy(), 0.1, calls_left())

        return self.best_x, self.best_f
