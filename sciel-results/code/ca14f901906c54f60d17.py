"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: L-BFGS-B local + CMA-ES global for BBOB f9."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 330s limit

        # Global best tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Initial exploration - try diverse starting points
        candidates = [np.zeros(dim)]
        for _ in range(15):
            candidates.append(rng.uniform(-4, 4, dim))

        best_start = None
        best_start_f = np.inf
        for c in candidates:
            if call_count >= budget:
                break
            f = safe_func(c)
            if f < best_start_f:
                best_start_f = f
                best_start = c.copy()

        if best_start is None:
            best_start = np.zeros(dim)
            best_start_f = safe_func(best_start)

        # Phase 2: L-BFGS-B style local optimization from best start
        # Use numerical gradient with central differences
        eps = 1e-7

        def numerical_gradient(x):
            grad = np.zeros(dim)
            for i in range(dim):
                if call_count >= budget:
                    break
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                fp = safe_func(x_plus)
                fm = safe_func(x_minus)
                grad[i] = (fp - fm) / (2 * eps)
            return grad

        def l_bfgs_local(start_x, max_iter=20):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return start_x, best_f

            x = start_x.copy()
            fx = best_f
            m = 10  # memory for L-BFGS
            s_list = []
            y_list = []

            for iteration in range(max_iter):
                if call_count >= budget or time.time() > deadline:
                    break

                g = numerical_gradient(x)
                if call_count >= budget:
                    break

                if np.linalg.norm(g) < 1e-10:
                    break

                # L-BFGS two-loop recursion to compute search direction
                if len(s_list) == 0:
                    direction = -g
                else:
                    q = g.copy()
                    rho = []
                    alpha = []
                    for i in range(len(s_list) - 1, -1, -1):
                        s_i = s_list[i]
                        y_i = y_list[i]
                        rho_i = 1.0 / np.dot(s_i, y_i)
                        alpha_i = rho_i * np.dot(s_i, q)
                        q = q - alpha_i * y_i
                        rho.append(rho_i)
                        alpha.append(alpha_i)

                    # Initial Hessian approximation: gamma * I
                    gamma = np.dot(s_list[-1], s_list[-1]) / np.dot(s_list[-1], y_list[-1])
                    r = gamma * q

                    for i in range(len(s_list)):
                        s_i = s_list[i]
                        y_i = y_list[i]
                        beta = rho[i] * np.dot(y_i, r)
                        r = r + (alpha[i] - beta) * s_i
                    direction = r

                # Line search (Armijo)
                alpha = 1.0
                c1 = 1e-4
                for ls_iter in range(20):
                    if call_count >= budget:
                        break
                    x_trial = np.clip(x + alpha * direction, -5.0, 5.0)
                    f_trial = safe_func(x_trial)
                    if f_trial <= fx + c1 * alpha * np.dot(g, direction):
                        break
                    alpha *= 0.5

                x_new = np.clip(x + alpha * direction, -5.0, 5.0)
                f_new = safe_func(x_new) if call_count < budget else f_trial

                # Update L-BFGS memory
                s = x_new - x
                y = numerical_gradient(x_new) - g
                if np.dot(s, y) > 1e-12:
                    s_list.append(s)
                    y_list.append(y)
                    if len(s_list) > m:
                        s_list.pop(0)
                        y_list.pop(0)

                x = x_new
                fx = f_new

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            return x, fx

        # Run L-BFGS from the best starting point
        if call_count < budget:
            _, _ = l_bfgs_local(best_start, max_iter=15)

        # Phase 3: CMA-ES for global exploration around best found
        remaining = budget - call_count
        if remaining > 20:
            pop_size = max(4, min(12, int(4 + 3 * np.log(dim))))
            if remaining < pop_size * 5:
                pop_size = max(2, remaining // 5)

            x = best_x.copy()
            f = best_f
            sigma = 0.5  # smaller initial step since we have a good local solution

            # CMA-ES state
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            cs = (pop_size + 2) / (dim + 4)
            ds = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + cs
            cc = (4 + dim / 4) / (dim + 4 + 2 * pop_size / (dim + 4))
            c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
            cmu = min(1 - c1, 2 * (2 - (dim + 2) / (dim + 4)) / ((dim + 2) ** 2 + pop_size))
            chi_d = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            while call_count < budget:
                if time.time() > deadline:
                    break

                # Generate population
                w = np.zeros(pop_size)
                for i in range(pop_size):
                    w[i] = np.log(pop_size / 2 + 0.5) - np.log(i + 1)
                w_sum = w.sum()
                w = w / w_sum

                # Generate offspring
                ys = np.zeros((pop_size, dim))
                for i in range(pop_size):
                    if call_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = x + sigma * (C @ z)
                    ys[i] = np.clip(y, -5.0, 5.0)

                # Evaluate offspring
                fs = np.full(pop_size, np.inf)
                for i in range(pop_size):
                    if call_count >= budget:
                        break
                    if time.time() > deadline:
                        break
                    fs[i] = safe_func(ys[i])

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = ys[order]
                fs_sorted = fs[order]

                # Update mean
                x_new = np.sum(w[:, np.newaxis] * ys_sorted, axis=0)
                x_new = np.clip(x_new, -5.0, 5.0)

                # Update covariance matrix
                delta_x = x_new - x
                C = (1 - c1 - cmu) * C + c1 * (np.outer(delta_x, delta_x) / max(sigma ** 2, 1e-20) + (1 - (1 - cs) ** 2) * C)
                for i in range(pop_size // 2 + 1):
                    vi = (ys_sorted[i] - x) / max(sigma, 1e-10)
                    C += cmu * w[i] * np.outer(vi, vi)

                # Ensure C is positive semi-definite
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except:
                    C = np.eye(dim)

                # Update sigma
                norm_delta = np.linalg.norm(delta_x)
                if norm_delta > 1e-12:
                    p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs)) * delta_x / max(sigma, 1e-10)
                    sigma *= np.exp(cs / chi_d * (np.linalg.norm(p_sigma) / chi_d - 1))
                    sigma = np.clip(sigma, 1e-8, 10.0)

                x = x_new

                # If CMA-ES finds a significantly better point, run L-BFGS polish
                if fs_sorted[0] < best_f - 0.1 and call_count < budget - 10:
                    x_candidate = ys_sorted[0].copy()
                    _, _ = l_bfgs_local(x_candidate, max_iter=5)

        return best_x, best_f
# EVOLVE-BLOCK-END
