"""Rewritten optimizer: Multi-start CMA-ES with ridge-aware coordinate descent for Sharp Ridge."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer using Multi-start initialization, CMA-ES refinement,
    and a final ridge-aware coordinate descent polish for high-precision convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = clip(x)
            f = float(func(x))
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---------- Phase 1: Multi-start Initialization ----------
        n_starts = 15
        for _ in range(n_starts):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_point(x)

        if best_x is None:
            return None, np.inf

        # ---------- Phase 2: CMA-ES Refinement ----------
        x_mean = best_x.copy()
        f_mean = best_f

        lambda_ = 5
        sigma = 1.0
        C = np.eye(dim)

        c_sigma = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 2) / (dim + 1) - 1) + c_sigma)
        cs = (dim + lambda_ + 2) / (dim + 5 + 3 * lambda_)
        c1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        cmu = min(1 - c1, 2 * ((lambda_ / 2 - 1) + 1) / ((dim + 2) ** 2 + lambda_))

        ps = np.zeros(dim)
        pc = np.zeros(dim)

        def get_transformation(C):
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            return L

        T = get_transformation(C)

        weights = np.array([np.log(lambda_ / 2 + 0.5) - np.log(i + 1) for i in range(lambda_)])
        weights /= np.sum(weights)
        mu = lambda_ // 2
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff - 2) + 1) / ((dim + 2) ** 2 + mu_eff))

        while evaluated < budget and elapsed_ok():
            pop_x = []
            pop_f = []
            pop_z = []

            for i in range(lambda_):
                if evaluated >= budget or not elapsed_ok():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (T @ z)
                x_i = clip(y)
                f_i = eval_point(x_i)
                pop_x.append(x_i)
                pop_f.append(f_i)
                pop_z.append(z)

            if len(pop_x) < lambda_:
                break

            pop_x = np.array(pop_x)
            pop_f = np.array(pop_f)
            pop_z = np.array(pop_z)

            order = np.argsort(pop_f)
            pop_x = pop_x[order]
            pop_f = pop_f[order]
            pop_z = pop_z[order]

            w = weights[:mu]
            w_sum = np.sum(w)

            y_w = np.sum(w[:, None] * pop_z[:mu], axis=0)
            x_w = np.sum(w[:, None] * pop_x[:mu], axis=0)

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C ** 0.5) @ y_w
            h_sigma = 1.0 if np.sqrt(np.sum(ps ** 2) / (1 - (1 - c_sigma) ** 2)) / (1 - c_sigma) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)) else 0.0

            pc = (1 - cs) * pc + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * (x_w - x_mean)
            x_mean = x_w
            f_mean = pop_f[0]

            delta_mean = x_w - x_mean
            C = (1 + c1 * (h_sigma * (2 - cs) - 1) + cmu) * C
            C -= c1 * (np.outer(pc, pc) - (1 - cs) * (2 - cs) * C)
            C += cmu * np.sum(w[:, None, None] * np.outer(pop_z[:mu] - y_w, pop_z[:mu] - y_w), axis=0)

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt((1 - (1 - c_sigma) ** 2)) - np.sqrt(dim)))
            sigma = np.clip(sigma, 1e-8, 10.0)

            T = get_transformation(C)

        # ---------- Phase 3: Ridge-Aware Coordinate Descent Polish ----------
        if best_x is not None and evaluated < budget and elapsed_ok():
            x = best_x.copy()
            f = best_f
            max_cd_iters = 50

            for cd_iter in range(max_cd_iters):
                if evaluated >= budget or not elapsed_ok():
                    break

                # Estimate principal directions from recent evaluations
                # We don't have recent evaluations stored, so we estimate from the current neighborhood
                # by sampling a few points around x
                if evaluated < budget - 10:
                    # Sample 8 points around x to estimate local geometry
                    sample_points = []
                    sample_vals = []
                    for _ in range(8):
                        if evaluated >= budget or not elapsed_ok():
                            break
                        dx = rng.standard_normal(dim) * 0.1
                        xp = clip(x + dx)
                        fp = eval_point(xp)
                        sample_points.append(xp - x)
                        sample_vals.append(fp)

                    if len(sample_points) >= 3:
                        sample_points = np.array(sample_points)
                        sample_vals = np.array(sample_vals)
                        # Center the samples
                        sample_points_centered = sample_points - np.mean(sample_points, axis=0)
                        # Compute covariance-like matrix
                        cov = np.cov(sample_points_centered, rowvar=False)
                        # Get principal directions
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        # Sort by eigenvalue (descending)
                        idx = np.argsort(eigvals)[::-1]
                        eigvecs = eigvecs[:, idx]
                        # Use top 2 directions
                        dir1 = eigvecs[:, 0]
                        dir2 = eigvecs[:, 1] if dim >= 2 else np.zeros(dim)
                    else:
                        dir1 = np.ones(dim) / np.sqrt(dim)
                        dir2 = np.zeros(dim)
                        if dim >= 2:
                            dir2[0] = 1
                            dir2[1] = -1
                            dir2 /= np.linalg.norm(dir2)
                else:
                    dir1 = np.ones(dim) / np.sqrt(dim)
                    dir2 = np.zeros(dim)
                    if dim >= 2:
                        dir2[0] = 1
                        dir2[1] = -1
                        dir2 /= np.linalg.norm(dir2)

                # Coordinate descent along principal directions
                for direction in [dir1, dir2]:
                    if evaluated >= budget or not elapsed_ok():
                        break
                    if np.linalg.norm(direction) < 1e-10:
                        continue

                    # Adaptive line search
                    step = 0.5
                    for _ in range(20):
                        if evaluated >= budget or not elapsed_ok():
                            break
                        x_try = clip(x + step * direction)
                        f_try = eval_point(x_try)
                        if f_try < f:
                            x = x_try
                            f = f_try
                            best_x = x.copy()
                            best_f = f
                            step *= 1.5
                        else:
                            step *= 0.5
                            if step < 1e-6:
                                break

                # Also do standard coordinate descent as backup
                for i in range(dim):
                    if evaluated >= budget or not elapsed_ok():
                        break
                    step = 0.5
                    for _ in range(10):
                        if evaluated >= budget or not elapsed_ok():
                            break
                        x_try = x.copy()
                        x_try[i] += step
                        x_try = clip(x_try)
                        f_try = eval_point(x_try)
                        if f_try < f:
                            x = x_try
                            f = f_try
                            best_x = x.copy()
                            best_f = f
                            step *= 1.5
                        else:
                            x_try = x.copy()
                            x_try[i] -= step
                            x_try = clip(x_try)
                            f_try = eval_point(x_try)
                            if f_try < f:
                                x = x_try
                                f = f_try
                                best_x = x.copy()
                                best_f = f
                                step *= 1.5
                            else:
                                step *= 0.5
                                if step < 1e-6:
                                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
