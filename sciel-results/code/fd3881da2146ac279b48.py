"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Nelder-Mead for basin finding + Hessian Newton polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 275  # safety margin before 300s limit
        rng = np.random.default_rng(12345)

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # --- Phase 1: Nelder-Mead to find a good basin ---
        step = 2.0
        simplex = np.zeros((dim + 1, dim))
        for i in range(dim):
            simplex[i + 1, i] = step
        simplex[0] = 0.0

        fvals = np.array([func(simplex[i]) for i in range(dim + 1)])
        calls_made = dim + 1

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5

        best_idx = np.argmin(fvals)
        x_best = simplex[best_idx].copy()
        f_best = fvals[best_idx]

        nm_iterations = 0
        max_nm_iter = 200

        while calls_made < budget - 50 and nm_iterations < max_nm_iter and time.time() < deadline:
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            if np.max(np.abs(fvals - fvals[0])) < 1e-12:
                break

            x_worst = simplex[-1]
            f_worst = fvals[-1]
            x_centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = x_centroid + alpha * (x_centroid - x_worst)
            x_ref = clip(x_ref)
            f_ref = func(x_ref)
            calls_made += 1

            if f_ref < fvals[0]:
                x_exp = x_centroid + gamma * (x_ref - x_worst)
                x_exp = clip(x_exp)
                f_exp = func(x_exp)
                calls_made += 1
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                if f_ref < f_worst:
                    x_con = x_centroid + rho * (x_ref - x_worst)
                else:
                    x_con = x_centroid - rho * (x_worst - x_centroid)
                x_con = clip(x_con)
                f_con = func(x_con)
                calls_made += 1
                if f_con <= f_ref:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls_made += 1

            nm_iterations += 1

            new_best_idx = np.argmin(fvals)
            if fvals[new_best_idx] < f_best:
                f_best = fvals[new_best_idx]
                x_best = simplex[new_best_idx].copy()

        # --- Phase 2: Hessian Newton polishing ---
        calls_left = budget - calls_made
        if calls_left > 10 and time.time() < deadline:
            x_curr = x_best.copy()
            f_curr = f_best

            # Estimate gradient and Hessian using central finite differences
            h = 0.05

            for newton_round in range(10):
                if calls_left <= 2 * dim + dim * (dim + 1) // 2 or time.time() >= deadline:
                    break

                # Estimate gradient
                grad = np.zeros(dim)
                for i in range(dim):
                    xp = x_curr.copy()
                    xp[i] += h
                    xm = x_curr.copy()
                    xm[i] -= h
                    xp = clip(xp)
                    xm = clip(xm)
                    fpx = func(xp)
                    fxm = func(xm)
                    calls_made += 2
                    calls_left -= 2
                    grad[i] = (fpx - fxm) / (2 * h)

                if calls_left <= dim * (dim + 1) // 2 or time.time() >= deadline:
                    break

                if np.linalg.norm(grad) < 1e-12:
                    break

                # Estimate Hessian
                H = np.zeros((dim, dim))
                for i in range(dim):
                    for j in range(i, dim):
                        if i == j:
                            xp = x_curr.copy()
                            xp[i] += h
                            xm = x_curr.copy()
                            xm[i] -= h
                            xp = clip(xp)
                            xm = clip(xm)
                            fpp = func(xp)
                            fxm = func(xm)
                            calls_made += 2
                            calls_left -= 2
                            H[i, i] = (fpp - 2 * f_curr + fxm) / (h * h)
                        else:
                            xpp = x_curr.copy()
                            xpp[i] += h
                            xpp[j] += h
                            xpm = x_curr.copy()
                            xpm[i] += h
                            xpm[j] -= h
                            xmp = x_curr.copy()
                            xmp[i] -= h
                            xmp[j] += h
                            xmm = x_curr.copy()
                            xmm[i] -= h
                            xmm[j] -= h
                            xpp = clip(xpp)
                            xpm = clip(xpm)
                            xmp = clip(xmp)
                            xmm = clip(xmm)
                            fpp = func(xpp)
                            fpm = func(xpm)
                            fmp = func(xmp)
                            fxm = func(xmm)
                            calls_made += 4
                            calls_left -= 4
                            H[i, j] = (fpp - fpm - fmp + fxm) / (4 * h * h)
                            H[j, i] = H[i, j]

                if calls_left <= 0 or time.time() >= deadline:
                    break

                # Regularize Hessian
                H_reg = H + 1e-6 * np.eye(dim)

                # Solve Newton step
                try:
                    step_newton = np.linalg.solve(H_reg, grad)
                except np.linalg.LinAlgError:
                    step_newton = -grad * 0.1

                # Line search
                alpha_ls = 1.0
                improved = False
                for ls_iter in range(8):
                    if calls_left <= 0 or time.time() >= deadline:
                        break

                    x_trial = clip(x_curr - alpha_ls * step_newton)
                    f_trial = func(x_trial)
                    calls_made += 1
                    calls_left -= 1

                    if f_trial < f_curr:
                        x_curr = x_trial
                        f_curr = f_trial
                        improved = True
                        if f_curr < f_best:
                            f_best = f_curr
                            x_best = x_curr.copy()
                        break
                    else:
                        alpha_ls *= 0.5

                if not improved:
                    # Try smaller step
                    x_trial = clip(x_curr - 0.1 * grad)
                    f_trial = func(x_trial)
                    calls_made += 1
                    calls_left -= 1
                    if f_trial < f_curr and calls_left > 0:
                        x_curr = x_trial
                        f_curr = f_trial
                        if f_curr < f_best:
                            f_best = f_curr
                            x_best = x_curr.copy()
                    break

        return x_best, f_best
# EVOLVE-BLOCK-END
