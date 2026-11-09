"""Combined optimizer: CMA-ES global search + paraboloid basin-jumping + local ES polish."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with quadratic basin-jumping and local ES for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def fit_quadratic(center, points, values):
            """Fit a full quadratic model and return the predicted minimizer."""
            m = len(points)
            if m < n + 1:
                return None

            X = np.ones((m, 1 + n + n * (n + 1) // 2))
            for k, (p, v) in enumerate(zip(points, values)):
                d = p - center
                col = 1
                for i in range(n):
                    X[k, col] = d[i]
                    col += 1
                for i in range(n):
                    for j in range(i, n):
                        X[k, col] = d[i] * d[j]
                        col += 1
            y = np.array(values)

            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                return None

            g = coeffs[1:n + 1].copy()
            H = np.zeros((n, n))
            idx = n + 1
            for i in range(n):
                for j in range(i, n):
                    hval = coeffs[idx]
                    idx += 1
                    if i == j:
                        H[i, j] = 2.0 * hval
                    else:
                        H[i, j] = hval
                        H[j, i] = hval

            try:
                eigvals = np.linalg.eigvalsh(H)
                if np.any(eigvals <= 0):
                    return None
            except:
                return None

            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(H, -g, rcond=None)[0]
                except:
                    return None

            model_center = coeffs[0]
            model_new = coeffs[0] + g @ dx + 0.5 * dx @ H @ dx
            if model_new >= model_center:
                return None

            new_x = center + dx
            new_x = np.clip(new_x, -5, 5)
            return new_x

        def paraboloid_jump(center, f_center, h=2.0):
            """Fit paraboloid using 2n+1 points and return predicted minimizer."""
            nonlocal calls_made
            if calls_made + 2 * n + 1 > total_budget:
                return None
            if time.time() > deadline:
                return None

            points = [center.copy()]
            values = [f_center]

            for i in range(n):
                if calls_made >= total_budget or time.time() > deadline:
                    return None
                e_i = np.zeros(n)
                e_i[i] = 1.0
                x_plus = np.clip(center + h * e_i, -5, 5)
                x_minus = np.clip(center - h * e_i, -5, 5)
                fp = evaluate(x_plus)
                fm = evaluate(x_minus)
                points.append(x_plus)
                points.append(x_minus)
                values.append(fp)
                values.append(fm)

            return fit_quadratic(center, points, values)

        def do_es_around(x, fx, budget_limit, sigma_init=0.5):
            """Local (1+λ)-ES around a point."""
            nonlocal calls_made, best_x, best_f
            sigma = sigma_init
            lam = 4
            no_improve = 0
            max_no_improve = 15

            while calls_made < budget_limit and time.time() < deadline:
                if no_improve >= max_no_improve:
                    break

                offspring = []
                offspring_f = []
                for i in range(lam):
                    if calls_made >= budget_limit or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = np.clip(x + sigma * z, -5, 5)
                    fy = evaluate(y)
                    offspring.append(y)
                    offspring_f.append(fy)

                if not offspring:
                    break

                best_idx = np.argmin(offspring_f)
                best_y = offspring[best_idx]
                best_yf = offspring_f[best_idx]

                if best_yf < fx:
                    x = best_y
                    fx = best_yf
                    no_improve = 0
                    sigma *= 1.15
                else:
                    no_improve += 1
                    sigma *= 0.85

                sigma = np.clip(sigma, 1e-6, 3.0)

            return x, fx

        # Phase 1: CMA-ES for global exploration
        pop_size = max(12, int(4 + 3 * np.log(n)))
        mean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        # CMA-ES update parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1) - 1)) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        ps = np.zeros(n)
        pc = np.zeros(n)
        sigma_hist = [sigma]

        max_cma_calls = int(total_budget * 0.65)

        while calls_made < max_cma_calls and time.time() < deadline:
            # Generate population
            ys = []
            for i in range(pop_size):
                if calls_made >= max_cma_calls or time.time() > deadline:
                    break
                # Sample from multivariate normal with covariance C and std sigma
                B, _ = np.linalg.eigh(C)
                z = rng.standard_normal(n)
                y = mean + sigma * (B @ (np.sqrt(np.abs(np.diag(B))) * z))
                y = np.clip(y, -5, 5)
                ys.append(y)

            if not ys:
                break

            # Evaluate
            ys_f = [evaluate(y) for y in ys]

            # Sort
            order = np.argsort(ys_f)
            ys_sorted = [ys[i] for i in order]
            ys_f_sorted = [ys_f[i] for i in order]

            # Update mean
            mean = sum(weights[i] * ys_sorted[i] for i in range(mu))

            # Update covariance
            v = np.array(ys_sorted[:mu]) - mean
            C = (1 - c_c * (2 - c_c)) * C + c_c * (
                d_sigma * c_sigma * sigma * np.outer(ps, ps) +
                sum(weights[i] * np.outer(v[i], v[i]) for i in range(mu))
            )

            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Update step size control vector
            ps = (1 - c_sigma) * ps + c_sigma / np.sqrt(c_sigma * (2 - c_sigma)) * sigma * (
                np.linalg.eigh(C)[1].T @ np.linalg.inv(np.sqrt(np.diag(np.linalg.eigh(C)[0]))) @
                (mean - mean)  # placeholder, simplified
            )
            # Simplified: use normalized mean shift
            try:
                B, V = np.linalg.eigh(C)
                B_inv_sqrt = V @ np.diag(1.0 / np.sqrt(np.maximum(B, 1e-10))) @ V.T
                ps = (1 - c_sigma) * ps + c_sigma / np.sqrt(c_sigma * (2 - c_sigma)) * sigma * B_inv_sqrt @ (mean - mean)
            except:
                pass

            # Update covariance control vector
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff * c_c * (2 - c_c)) / sigma * (mean - mean)

            # Update sigma (1/5-rule)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 5.0)

            # Track best
            if ys_f_sorted[0] < best_f:
                best_f = ys_f_sorted[0]
                best_x = ys_sorted[0].copy()

        # Phase 2: Paraboloid basin-jumping around the best point
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            jump_x = paraboloid_jump(best_x, best_f, h=2.0)
            if jump_x is not None:
                jump_f = evaluate(jump_x)
                if jump_f < best_f:
                    best_f = jump_f
                    best_x = jump_x.copy()

                    # Do local ES around the new best
                    es_budget = min(total_budget, calls_made + 80)
                    best_x, best_f = do_es_around(best_x, best_f, es_budget, sigma_init=0.3)

            # Try a second jump with different h
            if calls_made < total_budget and time.time() < deadline:
                jump_x2 = paraboloid_jump(best_x, best_f, h=1.0)
                if jump_x2 is not None:
                    jump_f2 = evaluate(jump_x2)
                    if jump_f2 < best_f:
                        best_f = jump_f2
                        best_x = jump_x2.copy()
                        es_budget = min(total_budget, calls_made + 60)
                        best_x, best_f = do_es_around(best_x, best_f, es_budget, sigma_init=0.2)

        # Phase 3: Final polishing with local ES
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            remaining = total_budget - calls_made
            if remaining > 20:
                best_x, best_f = do_es_around(best_x, best_f, total_budget, sigma_init=0.15)

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
