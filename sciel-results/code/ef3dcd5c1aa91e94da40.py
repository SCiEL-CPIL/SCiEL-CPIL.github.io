"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with quasi-random init, adaptive restarts, and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        # Population size
        lambda_ = int(4 + 3 * np.log(dim))
        mu = lambda_ // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 / (dim + 1.3) ** 2 + 2 / (dim + mu_eff)) * (1 - c_c)
        c_mu = (2 / (dim + 1.3) ** 2 + 2 / (dim + mu_eff)) * (
            min(1, (mu_eff - 2 + np.prod(w)) / (dim + 2))
        )
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Sobol-like quasi-random points for initial exploration
        def sobol_like_points(n, dim, seed=0):
            """Generate quasi-random points in [0,1]^dim using a simple low-discrepancy sequence."""
            rng = np.random.default_rng(seed)
            points = []
            # Use Halton-like base for quasi-randomness
            for i in range(n):
                pt = np.zeros(dim)
                for d in range(dim):
                    # Use different prime bases per dimension
                    base = 2 + d
                    x = 0.0
                    f = 1.0 / base
                    i_val = i + 1
                    while i_val > 0:
                        x += f * (i_val % base)
                        i_val //= base
                        f /= base
                    pt[d] = x
                points.append(pt)
            # Add small random perturbation to break any exact patterns
            points = np.array(points)
            points += rng.uniform(0, 0.01, points.shape)
            points = np.clip(points, 0, 1)
            return points

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        calls = 0
        rng = np.random.default_rng(42)

        def evaluate(z):
            nonlocal best_x, best_f, calls
            z = clip(z)
            f = func(z)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = z.copy()
            return f

        # Phase 1: Quasi-random global exploration (first ~20% of budget)
        n_explore = max(10, budget // 5)
        if calls < budget:
            # Generate quasi-random points spread across the domain
            sobol_pts = sobol_like_points(n_explore, dim, seed=7)
            # Map to [-5, 5]^dim
            for pt in sobol_pts:
                if calls >= budget:
                    break
                x_try = -5.0 + 10.0 * pt
                evaluate(x_try)

        # Phase 2: CMA-ES from best point found
        if calls < budget and time.time() < deadline - 10.0:
            # Start CMA-ES from the best point
            x = best_x.copy() if best_x is not None else np.zeros(dim)
            sigma = 1.5  # initial step size
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)

            # CMA-ES loop
            stagnation_counter = 0
            best_f_prev = best_f
            max_stagnation = 30  # restart if no improvement for this many generations

            while calls < budget and time.time() < deadline - 15.0:
                # Generate population
                pop = []
                for i in range(lambda_):
                    if calls >= budget:
                        break
                    # Sample from N(x, sigma^2 * C)
                    z = x + sigma * rng.multivariate_normal(np.zeros(dim), C)
                    f = evaluate(z)
                    pop.append((z, f))
                if len(pop) == 0:
                    break

                # Sort by fitness
                pop.sort(key=lambda t: t[1])
                selected = pop[:mu]
                yk = [s[0] for s in selected]

                # Weighted mean
                x_new = np.sum(np.array([w[i] * yk[i] for i in range(mu)]), axis=0)

                # Evolution path for sigma
                y_sigma = np.zeros(dim)
                for i in range(mu):
                    y_sigma += w[i] * (yk[i] - x) / (sigma + 1e-30)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls + 1))) / chi_n < (1.4 + 2 / (dim + 1))
                sigma *= np.exp((c_sigma / chi_n) * (np.linalg.norm(p_sigma) / chi_n - 1) * (1.0 if hsig else 1.0))

                # Evolution path for C
                p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x) / sigma
                x = x_new

                # Covariance update
                delta_h = (1 - hsig) * c_c * (2 - c_c)
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(p_c, p_c) + delta_h * np.eye(dim))
                for i in range(mu):
                    diff = (yk[i] - x) / sigma
                    C += c_mu * w[i] * np.outer(diff, diff)

                # Symmetrize and clip sigma
                C = 0.5 * (C + C.T)
                sigma = float(np.clip(sigma, 1e-12, 100.0))

                # Check for stagnation
                if best_f < best_f_prev - 1e-10:
                    stagnation_counter = 0
                    best_f_prev = best_f
                else:
                    stagnation_counter += 1

                # Restart if stagnant
                if stagnation_counter >= max_stagnation and calls < budget - 20:
                    # Re-seed from best point with smaller sigma
                    x = best_x.copy()
                    sigma = max(0.3, sigma * 0.5)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    C = np.eye(dim)
                    stagnation_counter = 0
                    best_f_prev = best_f
                    # Add some exploration
                    if calls < budget:
                        for _ in range(min(5, budget - calls)):
                            z = x + 0.5 * rng.standard_normal(dim)
                            evaluate(z)

                if calls >= budget or time.time() > deadline - 10.0:
                    break

        # Phase 3: Local refinement (Nelder-Mead style)
        if calls < budget - 5 and time.time() < deadline - 3.0 and best_x is not None:
            # Simple local refinement with adaptive step size
            x = best_x.copy()
            fx = best_f
            step = 0.3
            for _ in range(min(30, budget - calls)):
                if calls >= budget or time.time() > deadline - 1.0:
                    break
                # Try moving in each coordinate direction
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline - 1.0:
                        break
                    # Positive direction
                    x_try = x.copy()
                    x_try[d] += step
                    x_try = clip(x_try)
                    f_try = func(x_try)
                    calls += 1
                    if f_try < fx:
                        x = x_try
                        fx = f_try
                        improved = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    # Negative direction
                    if calls < budget:
                        x_try = x.copy()
                        x_try[d] -= 2 * step
                        x_try = clip(x_try)
                        f_try = func(x_try)
                        calls += 1
                        if f_try < fx:
                            x = x_try
                            fx = f_try
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                if improved:
                    step *= 1.1
                else:
                    step *= 0.7
                if step < 1e-5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
