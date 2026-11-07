"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-restart CMA-ES for black-box optimization on BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def check_time():
            return time.time() > deadline

        # CMA-ES parameters
        # Population size: standard is 4 + floor(3*ln(n)) = 4 + 3*1.609 = 8
        pop_size = 4 + int(3 * np.log(dim))  # 8 for dim=5
        # Minimum calls per restart to be useful
        min_calls_per_restart = pop_size + dim  # need at least one generation + simplex

        # Allocate budget across restarts
        # First restart gets more budget for exploration
        # We'll do adaptive: keep restarting while we have budget
        # Each restart: initial mean = best_x (warm start) or random (first)
        # Sigma: start large for first, smaller for subsequent

        # Phase 0: Quick random sampling to get an initial estimate
        # Use 10 random points to find a decent starting region
        n_random = min(10, budget)
        for _ in range(n_random):
            if calls_made >= budget or check_time():
                break
            x = np.random.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            update_best(x, f)

        if best_x is None:
            return best_x, best_f

        # Multi-restart CMA-ES
        # Budget allocation: first restart gets 40% of remaining, subsequent get 25% each
        # Adjust dynamically
        remaining = budget - calls_made

        # CMA-ES implementation
        def cma_es(mean_init, sigma_init, max_calls):
            nonlocal best_x, best_f, calls_made

            n = dim
            # Population size
            lambda_ = pop_size
            # Learning rates
            mu = lambda_ // 2  # number of weighted individuals
            # Weights: proportional to rank
            w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mueff = 1.0 / np.sum(w ** 2)

            c_sigma = (mueff + 2) / ((n + mueff) * (mueff + 2) / 2 + 1)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            # State
            x_mean = mean_init.copy()
            sigma = sigma_init
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            one_over_sqrt_chi_n = 1.0 / np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            local_calls = 0
            best_local_x = x_mean.copy()
            best_local_f = float('inf')

            # Evaluate initial mean
            if not check_time() and calls_made < budget:
                f_mean = func(x_mean)
                calls_made += 1
                local_calls += 1
                update_best(x_mean, f_mean)
                best_local_f = f_mean
                best_local_x = x_mean.copy()

            while local_calls < max_calls and calls_made < budget and not check_time():
                # Generate population
                # Cholesky of C
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback: identity
                    B = np.eye(n)
                    C = np.eye(n)

                # Sample lambda_ points
                Z = np.random.standard_normal((lambda_, n))
                Y = Z @ B.T  # n x lambda_? No: Z is (lambda_, n), B is (n, n)
                # Y[i] = Z[i] @ B.T gives us the transformed points
                # Actually: y = B @ z for each z
                # Z is (lambda_, n), we want Y[i] = B @ Z[i]
                # Y = Z @ B.T is wrong. We need Y[i,j] = sum_k B[k,j] * Z[i,k]
                # That's Y = Z @ B.T only if B is symmetric. C is symmetric, B from cholesky is lower triangular.
                # Y[i] = B @ Z[i].T -> Y = (Z @ B.T).T? No.
                # Let's do: Y = np.array([B @ z for z in Z]) -- slow but correct
                # Or: Y = Z @ B.T works if we think of it as: (Z @ B.T)[i,j] = sum_k Z[i,k]*B.T[k,j] = sum_k Z[i,k]*B[j,k]
                # We want Y[i,j] = sum_k B[k,j] * Z[i,k] = sum_k Z[i,k] * B[k,j]
                # So Y[i,j] = (Z @ B.T)[i,j] where B.T[k,j] = B[j,k]. That's not right.
                # Let me just use: Y[i] = B @ Z[i].T, i.e., Y = (Z @ B.T).T is wrong.
                # Correct: Y[i] = B @ Z[i] (both column vectors)
                # In matrix form: Y = (Z @ B.T).T? 
                # Z @ B.T: (lambda_, n) @ (n, n) = (lambda_, n). [i,j] = sum_k Z[i,k]*B.T[k,j] = sum_k Z[i,k]*B[j,k]
                # We want: Y[i,j] = sum_k B[k,j]*Z[i,k] = sum_k Z[i,k]*B[k,j]
                # So Y[i,j] = sum_k Z[i,k]*B[k,j] which is (Z @ B)[i,j]. 
                # Wait: (Z @ B)[i,j] = sum_k Z[i,k]*B[k,j]. Yes! So Y = Z @ B.
                Y = Z @ B  # (lambda_, n)

                # Generate candidates
                x_pop = x_mean + sigma * Y  # (lambda_, n)
                # Clip to box
                x_pop = np.clip(x_pop, -5, 5)

                # Evaluate population (one by one to track budget)
                f_pop = np.full(lambda_, np.inf)
                for i in range(lambda_):
                    if calls_made >= budget or check_time():
                        break
                    f_pop[i] = func(x_pop[i])
                    calls_made += 1
                    local_calls += 1
                    if f_pop[i] < best_local_f:
                        best_local_f = f_pop[i]
                        best_local_x = x_pop[i].copy()
                    update_best(x_pop[i], f_pop[i])

                # Sort by fitness
                order = np.argsort(f_pop)
                x_sorted = x_pop[order]
                f_sorted = f_pop[order]

                # Check if we have enough valid evaluations
                valid_count = np.sum(np.isfinite(f_sorted))
                if valid_count < mu:
                    break

                # Weighted mean
                x_new = np.sum(w[:mu, np.newaxis] * x_sorted[:mu], axis=0)

                # Update step size control variable ps
                # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff / sigma) * B @ (x_new - x_mean) / sigma
                # Actually: ps = (1-c_sigma)*ps + sqrt(c_sigma*(2-c_sigma)*mueff) / sigma * B @ (x_new - x_mean)
                # Standard: ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) * (B @ (x_new - x_mean)) / sigma
                # Let me use the standard formula:
                # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * B @ (x_new - x_mean)
                diff = x_new - x_mean
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B @ diff)

                # Update sigma
                sigma = sigma * np.exp(c_sigma / c_sigma * (np.linalg.norm(ps) / one_over_sqrt_chi_n - 1))
                # Wait, standard is: sigma *= exp(c_sigma * (||ps|| / chi_n - 1))
                # Let me redo:
                sigma = sigma * np.exp(c_sigma * (np.linalg.norm(ps) / (one_over_sqrt_chi_n * np.sqrt(n)) - 1))
                # Hmm, let me use the simpler standard form:
                # chi_n = sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
                # sigma *= exp(c_sigma * (||ps|| / chi_n - 1))
                chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
                sigma = sigma * np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50.0)

                # Update evolution path pc
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * diff

                # Update covariance matrix
                # delta_hs = (1 + h_sig) * (1 - c_c) * (2 - c_c)
                # h_sig = 1 if (||ps||/sqrt(1-(1-c_sigma)^{2*(gen+1)})) / chi_n < (1.4 + 2/(n+1)) * chi_n
                # Simplified: use h_sig = 1 if ||ps||/chi_n < 1.4 + 2/(n+1)
                h_sig = 1.0 if (np.linalg.norm(ps) / chi_n < 1.4 + 2.0 / (n + 1)) else 0.0
                delta_hs = (1 + h_sig) * (1 - c_c) * (2 - c_c)

                # Outer product of pc
                pc_outer = np.outer(pc, pc)
                # Sum of weighted outer products
                sum_outer = np.zeros((n, n))
                for i in range(mu):
                    diff_i = (x_sorted[i] - x_mean) / sigma
                    sum_outer += w[i] * np.outer(diff_i, diff_i)

                C = (1 + c_1 * delta_hs - c_1 - c_mu) * C + c_1 * (pc_outer + delta_hs * np.eye(n)) + c_mu * sum_outer

                # Symmetrize
                C = (C + C.T) / 2

                # Update mean
                x_mean = x_new

                # Periodic re-orthogonalization of C to prevent numerical issues
                # Every 10 generations or so
                if local_calls % (10 * lambda_) < lambda_:
                    # Ensure positive definiteness
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C = (eigvecs * eigvals) @ eigvecs.T
                    C = (C + C.T) / 2

            return best_local_x, best_local_f

        # Multi-restart schedule
        # First restart: larger sigma, more budget
        # Subsequent: smaller sigma, warm-started from best

        # Determine number of restarts based on remaining budget
        # Each restart needs at least pop_size calls for one generation
        # Let's do: first restart gets 50% of remaining, then split rest
        remaining = budget - calls_made
        if remaining < min_calls_per_restart:
            return best_x, best_f

        # First restart
        sigma_1 = 3.0  # Large to explore
        max_calls_1 = int(remaining * 0.45)
        max_calls_1 = max(max_calls_1, min_calls_per_restart)

        if not check_time() and calls_made < budget:
            bx, bf = cma_es(best_x.copy(), sigma_1, max_calls_1)

        remaining = budget - calls_made
        if remaining < min_calls_per_restart:
            return best_x, best_f

        # Subsequent restarts: smaller sigma, warm-started
        n_restarts = 0
        while remaining >= min_calls_per_restart and not check_time() and calls_made < budget:
            n_restarts += 1
            # Decreasing sigma for each restart
            sigma_r = 3.0 * (0.6 ** n_restarts)
            sigma_r = max(sigma_r, 0.1)

            # Budget per restart: decreasing
            max_calls_r = int(remaining * 0.35)
            max_calls_r = max(max_calls_r, min_calls_per_restart)

            # Warm-start from best, but add small random perturbation for diversity
            perturbation = sigma_r * 0.5
            mean_init = best_x + np.random.standard_normal(dim) * perturbation
            mean_init = np.clip(mean_init, -5, 5)

            bx, bf = cma_es(mean_init, sigma_r, max_calls_r)

            remaining = budget - calls_made

        # Final refinement: if we have budget left, do a local NM-like refinement
        remaining = budget - calls_made
        if remaining >= 10 and not check_time() and best_x is not None:
            # Simple coordinate descent / local search
            step = 0.1
            for _ in range(remaining):
                if calls_made >= budget or check_time():
                    break
                # Try small perturbations in each direction
                improved = False
                for i in range(dim):
                    for direction in [1, -1]:
                        x_trial = best_x.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        if calls_made >= budget or check_time():
                            break
                        f_trial = func(x_trial)
                        calls_made += 1
                        if f_trial < best_f:
                            best_x = x_trial
                            best_f = f_trial
                            improved = True
                    if calls_made >= budget or check_time():
                        break
                if improved:
                    step = max(step * 0.9, 0.001)
                else:
                    step = max(step * 0.5, 0.001)
                    if step <= 0.001:
                        break

        return best_x, best_f

# EVOLVE-BLOCK-END
