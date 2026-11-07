# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Coarse global exploration ---
        # Spend ~50 calls on global random sampling to find a good starting point.
        # This is robust to hidden rotations/shifts.
        n_global = 50
        for i in range(n_global):
            if check_time() or used_calls >= budget:
                break
            x = rng.uniform(-5, 5, d)
            f = func(x)
            used_calls += 1
            try_update(x, f)

        # --- Phase 2: CMA-ES local refinement from the best global point ---
        def cma_es(x0, f0, max_calls, sigma0=1.0):
            nonlocal used_calls, best_f, best_x
            if max_calls <= 0 or check_time():
                return

            x = x0.copy()
            sigma = sigma0
            C = np.eye(d)
            C_sqrt = np.eye(d)
            pc = np.zeros(d)

            # CMA-ES parameters
            cc = (4 + d / 4) / (d + 4 + 2 * d / 4)
            cs = cc
            c1 = 1 / (d + 4)
            cmu = 1 / (d + 8)
            damps = 1 + 2 * max(0, np.sqrt((d - 1) / (d + 3)) - 1) + c1
            chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))

            # Population size
            lambda_ = max(4, int(4 + 3 * np.log(d)))
            mu = lambda_ // 2

            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()

            no_improve = 0
            max_no_improve = 30

            while used_calls < budget and not check_time():
                # Generate population
                pop = []
                for i in range(lambda_):
                    if used_calls >= budget or check_time():
                        break
                    z = rng.standard_normal(d)
                    y = C_sqrt @ z
                    xi = x + sigma * y
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    used_calls += 1
                    pop.append((xi, fi))

                if len(pop) < lambda_:
                    break

                # Sort
                pop.sort(key=lambda t: t[1])

                # Update best
                if pop[0][1] < best_f:
                    best_f = pop[0][1]
                    best_x = pop[0][0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Compute mean of top mu
                x_new = np.zeros(d)
                for i in range(min(mu, len(pop))):
                    x_new += weights[i] * pop[i][0]

                # Update step size
                y_mean = (x_new - x) / sigma if sigma > 0 else np.zeros(d)
                sigma *= np.exp(cs / damps * (np.linalg.norm(y_mean) / chiN - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

                # Update path
                pc = (1 - cs) * pc + cs * y_mean
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs)**2) / chiN < 1.4 + 2 / (d + 1))

                # Update covariance
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cs * (1 - cs) * C)
                for i in range(min(mu, len(pop))):
                    yi = (pop[i][0] - x) / sigma
                    C += cmu * weights[i] * np.outer(yi, yi)

                x = x_new

                # Recompute C_sqrt
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                except:
                    C_sqrt = np.eye(d)

                # Check for convergence
                if sigma < 1e-8:
                    break
                if no_improve >= max_no_improve:
                    break

        if best_x is not None:
            cma_es(best_x, best_f, budget - used_calls, sigma0=0.5)

        # --- Phase 3: Local polish with (1+1)-ES ---
        if best_x is not None and used_calls < budget and not check_time():
            x = best_x.copy()
            f = best_f
            sigma = 0.01
            for _ in range(100):
                if check_time() or used_calls >= budget:
                    break
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                used_calls += 1
                if fy < f:
                    x = y
                    f = fy
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                sigma = float(np.clip(sigma, 1e-7, 5.0))
                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
