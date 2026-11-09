import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = float(fx)
                best_x = x.copy()
            return float(fx)

        def time_ok():
            return time.time() < deadline

        # ---- Phase 1: Brief diverse sampling for initial point ----
        samples = [
            np.zeros(d),
            rng.uniform(-3.0, 3.0, d),
            rng.uniform(-2.0, 2.0, d),
            rng.normal(0, 1.5, d),
            rng.uniform(-4.0, 4.0, d),
            np.ones(d) * 1.0,
            np.ones(d) * -1.0,
        ]
        for x in samples:
            if eval_count >= 8 or not time_ok():
                break
            evaluate(x)

        # ---- Phase 2: SPSA optimization with time-decaying steps ----
        def spsa(x0, max_iter=None, a0=1.0, c0=0.5, gamma_a=0.6, gamma_c=0.6):
            nonlocal eval_count, best_x, best_f
            if max_iter is None:
                max_iter = (budget - eval_count) // 2

            x = x0.copy()
            f_curr = evaluate(x)

            # SPSA parameters
            A = 10  # effective starting iteration for decay
            for k in range(max_iter):
                if not time_ok() or eval_count >= budget - 5:
                    break

                # Time-decaying step size and perturbation magnitude
                kk = k + A
                a_k = a0 / (kk ** gamma_a)
                c_k = c0 / (kk ** gamma_c)

                # Generate random direction ±1
                delta = rng.choice([-1.0, 1.0], size=d)

                # Evaluate at x + c*delta and x - c*delta
                x_plus = np.clip(x + c_k * delta, -5.0, 5.0)
                x_minus = np.clip(x - c_k * delta, -5.0, 5.0)

                f_plus = evaluate(x_plus)
                if eval_count >= budget - 5 or not time_ok():
                    break
                f_minus = evaluate(x_minus)

                # Gradient estimate
                grad_est = (f_plus - f_minus) / (2.0 * c_k) * delta

                # Update
                x_new = np.clip(x - a_k * grad_est, -5.0, 5.0)

                # Check if improvement
                f_new = evaluate(x_new)
                if f_new < f_curr:
                    x = x_new.copy()
                    f_curr = f_new
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x.copy()
                else:
                    # Revert if no improvement, keep x
                    pass

            return x

        # Run SPSA from current best
        if eval_count < budget - 20 and time_ok():
            x_spsa = spsa(best_x.copy(), max_iter=(budget - eval_count) // 3, a0=1.5, c0=0.8, gamma_a=0.55, gamma_c=0.5)
            if x_spsa is not None and np.all(np.isfinite(x_spsa)):
                best_x = np.clip(x_spsa, -5.0, 5.0)
                if eval_count < budget - 10:
                    evaluate(best_x)

        # ---- Phase 3: Second SPSA pass with smaller initial steps ----
        if eval_count < budget - 15 and time_ok():
            x_spsa2 = spsa(best_x.copy(), max_iter=(budget - eval_count) // 3, a0=0.5, c0=0.3, gamma_a=0.6, gamma_c=0.55)
            if x_spsa2 is not None and np.all(np.isfinite(x_spsa2)):
                best_x = np.clip(x_spsa2, -5.0, 5.0)
                if eval_count < budget - 8:
                    evaluate(best_x)

        # ---- Phase 4: Coordinate-wise fine-tuning ----
        if eval_count < budget - 10 and time_ok():
            for i in range(d):
                if eval_count >= budget - 5 or not time_ok():
                    break
                for step_size in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
                    if eval_count >= budget - 5 or not time_ok():
                        break
                    for direction in [1.0, -1.0]:
                        if eval_count >= budget - 5 or not time_ok():
                            break
                        test_x = best_x.copy()
                        test_x[i] = np.clip(test_x[i] + direction * step_size, -5.0, 5.0)
                        fx = evaluate(test_x)
                        if fx < best_f:
                            best_f = fx
                            best_x = test_x.copy()

        # ---- Phase 5: Final SPSA polish ----
        if eval_count < budget - 5 and time_ok():
            x_final = spsa(best_x.copy(), max_iter=(budget - eval_count) // 2, a0=0.1, c0=0.1, gamma_a=0.7, gamma_c=0.6)
            if x_final is not None and np.all(np.isfinite(x_final)):
                best_x = np.clip(x_final, -5.0, 5.0)
                if eval_count < budget - 2:
                    evaluate(best_x)

        return best_x, best_f
