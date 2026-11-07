"""Combined optimizer for BBOB f12 (Bent cigar) with Hooke-Jeeves and surrogate guidance."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)

        # Start at origin (common for BBOB shifted functions)
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Keep a sliding window of recent evaluations for the quadratic model
        history_x = [best_x.copy()]
        history_f = [best_f]
        max_hist = 30

        # Hooke-Jeeves parameters
        sigma = 1.0
        sigma_cap = 10.0
        sigma_floor = 1e-5
        growth = 1.5
        decay = 0.6
        no_improve = 0
        max_no_improve = 25

        # Phase transition: after ~300 calls, switch to polish mode
        phase1_end = 300
        polish_sigma = 0.05
        polish_growth = 1.1
        polish_decay = 0.7
        polish_cap = 1.0

        for i in range(budget - 1):
            if calls >= budget or time.time() > deadline:
                break

            # Update history with current best
            history_x.append(best_x.copy())
            history_f.append(best_f)
            if len(history_x) > max_hist:
                history_x.pop(0)
                history_f.pop(0)

            # Phase transition
            if i >= phase1_end:
                sigma = polish_sigma
                growth = polish_growth
                decay = polish_decay
                sigma_cap = polish_cap

            used_surrogate = False
            n_hist = len(history_x)
            n_params = 1 + dim + dim * (dim + 1) // 2  # 21 for dim=5

            # Try surrogate-guided step if we have enough data
            if n_hist >= n_params and i > 20:
                X = np.array(history_x)
                F = np.array(history_f)

                # Build design matrix for quadratic features
                M = np.ones((n_hist, n_params))
                M[:, 1:dim + 1] = X
                col = dim + 1
                for i_idx in range(dim):
                    for j_idx in range(i_idx, dim):
                        M[:, col] = X[:, i_idx] * X[:, j_idx]
                        col += 1

                try:
                    coeffs, _, rank, _ = np.linalg.lstsq(M, F, rcond=None)
                    if rank >= n_params - 1:
                        # Extract Hessian C from quadratic coefficients
                        C = np.zeros((dim, dim))
                        col = dim + 1
                        for i_idx in range(dim):
                            for j_idx in range(i_idx, dim):
                                val = coeffs[col]
                                C[i_idx, j_idx] = val
                                if i_idx != j_idx:
                                    C[j_idx, i_idx] = val
                                col += 1

                        # Check if Hessian (2*C) is positive definite
                        eigvals = np.linalg.eigvalsh(C)
                        if eigvals[0] > 1e-6:
                            # Newton step: solve C * delta = -c1/2
                            c1 = coeffs[1:dim + 1]
                            try:
                                x_pred = -np.linalg.solve(C, c1) / 2.0
                                x_pred = np.clip(x_pred, -5, 5)

                                dist = np.linalg.norm(x_pred - best_x)
                                if dist < sigma * 5 and dist > 1e-10:
                                    fy = func(x_pred)
                                    calls += 1

                                    if fy < best_f:
                                        best_x = x_pred.copy()
                                        best_f = fy
                                        used_surrogate = True
                                        sigma *= 0.7
                                        no_improve = 0
                                    else:
                                        # Surrogate was wrong, fall through to HJ
                                        pass
                            except np.linalg.LinAlgError:
                                pass
                        else:
                            # Not a local min, use gradient direction
                            grad = c1 + 2 * C @ best_x
                            grad_norm = np.linalg.norm(grad)
                            if grad_norm > 1e-12:
                                y = best_x - 0.5 * grad / grad_norm * sigma
                                y = np.clip(y, -5, 5)
                                fy = func(y)
                                calls += 1
                                if fy < best_f:
                                    best_x = y.copy()
                                    best_f = fy
                                    sigma *= growth
                                    no_improve = 0
                                else:
                                    sigma *= decay
                                    no_improve += 1
                                used_surrogate = True
                            else:
                                # Gradient is zero, use random step
                                y = best_x + sigma * rng.standard_normal(dim)
                                y = np.clip(y, -5, 5)
                                fy = func(y)
                                calls += 1
                                if fy < best_f:
                                    best_x = y.copy()
                                    best_f = fy
                                    sigma *= growth
                                    no_improve = 0
                                else:
                                    sigma *= decay
                                    no_improve += 1
                except Exception:
                    pass

            if not used_surrogate:
                # Hooke-Jeeves pattern search
                improved = False
                x_orig = best_x.copy()

                for i_idx in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break

                    # Test positive direction
                    x_plus = x_orig.copy()
                    x_plus[i_idx] += sigma
                    x_plus = np.clip(x_plus, -5, 5)
                    f_plus = func(x_plus)
                    calls += 1

                    # Test negative direction
                    x_minus = x_orig.copy()
                    x_minus[i_idx] -= sigma
                    x_minus = np.clip(x_minus, -5, 5)
                    f_minus = func(x_minus)
                    calls += 1

                    if f_plus < best_f:
                        best_x = x_plus
                        best_f = f_plus
                        sigma = min(sigma * growth, sigma_cap)
                        improved = True
                        no_improve = 0
                    elif f_minus < best_f:
                        best_x = x_minus
                        best_f = f_minus
                        sigma = min(sigma * growth, sigma_cap)
                        improved = True
                        no_improve = 0

                if not improved:
                    sigma = max(sigma * decay, sigma_floor)
                    no_improve += 1

                    if no_improve >= max_no_improve:
                        # Stagnation: try a larger jump to escape potential local minima
                        sigma = min(sigma * 3.0, sigma_cap)
                        no_improve = 0

                        # Restart: perturb the global best
                        perturbation = sigma * 0.5 * rng.standard_normal(dim)
                        new_x = best_x + perturbation
                        new_x = np.clip(new_x, -5, 5)
                        f_new = func(new_x)
                        calls += 1

                        if f_new < best_f:
                            best_x = new_x.copy()
                            best_f = f_new
                        else:
                            # If restart didn't help, try a larger jump from current
                            new_x = best_x + sigma * rng.standard_normal(dim)
                            new_x = np.clip(new_x, -5, 5)
                            f_new = func(new_x)
                            calls += 1
                            best_x = new_x.copy()
                            best_f = f_new

                        sigma = 1.0

            sigma = np.clip(sigma, sigma_floor, sigma_cap)

        return best_x, best_f
