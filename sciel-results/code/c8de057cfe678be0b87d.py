"""Combined optimizer for BBOB f12 (Bent cigar) with surrogate-guided (1+1)-ES."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0  # safety margin before 330s limit

        rng = np.random.default_rng(0)

        # Start at origin (common for BBOB)
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Keep a sliding window of recent evaluations for the quadratic model
        history_x = [best_x.copy()]
        history_f = [best_f]
        max_hist = 30

        # Adaptive step size for (1+1)-ES
        sigma = 2.0
        sigma_cap = 15.0
        growth = 1.25
        decay = 0.85

        # Phase transition: after ~250 calls, switch to polish mode
        phase1_end = 250

        for i in range(budget - 1):
            if calls >= budget or time.time() > deadline:
                break

            # Update history
            history_x.append(best_x.copy())
            history_f.append(best_f)
            if len(history_x) > max_hist:
                history_x.pop(0)
                history_f.pop(0)

            # Phase transition
            if i >= phase1_end:
                sigma = 0.05
                growth = 1.1
                decay = 0.7
                sigma_cap = 1.0

            # Try to use surrogate model if we have enough data
            used_surrogate = False
            n_hist = len(history_x)
            n_params = 1 + dim + dim * (dim + 1) // 2  # 21 for dim=5

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
                        # Quadratic part: sum_{i<=j} coeffs[d+1+idx(i,j)] * x_i * x_j
                        # Hessian H = 2*C where C[i,j] = coeffs[...] for i!=j, C[i,i] = coeffs[...]
                        C = np.zeros((dim, dim))
                        col = dim + 1
                        for i_idx in range(dim):
                            for j_idx in range(i_idx, dim):
                                val = coeffs[col]
                                C[i_idx, j_idx] = val
                                if i_idx != j_idx:
                                    C[j_idx, i_idx] = val
                                col += 1

                        # Gradient at best_x: grad = c1 + 2*C*best_x
                        c1 = coeffs[1:dim + 1]
                        grad = c1 + 2 * C @ best_x

                        # Check if Hessian (2*C) is positive definite
                        eigvals = np.linalg.eigvalsh(C)
                        if eigvals[0] > 1e-6:
                            # Newton step: solve C * delta = -c1/2
                            # Actually: grad = 0 => c1 + 2*C*x = 0 => C*x = -c1/2
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
                                        # Reduce sigma on successful surrogate step
                                        sigma *= 0.7
                                    else:
                                        # Surrogate was wrong, fall back to ES
                                        pass
                            except np.linalg.LinAlgError:
                                pass
                        else:
                            # Not a local min, use gradient direction
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
                                else:
                                    sigma *= decay
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
                                else:
                                    sigma *= decay
                except Exception:
                    pass

            if not used_surrogate:
                # Standard (1+1)-ES step
                y = best_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1

                if fy < best_f:
                    best_x = y.copy()
                    best_f = fy
                    sigma *= growth
                else:
                    sigma *= decay

            sigma = np.clip(sigma, 1e-6, sigma_cap)

        return best_x, best_f
