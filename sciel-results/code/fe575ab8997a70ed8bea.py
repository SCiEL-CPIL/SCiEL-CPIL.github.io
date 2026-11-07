"""Combined optimizer for BBOB f12 (Bent cigar) with coordinate-wise line search and adaptive step sizes."""
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

        rng = np.random.default_rng(42)

        # Start at origin (common for BBOB)
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Adaptive step sizes per coordinate
        steps = np.ones(dim) * 1.0
        step_min = 1e-6
        step_max = 10.0

        # Line search parameters
        step_growth = 1.5
        step_decay = 0.5
        max_line_search_iters = 50

        # History for potential quadratic refinement (used sparingly)
        history_x = [best_x.copy()]
        history_f = [best_f]
        max_hist = 30

        for i in range(budget - 1):
            if calls >= budget or time.time() > deadline:
                break

            # Coordinate-wise line search (Hooke-Jeeves style)
            improved = False
            for d in range(dim):
                if calls >= budget or time.time() > deadline:
                    break

                direction = np.zeros(dim)
                direction[d] = 1.0

                # Try positive direction
                x_plus = best_x + steps[d] * direction
                x_plus = np.clip(x_plus, -5, 5)
                f_plus = func(x_plus)
                calls += 1

                if f_plus < best_f:
                    best_x = x_plus.copy()
                    best_f = f_plus
                    improved = True
                    steps[d] = min(steps[d] * step_growth, step_max)

                    # Continue in same direction while improving
                    while calls < budget and time.time() < deadline:
                        x_next = best_x + steps[d] * direction
                        x_next = np.clip(x_next, -5, 5)
                        f_next = func(x_next)
                        calls += 1
                        if f_next < best_f:
                            best_x = x_next.copy()
                            best_f = f_next
                        else:
                            break
                else:
                    # Try negative direction
                    x_minus = best_x - steps[d] * direction
                    x_minus = np.clip(x_minus, -5, 5)
                    f_minus = func(x_minus)
                    calls += 1

                    if f_minus < best_f:
                        best_x = x_minus.copy()
                        best_f = f_minus
                        improved = True
                        steps[d] = min(steps[d] * step_growth, step_max)

                        # Continue in negative direction while improving
                        while calls < budget and time.time() < deadline:
                            x_next = best_x - steps[d] * direction
                            x_next = np.clip(x_next, -5, 5)
                            f_next = func(x_next)
                            calls += 1
                            if f_next < best_f:
                                best_x = x_next.copy()
                                best_f = f_next
                            else:
                                break
                    else:
                        # No improvement in this coordinate, shrink step
                        steps[d] = max(steps[d] * step_decay, step_min)

            # Update history
            history_x.append(best_x.copy())
            history_f.append(best_f)
            if len(history_x) > max_hist:
                history_x.pop(0)
                history_f.pop(0)

            # If stagnating (no improvement this full pass), try a random perturbation
            if not improved:
                # Check if we've been stuck for a while
                if i > 50:
                    # Try a random jump to escape local minima
                    jump_size = max(1.0, np.max(steps))
                    x_jump = best_x + jump_size * rng.standard_normal(dim)
                    x_jump = np.clip(x_jump, -5, 5)
                    f_jump = func(x_jump)
                    calls += 1

                    if f_jump < best_f:
                        best_x = x_jump.copy()
                        best_f = f_jump
                        # Reset steps to medium
                        steps[:] = 1.0
                    else:
                        # Shrink all steps
                        steps[:] = np.maximum(steps * 0.7, step_min)

            # Periodic quadratic refinement when we have enough data and are close to convergence
            if i > 100 and i % 50 == 0 and len(history_x) >= 21:
                X = np.array(history_x)
                F = np.array(history_f)

                # Build design matrix for quadratic features
                n_hist = len(X)
                n_params = 1 + dim + dim * (dim + 1) // 2

                M = np.ones((n_hist, n_params))
                M[:, 1:dim + 1] = X
                col = dim + 1
                for i_idx in range(dim):
                    for j_idx in range(i_idx, dim):
                        M[:, col] = X[:, i_idx] * X[:, j_idx]
                        col += 1

                try:
                    coeffs, _, rank, _ = np.linalg.lstsq(M, F, rcond=None)
                    if rank >= n_params - 2:
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

                        c1 = coeffs[1:dim + 1]

                        # Check if Hessian is positive definite
                        eigvals = np.linalg.eigvalsh(C)
                        if eigvals[0] > 1e-8:
                            try:
                                x_pred = -np.linalg.solve(C, c1) / 2.0
                                x_pred = np.clip(x_pred, -5, 5)

                                dist = np.linalg.norm(x_pred - best_x)
                                if dist < 5.0 and dist > 1e-10:
                                    fy = func(x_pred)
                                    calls += 1

                                    if fy < best_f:
                                        best_x = x_pred.copy()
                                        best_f = fy
                                        # Update steps based on distance
                                        steps[:] = np.maximum(np.abs(x_pred - best_x) * 0.5, step_min)
                                    else:
                                        # Fall back, slightly shrink steps
                                        steps[:] = np.maximum(steps * 0.8, step_min)
                            except np.linalg.LinAlgError:
                                pass
                except Exception:
                    pass

        return best_x, best_f
