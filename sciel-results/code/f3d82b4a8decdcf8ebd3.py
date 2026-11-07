"""Combined optimizer: CMA-ES-inspired with momentum and adaptive sigma."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-inspired optimizer with adaptive step sizes, momentum, and ridge-aware search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 295.0

        def elapsed_ok():
            return time.time() < deadline

        # Initial point
        x = rng.uniform(-4, 4, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step size
        sigma = 1.5
        sigma_min = 1e-5
        sigma_max = 5.0

        # Track recent improvements for adaptive sigma
        success_count = 0
        total_count = 0
        window_size = 15

        # Momentum for direction
        velocity = np.zeros(n)
        momentum = 0.6

        calls = 1

        # Phase 1: Use some evaluations to build a rough quadratic model
        # Sample a few points to estimate gradient direction and curvature
        model_points = [x.copy()]
        model_values = [fx]

        # Sample along coordinate axes to estimate local structure
        probe_sigma = 0.5
        for i in range(n):
            if calls >= budget or not elapsed_ok():
                break
            for sign in [1, -1]:
                if calls >= budget or not elapsed_ok():
                    break
                probe = x.copy()
                probe[i] += sign * probe_sigma
                probe = np.clip(probe, -5, 5)
                fp = func(probe)
                calls += 1
                model_points.append(probe.copy())
                model_values.append(fp)
                if fp < best_fx:
                    best_fx = fp
                    best_x = probe.copy()

        # Estimate gradient from the probes
        grad = np.zeros(n)
        for i in range(n):
            # Find the two probes that differ only in dimension i
            pos_idx = None
            neg_idx = None
            for j, pt in enumerate(model_points):
                if j == 0:
                    continue
                diff = np.abs(pt - x)
                if np.sum(diff > 1e-10) == 1:
                    dim_changed = np.argwhere(diff > 1e-10)[0, 0]
                    if dim_changed == i:
                        if pt[i] > x[i]:
                            pos_idx = j
                        else:
                            neg_idx = j
            if pos_idx is not None and neg_idx is not None:
                grad[i] = (model_values[pos_idx] - model_values[neg_idx]) / (2 * probe_sigma)
            elif pos_idx is not None:
                grad[i] = (model_values[pos_idx] - fx) / probe_sigma
            elif neg_idx is not None:
                grad[i] = (fx - model_values[neg_idx]) / probe_sigma

        # Identify ridge direction: if gradient is small but there's curvature,
        # the ridge is along directions with low curvature
        # For now, use gradient direction as the primary search direction
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-8:
            grad_dir = grad / grad_norm
        else:
            grad_dir = np.zeros(n)

        # Phase 2: Main optimization loop with CMA-ES-like adaptation
        for i in range(budget - calls):
            if calls >= budget or not elapsed_ok():
                break

            # Generate candidate with adaptive sigma
            noise = rng.standard_normal(n)

            # Blend random exploration with gradient/momentum direction
            if grad_norm > 1e-8:
                # Move in negative gradient direction with some noise
                noise = 0.6 * noise + 0.4 * (-grad_dir) * np.linalg.norm(noise)

            if np.linalg.norm(velocity) > 1e-8:
                dir_unit = velocity / np.linalg.norm(velocity)
                noise = 0.7 * noise + 0.3 * dir_unit * np.linalg.norm(noise)

            candidate = x + sigma * noise
            candidate = np.clip(candidate, -5, 5)

            fy = func(candidate)
            calls += 1

            if fy < fx:
                # Improvement
                velocity = momentum * velocity + (1 - momentum) * (candidate - x)
                x = candidate
                fx = fy
                sigma = min(sigma * 1.15, sigma_max)
                success_count += 1

                if fy < best_fx:
                    best_fx = fy
                    best_x = candidate.copy()

                # Update gradient estimate periodically
                if i % 20 == 19:
                    # Quick gradient re-estimation
                    new_grad = np.zeros(n)
                    for j in range(n):
                        step = max(0.1, sigma * 0.5)
                        x_plus = x.copy()
                        x_plus[j] += step
                        x_plus = np.clip(x_plus, -5, 5)
                        x_minus = x.copy()
                        x_minus[j] -= step
                        x_minus = np.clip(x_minus, -5, 5)
                        if calls + 1 < budget and elapsed_ok():
                            fp = func(x_plus)
                            fm = func(x_minus)
                            calls += 1
                            new_grad[j] = (fp - fm) / (2 * step)
                    if np.linalg.norm(new_grad) > 1e-8:
                        grad = new_grad
                        grad_norm = np.linalg.norm(grad)
                        grad_dir = grad / grad_norm

            else:
                # No improvement
                velocity = momentum * velocity
                sigma = max(sigma * 0.92, sigma_min)

            total_count += 1

            # Periodically adjust sigma based on success rate
            if total_count % window_size == 0:
                success_rate = success_count / total_count
                if success_rate > 0.6:
                    sigma = min(sigma * 1.1, sigma_max)
                elif success_rate < 0.25:
                    sigma = max(sigma * 0.85, sigma_min)
                success_count = 0
                total_count = 0

            # Occasionally try a larger jump to escape local optima
            if i % 40 == 39 and sigma < 2.5:
                big_noise = rng.standard_normal(n) * 1.5
                big_candidate = np.clip(x + big_noise, -5, 5)
                if calls < budget and elapsed_ok():
                    big_fy = func(big_candidate)
                    calls += 1
                    if big_fy < fx:
                        x = big_candidate
                        fx = big_fy
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        sigma = min(sigma * 1.3, sigma_max)

        return best_x, best_fx
# EVOLVE-BLOCK-END
