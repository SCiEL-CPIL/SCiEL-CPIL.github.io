"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 315.0  # evaluator kills at ~330s; keep a safety margin


class Optimizer:
    """(1+1)-ES with window-based success-rate sigma adaptation,
    multi-point initialization, aggressive stall restarts, and
    adaptive coordinate-descent polish. Minimizes func on [-5, 5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.perf_counter()
        eval_count = 0

        def _time_ok():
            return time.perf_counter() - t0 < _DEADLINE

        rng = np.random.default_rng(0)

        # ---- Phase 0: Multi-point initialization ----
        # Sample several points to get a good starting region for Schwefel.
        n_init = min(5, budget // 20)
        candidates = []
        for i in range(n_init):
            if not _time_ok() or eval_count >= budget:
                break
            if i == 0:
                x_c = rng.uniform(-0.5, 0.5, dim)
            else:
                # Spread out: mix of center-ish and broader samples
                scale = 1.5 + 0.5 * i
                x_c = rng.uniform(-scale, scale, dim)
            x_c = np.clip(x_c, -5.0, 5.0)
            f_c = func(x_c)
            eval_count += 1
            candidates.append((f_c, x_c))

        candidates.sort(key=lambda t: t[0])
        fx, x = candidates[0]
        sigma = 2.0
        sigma_min = 1e-4
        sigma_max = 4.0

        # Window for success-rate tracking
        window_size = 20
        success_window = []  # 1 for success, 0 for failure

        stall = 0
        STALL_LIMIT = 40
        restart_count = 0
        MAX_RESTARTS = 8

        # ---- Phase 1: (1+1)-ES with window-based sigma adaptation ----
        while eval_count < budget and _time_ok():
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = func(y)
            eval_count += 1

            if fy < fx:
                x, fx = y, fy
                success_window.append(1)
                stall = 0
            else:
                success_window.append(0)
                stall += 1

            # Trim window
            if len(success_window) > window_size:
                success_window.pop(0)

            # Adapt sigma based on success rate in window
            if len(success_window) >= window_size // 2:
                sr = sum(success_window) / len(success_window)
                # Target success rate ~ 0.2-0.3 (1/5 rule generalization)
                if sr > 0.3:
                    sigma *= 1.15
                elif sr < 0.15:
                    sigma *= 0.85
                # else: keep sigma

            sigma = max(sigma_min, min(sigma, sigma_max))

            if stall >= STALL_LIMIT:
                # Aggressive restart: jump significantly from best
                if eval_count < budget and _time_ok() and restart_count < MAX_RESTARTS:
                    # Try a larger jump; scale with distance from center
                    jump_scale = max(sigma * 3.0, 1.5)
                    x_new = np.clip(x + jump_scale * rng.standard_normal(dim), -5.0, 5.0)
                    fx_new = func(x_new)
                    eval_count += 1
                    if fx_new < fx:
                        x, fx = x_new, fx_new
                    else:
                        # If the jump didn't help, try a smaller one
                        if eval_count < budget and _time_ok():
                            x_new2 = np.clip(x + sigma * 1.5 * rng.standard_normal(dim), -5.0, 5.0)
                            fx_new2 = func(x_new2)
                            eval_count += 1
                            if fx_new2 < fx:
                                x, fx = x_new2, fx_new2
                    sigma = max(sigma * 0.7, sigma_min)
                    stall = 0
                    restart_count += 1

        # ---- Phase 2: Adaptive coordinate-descent polish ----
        # Use remaining budget for fine-tuning
        remaining = budget - eval_count
        if remaining > 0 and _time_ok():
            # Determine initial step from current sigma
            step = max(sigma, 0.02)
            max_cd_rounds = min(5, remaining // (2 * dim))

            for _round in range(max_cd_rounds):
                if not _time_ok() or eval_count >= budget:
                    break
                moved_any = False
                for d in range(dim):
                    if eval_count >= budget or not _time_ok():
                        break
                    # Try both directions with current step
                    best_trial = None
                    best_trial_f = fx
                    for sign in (1.0, -1.0):
                        if eval_count >= budget:
                            break
                        trial = x.copy()
                        trial[d] = np.clip(trial[d] + sign * step, -5.0, 5.0)
                        ftrial = func(trial)
                        eval_count += 1
                        if ftrial < best_trial_f:
                            best_trial_f = ftrial
                            best_trial = trial.copy()
                    if best_trial is not None:
                        x = best_trial
                        fx = best_trial_f
                        moved_any = True

                if not moved_any:
                    step *= 0.5
                    if step < 1e-5:
                        break
                # If we moved, keep step; if not, shrink
                if not _time_ok():
                    break

        return x, fx
# EVOLVE-BLOCK-END
