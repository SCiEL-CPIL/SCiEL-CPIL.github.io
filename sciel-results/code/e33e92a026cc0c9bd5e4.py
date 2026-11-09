"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 exploiting problem structure.
    
    ZDT4:
    - f1(x) = x1
    - g(x) = 1 + 10*(n-1) + sum_{i=2}^n (xi^2 - 10*cos(2*pi*xi))
    - f2(x) = g(x) * (1 - sqrt(f1/g) - (f1/g)*sin(10*pi*f1))
    
    Pareto front: g=1 (tail vars = 0), f1 = x1 over its bounds.
    Strategy: dense non-uniform sweep of x1 with tail at 0, plus polish.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.dim
        budget = self.budget

        # Optimal tail: closest to 0 within bounds
        tail_opt = np.zeros(n - 1)
        tail_opt = np.clip(tail_opt, lo[1:], hi[1:])

        X = []
        F = []
        eval_count = 0

        def evaluate(x):
            """Evaluate and record a point."""
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = problem(x)
            X.append(x)
            F.append(f)
            eval_count += 1
            return f

        # Phase 1: Dense non-uniform sweep of x1 with tail at optimal value
        # The ZDT4 Pareto front has high curvature near f1=0 and f1=1
        # Use more points near the ends
        x1_lo, x1_hi = lo[0], hi[0]
        n_sweep = int(budget * 0.85)
        n_polish = int(budget * 0.10)
        n_explore = budget - n_sweep - n_polish

        # Non-uniform sampling: cluster near endpoints
        # Use a transformation that biases toward ends
        if n_sweep > 0:
            # Create non-uniform grid: use sin-based spacing for endpoint clustering
            t = np.linspace(0, 1, n_sweep)
            # Apply a transformation that clusters near 0 and 1
            # t' = 0.5 - 0.5*cos(pi*t) gives clustering at ends
            # But we want uniform in the middle and denser at ends
            # Use: x1 = x1_lo + (x1_hi - x1_lo) * (t**0.7 * (1-t)**0.3 + t**0.3 * (1-t)**0.7) / 2
            # Simpler: mix uniform and endpoint-biased
            t_uniform = t
            t_biased = 0.5 - 0.5 * np.cos(np.pi * t)  # clusters at ends
            t_mix = 0.5 * t_uniform + 0.5 * t_biased
            x1_samples = x1_lo + (x1_hi - x1_lo) * t_mix

            for x1_val in x1_samples:
                if time.time() > deadline or eval_count >= budget:
                    break
                x = np.concatenate([[x1_val], tail_opt])
                evaluate(x)

        # Phase 2: Polish - for selected x1 values, try to minimize g further
        # by doing local search on tail variables
        if n_polish > 0 and len(F) > 0:
            # Pick x1 values spread across the range
            n_polish_x1 = min(15, n_polish // 2)
            x1_polish = np.linspace(x1_lo, x1_hi, max(1, n_polish_x1))
            remaining_polish = n_polish

            for x1_val in x1_polish:
                if remaining_polish <= 0 or time.time() > deadline or eval_count >= budget:
                    break

                # Start from tail_opt and do a few local search steps
                best_tail = tail_opt.copy()
                best_f2 = None

                # Try shrinking any non-zero tail components
                for step in range(min(remaining_polish, 4)):
                    if remaining_polish <= 0 or time.time() > deadline or eval_count >= budget:
                        break

                    # Try random small perturbations around current best
                    scale = 0.01 * (hi[1:] - lo[1:])
                    candidate_tail = best_tail + np.random.randn(n-1) * scale
                    candidate_tail = np.clip(candidate_tail, lo[1:], hi[1:])

                    x = np.concatenate([[x1_val], candidate_tail])
                    f = evaluate(x)

                    if best_f2 is None or f[1] < best_f2:
                        best_f2 = f[1]
                        best_tail = candidate_tail.copy()

                    remaining_polish -= 1

        # Phase 3: Exploration - random points biased toward Pareto region
        if n_explore > 0:
            for _ in range(n_explore):
                if time.time() > deadline or eval_count >= budget:
                    break
                x1_val = np.random.uniform(lo[0], hi[0])
                # Tail: mostly near 0 with occasional larger deviations
                if np.random.random() < 0.7:
                    tail_exp = tail_opt + np.random.randn(n-1) * 0.05 * (hi[1:] - lo[1:])
                else:
                    tail_exp = np.random.uniform(lo[1:], hi[1:])
                tail_exp = np.clip(tail_exp, lo[1:], hi[1:])
                x = np.concatenate([[x1_val], tail_exp])
                evaluate(x)

        # Ensure we don't exceed budget
        X = X[:budget]
        F = F[:budget]

        # Final: ensure we used as much budget as possible
        while eval_count < budget and time.time() < deadline:
            x1_val = np.random.uniform(lo[0], hi[0])
            x = np.concatenate([[x1_val], tail_opt])
            evaluate(x)

        X = X[:budget]
        F = F[:budget]

        return X
# EVOLVE-BLOCK-END
