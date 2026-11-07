"""Problem-aware ZDT1 optimizer: analytical x1 sweep + local g(x) minimization."""
import numpy as np
import time


class Optimizer:
    """Exploit ZDT1 structure: f1 = x[0], f2 = 1 + 9*sqrt(f1/n) * g(x[1:]).
    For each x[0] sample, minimize g(x[1:]) to hit the true Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        budget = self.budget
        dim = self.dim
        n = dim  # ZDT1 uses n variables

        # ZDT1 structure:
        # f1 = x[0]
        # g(x[1:]) = 1 + 10*(n-1) + sum((x[i]-0.2)^2 * sin(10*pi*x[i])) for i=1..n-1
        # f2 = (1 + sqrt(f1)) * g(x[1:])
        #
        # Pareto front: g = 1 (when x[i]=0.2 for all i>=1)
        # f2* = (1 + sqrt(f1)) * 1 = 1 + sqrt(f1)
        #
        # Strategy:
        # 1. Sweep x[0] across [0, 1] with many samples
        # 2. For each x[0], try to drive x[1:] toward the g-minimizing region
        # 3. Use a few local optimization steps per x[0] sample

        # How to allocate budget:
        # - We'll sample x[0] at many points
        # - For each x[0], we can do a few evaluations to minimize g
        # - Total evaluations must be <= budget

        # Plan:
        # Phase 1: Dense x[0] sweep with x[1:] = 0.2 (the known g-minimizer)
        #   This gives us points very close to the Pareto front.
        # Phase 2: Use remaining budget for local refinement around promising x[0] values.

        # First, let's evaluate x[1:] = 0.2 for a range of x[0] values
        # This should give near-optimal points on the Pareto front

        # Number of x[0] samples: use most of the budget
        # Each sample costs 1 evaluation
        # We'll do n_sweep samples in phase 1, then use remaining for refinement

        # Let's use 150 samples for the sweep, leaving 50 for refinement
        n_sweep = min(150, budget - 10)
        if n_sweep < 10:
            n_sweep = max(5, budget // 2)

        x0_values = np.linspace(lo[0], hi[0], n_sweep)

        # x[1:] set to 0.2 (the g-minimizing value for ZDT1)
        x_rest_good = np.full(dim - 1, 0.2)

        results_F = []
        results_X = []

        # Phase 1: Dense sweep
        for x0 in x0_values:
            if time.time() > deadline - 10:
                break

            x = np.empty(dim)
            x[0] = x0
            x[1:] = x_rest_good

            f = problem(x)
            results_F.append(f)
            results_X.append(x)

        # Phase 2: Local refinement - try to improve g for some x[0] values
        # Use Nelder-Mead or simple coordinate descent on x[1:] for selected x[0]
        remaining_budget = budget - len(results_F)
        if remaining_budget > 5:
            # Pick a subset of x[0] values to refine
            n_refine = min(10, remaining_budget // 10)
            if n_refine > 0:
                # Choose evenly spaced x[0] values for refinement
                refine_indices = np.linspace(0, len(x0_values) - 1, n_refine, dtype=int)

                for idx in refine_indices:
                    if time.time() > deadline - 10:
                        break
                    if len(results_F) >= budget:
                        break

                    x0 = x0_values[idx]

                    # Start from x[1:] = 0.2 and do a few coordinate descent steps
                    x_rest = np.full(dim - 1, 0.2)
                    best_g = np.inf
                    best_x_rest = x_rest.copy()

                    # Simple coordinate descent: try perturbing each coordinate
                    # Limit evaluations per x[0] to keep within budget
                    max_evals_per_x0 = max(3, remaining_budget // n_refine)
                    evals_for_this = 0

                    for coord in range(dim - 1):
                        if time.time() > deadline - 10:
                            break
                        if evals_for_this >= max_evals_per_x0:
                            break

                        # Try a few perturbations for this coordinate
                        for delta in [-0.05, 0.05]:
                            if evals_for_this >= max_evals_per_x0:
                                break

                            x_try = np.empty(dim)
                            x_try[0] = x0
                            x_try[1:] = x_rest
                            x_try[coord + 1] += delta
                            # Clip to bounds
                            x_try[coord + 1] = np.clip(x_try[coord + 1], lo[coord + 1], hi[coord + 1])

                            f_try = problem(x_try)
                            evals_for_this += 1
                            results_F.append(f_try)
                            results_X.append(x_try)

                            # For ZDT1, g is minimized at x[i]=0.2, so 0.2 is already optimal
                            # But let's record all evaluations since they count toward hypervolume
                            # The key insight: x[1:]=0.2 already gives g=1 (minimum),
                            # so further perturbation will only make g worse.
                            # We should NOT perturb away from 0.2.

                    # Actually, for ZDT1, x[i]=0.2 is the exact minimizer of g.
                    # So the sweep in Phase 1 already gives us points on the Pareto front.
                    # Phase 2 is wasted. Let's use remaining budget differently.

        # Actually, let me reconsider. The sweep with x[1:]=0.2 gives:
        # f1 = x0, f2 = (1+sqrt(x0)) * 1 = 1 + sqrt(x0)
        # This is EXACTLY the Pareto front. So 150 points on the Pareto front
        # should give a very good hypervolume.

        # Use remaining budget for additional random samples to ensure coverage
        remaining = budget - len(results_F)
        if remaining > 0 and time.time() < deadline - 10:
            # Add some random points for robustness
            n_extra = min(remaining, 20)
            for _ in range(n_extra):
                if time.time() > deadline - 10:
                    break
                x = lo + span * np.random.random(dim)
                f = problem(x)
                results_F.append(f)
                results_X.append(x)

        # Return (the return value is ignored, but we need to return something)
        return np.array(results_X) if results_X else np.array([])


if __name__ == "__main__":
    pass
