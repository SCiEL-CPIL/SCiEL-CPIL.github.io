"""Multi-objective optimizer on ZDT1 using structure-aware search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware evolutionary multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # ZDT1 structure:
        # f1 = x[0]
        # g = 1 + 9/(n-1) * sum(x[1:])
        # f2 = g * (1 - sqrt(f1/g))
        # Pareto front: x[1:] = 0, f1 in [0,1], f2 = 1 - sqrt(f1)
        # So we should focus on x[0] for f1, and keep x[1:] near 0.

        n = self.dim
        budget = self.budget

        # Strategy:
        # 1. Sample x[0] densely across [0,1] (this determines f1)
        # 2. For each x[0], try to minimize x[1:] to get g close to 1
        # 3. Use remaining budget for local refinement

        # Phase 1: Dense sampling of x[0] with x[1:] = 0 (Pareto-optimal for ZDT1)
        # This gives us points on the true Pareto front
        n_phase1 = min(100, budget // 2)
        x0_vals = np.linspace(0.0, 1.0, n_phase1)

        archive_F = []
        archive_X = []
        used = 0

        for x0 in x0_vals:
            if time.time() > deadline:
                break
            if used >= budget:
                break

            x = np.zeros(n)
            x[0] = x0
            # x[1:] = 0 is optimal for ZDT1
            f = problem(x)
            archive_F.append(f)
            archive_X.append(x.copy())
            used += 1

        # Phase 2: Use remaining budget to explore and refine
        # Try small perturbations of x[1:] to see if we can improve
        # Also try different x[0] values with non-zero x[1:]
        remaining = budget - used

        # Create a grid of x[0] values and try a few x[1:] configurations
        n_phase2_x0 = min(50, max(1, remaining // 4))
        x0_vals2 = np.linspace(0.0, 1.0, n_phase2_x0)

        # Some perturbations for x[1:]
        perturbations = [
            np.zeros(n - 1),
            np.ones(n - 1) * 0.01,
            np.ones(n - 1) * 0.05,
            np.ones(n - 1) * 0.1,
            rng.random(n - 1) * 0.1,
            rng.random(n - 1) * 0.2,
        ]

        for x0 in x0_vals2:
            if time.time() > deadline:
                break
            if used >= budget:
                break

            for pert in perturbations:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break

                x = np.zeros(n)
                x[0] = x0
                x[1:] = pert
                x = np.clip(x, lo, hi)
                f = problem(x)
                archive_F.append(f)
                archive_X.append(x.copy())
                used += 1

        # Phase 3: Local refinement around promising points
        # Find non-dominated points and try to improve them
        remaining = budget - used
        if remaining > 0 and len(archive_F) > 0:
            archive_F = np.array(archive_F)
            archive_X = np.array(archive_X)

            # Find non-dominated points
            def is_dominated(f, archive):
                """Check if f is dominated by any point in archive."""
                for a in archive:
                    if np.all(a <= f) and np.any(a < f):
                        return True
                return False

            # Get non-dominated points
            non_dominated = []
            for i in range(len(archive_F)):
                if not is_dominated(archive_F[i], list(archive_F[:i]) + list(archive_F[i+1:])):
                    non_dominated.append(i)

            # Refine around non-dominated points
            step_sizes = [0.01, 0.005, 0.001, 0.0005]
            for idx in non_dominated[:min(10, len(non_dominated))]:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break

                base_x = archive_X[idx].copy()
                for step in step_sizes:
                    if time.time() > deadline:
                        break
                    if used >= budget:
                        break

                    # Try small perturbations
                    for _ in range(min(3, budget - used)):
                        if time.time() > deadline:
                            break
                        if used >= budget:
                            break

                        x = base_x + step * rng.standard_normal(n)
                        x = np.clip(x, lo, hi)
                        f = problem(x)
                        archive_F = np.vstack([archive_F, [f]])
                        archive_X = np.vstack([archive_X, [x]])
                        used += 1

        return archive_X if len(archive_X) > 0 else np.array([lo + (hi - lo) * 0.5])
# EVOLVE-BLOCK-END
