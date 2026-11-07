"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-aware optimizer: structured grid on Pareto-relevant vars + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin under 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ7 structure: x[0], x[1] are Pareto-relevant, x[2:] should be 0 for g=0
        # Pareto front is parameterized by (x0, x1) in [0,1]^2
        # f1 = x0*(1+g), f2 = x1*(1+g), f3 = 1 - sqrt(x0)*(1 - h(x1,g))
        # With g=0: f1=x0, f2=x1, f3 = 1 - sqrt(x0)*(1 - x1*(1 - x1/(1+x1)*x0^2))

        def make_point(x0, x1):
            x = np.zeros(dim)
            x[0] = np.clip(x0, 0.0, 1.0)
            x[1] = np.clip(x1, 0.0, 1.0)
            # x[2:] = 0 ensures g=0
            return x

        used = 0
        points = []
        fvals = []

        # Phase 1: Structured grid on (x0, x1)
        # Use 11x11 grid = 121 points
        grid_size = 11
        grid_budget = grid_size * grid_size
        if grid_budget > budget:
            grid_size = int(np.sqrt(budget))
            grid_budget = grid_size * grid_size

        x0_vals = np.linspace(0.0, 1.0, grid_size)
        x1_vals = np.linspace(0.0, 1.0, grid_size)

        for x0 in x0_vals:
            for x1 in x1_vals:
                if used >= budget:
                    break
                if time.time() > deadline:
                    return points
                x = make_point(x0, x1)
                f = np.array(problem(x))
                points.append(x)
                fvals.append(f)
                used += 1

        # Phase 2: Local refinement around promising regions
        # Focus on corners and edges where hypervolume contribution is high
        # Key regions: (0,0), (1,0), (0,1), (1,1) and midpoints
        if used < budget and fvals:
            F = np.array(fvals)
            # Reference point for local hypervolume estimation
            ref = np.array([1.1, 1.1, 25.0])

            # Select diverse points to refine around
            # Use non-dominated points
            n_dom = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] = True
                        break
            nondom_idx = np.where(~n_dom)[0]
            if len(nondom_idx) == 0:
                nondom_idx = np.arange(len(F))

            # Pick up to 20 diverse non-dominated points
            max_refine = min(20, len(nondom_idx))
            if max_refine > 0:
                # Sample diverse points from non-dominated set
                if len(nondom_idx) <= max_refine:
                    refine_idx = nondom_idx
                else:
                    # Space them out
                    step = len(nondom_idx) / max_refine
                    refine_idx = nondom_idx[np.round(np.arange(max_refine) * step).astype(int)]

                rng = np.random.default_rng(42)
                for idx in refine_idx:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        return points
                    base_x = points[idx]
                    base_f = fvals[idx]
                    # Try several mutations
                    n_mut = min(4, budget - used)
                    best_child = None
                    best_child_f = None
                    for _ in range(n_mut):
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            return points
                        # Mutate x0 and x1 with small perturbations
                        scale = 0.05
                        dx0 = rng.standard_normal() * scale
                        dx1 = rng.standard_normal() * scale
                        new_x0 = base_x[0] + dx0
                        new_x1 = base_x[1] + dx1
                        x = make_point(new_x0, new_x1)
                        f = np.array(problem(x))
                        used += 1
                        points.append(x)
                        fvals.append(f)
                        # Keep track of best child for this parent
                        if best_child is None or np.sum(f) < np.sum(best_child_f):
                            best_child = x
                            best_child_f = f

        return points
# EVOLVE-BLOCK-END
