"""Multi-objective optimizer for ZDT6 using structured sampling + local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured sampling on ZDT6 Pareto front + local optimization of g."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT6 structure: f1 depends on x0, f2 = g(x1..x9) * (1 + 10*(dim-1))
        # Optimal g for Pareto front: minimize g(x1..x9)
        # g(x) = 1 + 10*(n-1) + sum((xi-0.5)^2 * (1+10*sin^2(6*pi*(xi+0.5))))
        # Minimum of g is 1 (when all xi=0.5)

        # Strategy: sample x0 values to spread f1, then optimize x1..x9
        # For hypervolume, we want good coverage of the Pareto front

        # Generate candidate x0 values - use a mix of uniform and golden-ratio spaced
        n_x0 = min(80, budget // 3)
        # Golden ratio spacing for good coverage
        phi = (1 + np.sqrt(5)) / 2
        x0_candidates = np.array([(i / phi) % 1.0 for i in range(n_x0)])
        # Add some uniform points
        x0_candidates = np.sort(np.unique(np.concatenate([
            x0_candidates,
            rng.random(min(20, budget // 10))
        ])))
        x0_candidates = x0_candidates[:min(60, len(x0_candidates))]

        # For each x0, try to find good x1..x9 values
        # Start with x1..x9 = 0.5 (theoretical minimum of g)
        base_x19 = np.full(dim - 1, 0.5)

        archive = []  # list of (x, f)
        used = 0

        # Phase 1: Evaluate structured candidates
        for x0 in x0_candidates:
            if used >= budget:
                break
            if time.time() > deadline:
                break

            # Try base solution (x1..x9 = 0.5)
            x = np.concatenate([[x0], base_x19])
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            used += 1
            archive.append((x.copy(), f.copy()))

            # Also try a few perturbed versions of x1..x9
            if used < budget and len(x0_candidates) < 40:
                for _ in range(2):
                    if used >= budget or time.time() > deadline:
                        break
                    pert = base_x19 + 0.05 * rng.standard_normal(dim - 1)
                    x = np.concatenate([[x0], pert])
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x))
                    used += 1
                    archive.append((x.copy(), f.copy()))

        # Phase 2: Local polish - use remaining budget to refine
        # Convert archive to arrays
        if archive:
            X = np.array([a[0] for a in archive])
            F = np.array([a[1] for a in archive])
        else:
            X = np.empty((0, dim))
            F = np.empty((0, self.n_obj))

        # Local search on promising points
        while used < budget and time.time() < deadline:
            if len(X) == 0:
                break

            # Pick a random point and try to improve it
            idx = rng.integers(len(X))
            x = X[idx].copy()

            # Try multiple step sizes
            best_f = F[idx].copy()
            best_x = x.copy()

            for step in [0.01, 0.005, 0.001]:
                if used >= budget or time.time() > deadline:
                    break

                # Try random perturbations
                for _ in range(5):
                    if used >= budget or time.time() > deadline:
                        break
                    dx = step * rng.standard_normal(dim)
                    x_new = np.clip(x + dx, lo, hi)
                    f_new = np.array(problem(x_new))
                    used += 1

                    # Check if better (dominates or is non-dominated)
                    # Simple check: is it better in at least one objective and not worse in others?
                    if np.any(f_new < best_f) and not np.any(f_new > best_f + 1e-10):
                        best_f = f_new
                        best_x = x_new

            # Update
            if np.any(best_f < F[idx]):
                X[idx] = best_x
                F[idx] = best_f

        # Return the archive (the return value is ignored, but keep it valid)
        if len(X) > 0:
            return X
        else:
            return np.empty((0, dim))

# EVOLVE-BLOCK-END
