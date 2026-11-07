"""Decomposition-based multi-objective optimizer for WFG7 using Tchebycheff scalarization."""
import time
import numpy as np


class Optimizer:
    """Decomposition-based optimizer using Tchebycheff scalarization with local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random.default_rng(42)

        # Reference point for hypervolume calculation
        ref_point = np.array([3.0, 5.0, 7.0])

        def _dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def _is_non_dominated(F, f):
            """Check if f is not dominated by any point in F."""
            for ef in F:
                if _dominates(ef, f):
                    return False
            return True

        def _remove_dominated(F, f):
            """Remove points from F that are dominated by f."""
            return [ef for ef in F if not _dominates(f, ef)]

        def _hypervolume_3d(points, ref):
            """Compute exact hypervolume for 3D points below reference point."""
            valid = [p for p in points if np.all(p < ref)]
            if len(valid) == 0:
                return 0.0
            if len(valid) == 1:
                p = valid[0]
                return float(np.prod(ref - p))

            # Sort by first objective
            valid.sort(key=lambda p: p[0])
            n = len(valid)

            total_hv = 0.0
            for i in range(n):
                x1, y1, z1 = valid[i]
                x2 = valid[i + 1][0] if i + 1 < n else ref[0]
                dx = x2 - x1

                # Compute 2D HV in y-z plane for points i..n-1
                yz_pts = [(valid[j][1], valid[j][2]) for j in range(i, n)
                          if valid[j][1] < ref[1] and valid[j][2] < ref[2]]
                if not yz_pts:
                    continue

                yz_pts.sort(key=lambda p: p[0])
                m = len(yz_pts)

                # Sweep from high y to low y
                min_z = ref[2]
                for j in range(m - 1, -1, -1):
                    y, z = yz_pts[j]
                    if z < min_z:
                        min_z = z
                    y_next = yz_pts[j + 1][0] if j + 1 < m else ref[1]
                    dy = y_next - y
                    dz = ref[2] - min_z
                    if dy > 0 and dz > 0:
                        total_hv += dx * dy * dz

            return total_hv

        def _tchebycheff(x, w, f_ideal):
            """Tchebycheff scalarization function."""
            f = problem(x)
            # Handle potential issues with f
            f = np.clip(f, 0, 100)
            return np.max(w * (f - f_ideal))

        # Generate weight vectors uniformly distributed in the simplex
        n_weights = 30
        # For 3 objectives, generate points on the 2D simplex
        weight_vectors = []
        for i in range(n_weights):
            # Use a grid approach for 3 objectives
            # i represents the index in a triangular grid
            row = 0
            count = 0
            for r in range(n_weights):
                if count + (n_weights - r) > i:
                    row = r
                    break
                count += (n_weights - r)
            col = i - count
            w1 = row / (n_weights - 1)
            w2 = col / (n_weights - 1 - row) if row < n_weights - 1 else 0
            w3 = 1 - w1 - w2
            weight_vectors.append(np.array([w1, w2, w3]))

        # Estimate ideal point (minimum of each objective)
        # Start with random points to get a rough estimate
        n_init = min(20, budget // 5)
        X_init = lo + (hi - lo) * rng.random((n_init, dim))
        F_init = np.array([problem(x) for x in X_init], dtype=np.float64)
        used = n_init

        f_ideal = np.min(F_init, axis=0)
        f_ideal = np.maximum(f_ideal, 0.001)  # Avoid zero

        # Initialize one solution per weight vector
        # Start from the center of the search space
        center = (lo + hi) / 2
        solutions = []
        objectives = []
        for w in weight_vectors:
            # Start near center with some randomness
            x = center + 0.1 * (hi - lo) * rng.standard_normal(dim)
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            solutions.append(x)
            objectives.append(f)

        # Local search using Nelder-Mead-like approach for each sub-problem
        # We'll use a simple coordinate descent or pattern search since Nelder-Mead
        # requires many function evaluations

        # Budget allocation: use remaining budget for local search
        remaining_budget = budget - used

        # For each weight vector, perform local search
        # We'll use a simple adaptive step size search
        base_step = 0.15  # Initial step size as fraction of range
        min_step = 0.005
        max_iterations_per_subproblem = max(1, remaining_budget // n_weights)

        # Sort weight vectors by expected difficulty (optional)
        # For now, process all

        for idx, w in enumerate(weight_vectors):
            if used >= budget:
                break
            if time.time() > deadline - 10:
                break

            # Get current solution
            x = solutions[idx].copy()
            f = objectives[idx].copy()

            # Local search: coordinate descent with adaptive step size
            step_size = base_step
            stagnation_count = 0
            max_stagnation = 5

            for iteration in range(max_iterations_per_subproblem):
                if used >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                improved = False

                # Try moving in each coordinate direction
                for d in range(dim):
                    if used >= budget:
                        break
                    if time.time() > deadline - 5:
                        break

                    # Try positive direction
                    x_trial = x.copy()
                    x_trial[d] += step_size * (hi[d] - lo[d])
                    x_trial = np.clip(x_trial, lo, hi)
                    f_trial = problem(x_trial)
                    used += 1

                    # Tchebycheff comparison
                    obj_current = np.max(w * (f - f_ideal))
                    obj_trial = np.max(w * (f_trial - f_ideal))

                    if obj_trial < obj_current:
                        x = x_trial
                        f = f_trial
                        improved = True
                        stagnation_count = 0
                    else:
                        # Try negative direction
                        x_trial = x.copy()
                        x_trial[d] -= step_size * (hi[d] - lo[d])
                        x_trial = np.clip(x_trial, lo, hi)
                        f_trial = problem(x_trial)
                        used += 1

                        obj_trial = np.max(w * (f_trial - f_ideal))
                        if obj_trial < obj_current:
                            x = x_trial
                            f = f_trial
                            improved = True
                            stagnation_count = 0

                # Update solutions
                solutions[idx] = x
                objectives[idx] = f

                if not improved:
                    stagnation_count += 1
                    step_size *= 0.7
                    if step_size < min_step or stagnation_count > max_stagnation:
                        break
                else:
                    step_size = min(step_size * 1.1, base_step)

        # Build archive from all solutions
        archive_X = []
        archive_F = []
        for i in range(n_weights):
            if _is_non_dominated(archive_F, objectives[i]):
                archive_F = _remove_dominated(archive_F, objectives[i])
                archive_F.append(objectives[i])
                archive_X.append(solutions[i])

        # Final polishing: local search on archive points
        if time.time() < deadline - 10 and len(archive_X) > 0:
            # Select a few archive points for polishing
            n_polish = min(5, len(archive_X), (budget - used) // 10)
            # Pick diverse points from archive
            if len(archive_X) > n_polish:
                indices = np.linspace(0, len(archive_X) - 1, n_polish, dtype=int)
            else:
                indices = range(len(archive_X))

            for idx in indices:
                if used >= budget or time.time() > deadline - 5:
                    break

                base_x = archive_X[idx].copy()
                base_f = archive_F[idx].copy()
                step_size = 0.02

                for _ in range(3):
                    if used >= budget or time.time() > deadline - 3:
                        break

                    # Random perturbation
                    pert = base_x + step_size * (hi - lo) * rng.standard_normal(dim)
                    pert = np.clip(pert, lo, hi)
                    fp = problem(pert)
                    used += 1

                    # Check if it improves the archive
                    if _is_non_dominated(archive_F, fp):
                        archive_F = _remove_dominated(archive_F, fp)
                        archive_F.append(fp)
                        archive_X.append(pert)
                        # Update base if this is better in sum
                        if np.sum(fp) < np.sum(base_f):
                            base_x = pert
                            base_f = fp

        return solutions
