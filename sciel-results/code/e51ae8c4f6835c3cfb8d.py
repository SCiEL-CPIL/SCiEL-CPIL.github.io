# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer combining grid decomposition, Halton sampling, and local simplex polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # --- Archive management ---
        archive_x = []
        archive_f = []
        evals = 0

        def time_ok():
            return time.time() < deadline

        def is_dominated(f_new, archive_f_list):
            if len(archive_f_list) == 0:
                return False
            af = np.array(archive_f_list)
            leq = np.all(af <= f_new, axis=1)
            lt = np.any(af < f_new, axis=1)
            return np.any(leq & lt)

        def dominates(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        def add_to_archive(x, f):
            nonlocal evals
            if is_dominated(f, archive_f):
                return False
            new_archive_x = []
            new_archive_f = []
            for xi, fi in zip(archive_x, archive_f):
                if dominates(f, fi):
                    continue
                new_archive_x.append(xi)
                new_archive_f.append(fi)
            new_archive_x.append(x)
            new_archive_f.append(f)
            archive_x[:] = new_archive_x
            archive_f[:] = new_archive_f
            return True

        def compute_hv_2d(f_list, ref_point):
            if not f_list:
                return 0.0
            pts = [f for f in f_list if np.all(f < ref_point)]
            if not pts:
                return 0.0
            pts = sorted(pts, key=lambda x: x[0])
            hv = 0.0
            ref = np.array(ref_point)
            for i in range(len(pts)):
                f1, f2 = pts[i]
                next_f1 = pts[i + 1][0] if i < len(pts) - 1 else ref[0]
                width = next_f1 - f1
                height = ref[1] - f2
                if width > 0 and height > 0:
                    hv += width * height
            return hv

        def hv_contribution(idx):
            ref_point = [3000.0, 0.06]
            full_hv = compute_hv_2d(archive_f, ref_point)
            reduced_f = [f for i, f in enumerate(archive_f) if i != idx]
            reduced_hv = compute_hv_2d(reduced_f, ref_point)
            return full_hv - reduced_hv

        def crowding_distance(f_list):
            n = len(f_list)
            if n <= 2:
                return [float('inf')] * n
            distances = [0.0] * n
            f_arr = np.array(f_list)
            for m in range(self.n_obj):
                sorted_idx = np.argsort(f_arr[:, m])
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                f_min = f_arr[sorted_idx[0], m]
                f_max = f_arr[sorted_idx[-1], m]
                f_range = f_max - f_min
                if f_range == 0:
                    continue
                for i in range(1, n - 1):
                    idx = sorted_idx[i]
                    prev_idx = sorted_idx[i - 1]
                    next_idx = sorted_idx[i + 1]
                    distances[idx] += (f_arr[next_idx, m] - f_arr[prev_idx, m]) / f_range
            return distances

        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(n, dim):
            p = np.zeros(dim)
            primes = [2, 3, 5, 7]
            for d in range(dim):
                p[d] = halton_sequence(n, primes[d])
            return p

        # --- Phase 1: Halton initialization for uniform coverage ---
        init_size = 20
        for i in range(1, init_size + 1):
            if not time_ok():
                break
            p = halton_point(i, self.dim)
            x = lo + span * p
            f = problem(x)
            evals += 1
            add_to_archive(x, f)

        # --- Phase 2: Grid-based decomposition with local refinement ---
        # Partition the 4D space into a grid and explore each cell
        # Number of grid points per dimension: keep total cells manageable
        # For 4D, use 2x2x2x2 = 16 cells, each with a few evaluations
        grid_per_dim = 2
        total_cells = grid_per_dim ** self.dim

        if time_ok() and evals < self.budget:
            # Generate grid cell centers
            cell_centers = []
            for d in range(self.dim):
                # Midpoints of each cell
                cell_centers.append(np.linspace(lo[d] + span[d] / (2 * grid_per_dim),
                                                hi[d] - span[d] / (2 * grid_per_dim),
                                                grid_per_dim))

            # Create all grid points
            grid_points = np.array(np.meshgrid(*cell_centers, indexing='ij')).reshape(self.dim, -1).T

            # Evaluate all grid points
            for x in grid_points:
                if not time_ok() or evals >= self.budget:
                    break
                f = problem(x)
                evals += 1
                add_to_archive(x, f)

        # --- Phase 3: Local simplex-style refinement on archive points ---
        # For each point in the archive, try to improve it locally
        if time_ok() and evals < self.budget:
            # Sort archive by HV contribution to prioritize high-value points
            if len(archive_f) > 0:
                ref_point = [3000.0, 0.06]
                contributions = [hv_contribution(i) for i in range(len(archive_f))]
                order = np.argsort(contributions)[::-1]  # descending

                # Try to improve top points
                for idx in order:
                    if not time_ok() or evals >= self.budget:
                        break

                    x0 = np.array(archive_x[idx])
                    f0 = np.array(archive_f[idx])

                    # Local exploration: small perturbations around x0
                    # Use a few random steps with decreasing step size
                    base_step = 0.05 * span
                    for step_scale in [1.0, 0.5, 0.25]:
                        if not time_ok() or evals >= self.budget:
                            break

                        # Generate a few local candidates
                        for _ in range(3):
                            if not time_ok() or evals >= self.budget:
                                break
                            step = base_step * step_scale * np.random.randn(self.dim)
                            x_new = np.clip(x0 + step, lo, hi)
                            f_new = problem(x_new)
                            evals += 1

                            # Add if non-dominated
                            if not is_dominated(f_new, archive_f):
                                # Remove points dominated by f_new
                                new_archive_x = []
                                new_archive_f = []
                                for xi, fi in zip(archive_x, archive_f):
                                    if dominates(f_new, fi):
                                        continue
                                    new_archive_x.append(xi)
                                    new_archive_f.append(fi)
                                new_archive_x.append(x_new)
                                new_archive_f.append(f_new)
                                archive_x[:] = new_archive_x
                                archive_f[:] = new_archive_f
                                x0 = x_new  # Update starting point
                                f0 = f_new
                                break  # Restart local search from improved point

        # --- Phase 4: Final polish with targeted mutation ---
        if time_ok() and evals < self.budget:
            # Use remaining budget for random mutations of archive points
            while time_ok() and evals < self.budget:
                if len(archive_x) == 0:
                    break

                # Select parent: mix of random and HV-based
                if len(archive_x) > 10 and np.random.random() < 0.5:
                    # HV-based selection
                    contributions = [hv_contribution(i) for i in range(len(archive_f))]
                    idx = int(np.argmax(contributions))
                else:
                    idx = np.random.randint(len(archive_x))

                parent = np.array(archive_x[idx])

                # Adaptive mutation
                progress = evals / self.budget
                step_size = 0.1 * (1.0 - 0.3 * progress)
                child = parent + step_size * span * np.random.randn(self.dim)
                child = np.clip(child, lo, hi)

                f_child = problem(child)
                evals += 1

                # Add if non-dominated
                if not is_dominated(f_child, archive_f):
                    new_archive_x = []
                    new_archive_f = []
                    for xi, fi in zip(archive_x, archive_f):
                        if dominates(f_child, fi):
                            continue
                        new_archive_x.append(xi)
                        new_archive_f.append(fi)
                    new_archive_x.append(child)
                    new_archive_f.append(f_child)
                    archive_x[:] = new_archive_x
                    archive_f[:] = new_archive_f

        # Return all evaluated points (the evaluator records all objective vectors)
        # We return archive_x as the "result", but the evaluator uses recorded vectors
        return archive_x
# EVOLVE-BLOCK-END
