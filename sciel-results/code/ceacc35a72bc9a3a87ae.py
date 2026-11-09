"""Multi-objective optimizer on RE21 using HypE with coordinate-descent polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """HypE-style multi-objective optimizer with coordinate-descent local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref = np.array([3000.0, 0.06], dtype=np.float64)

        archive = []  # list of (x, f) tuples
        used = 0

        def is_valid(f):
            return np.all(f < ref) and np.all(np.isfinite(f))

        def compute_hypervolume(points, ref_point):
            """Compute hypervolume of 2D points relative to ref_point."""
            if len(points) == 0:
                return 0.0
            # Sort by first objective ascending
            sorted_pts = sorted(points, key=lambda p: p[0])
            hv = 0.0
            prev_y = ref_point[1]
            for p in sorted_pts:
                if p[0] >= ref_point[0] or p[1] >= ref_point[1]:
                    continue
                if p[1] < prev_y:
                    hv += (ref_point[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        def marginal_hypervolume(point, archive_points, ref_point):
            """Compute marginal hypervolume contribution of a point."""
            hv_with = compute_hypervolume(archive_points + [point], ref_point)
            hv_without = compute_hypervolume(archive_points, ref_point)
            return hv_with - hv_without

        def add_to_archive(x, f):
            nonlocal used
            if not is_valid(f):
                return False
            # Check if dominated by existing archive
            for _, af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return False
            # Remove dominated points
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            archive[:] = new_archive
            return True

        def evaluate(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            f = problem(x)
            used += 1
            return f

        def coordinate_descent_polish(x_init, f_init, max_steps=8):
            """
            Perform coordinate-descent local search from x_init.
            For each coordinate, try a small perturbation and keep if it improves
            the hypervolume contribution of the resulting point.
            """
            if time.time() > deadline or used >= self.budget:
                return x_init, f_init

            x_best = x_init.copy()
            f_best = f_init.copy()

            # Compute current hypervolume contribution
            archive_points = [f for _, f in archive]
            current_contrib = marginal_hypervolume(f_best, archive_points, ref)

            step_size = 0.02  # relative step size for coordinate descent

            for step in range(max_steps):
                if time.time() > deadline or used >= self.bBudget:
                    break

                improved = False
                for d in range(self.dim):
                    if time.time() > deadline or used >= self.budget:
                        break

                    # Try positive perturbation
                    x_try = x_best.copy()
                    span_d = hi[d] - lo[d]
                    x_try[d] = x_best[d] + step_size * span_d
                    x_try[d] = np.clip(x_try[d], lo[d], hi[d])

                    f_try = evaluate(x_try)
                    if f_try is None:
                        break

                    if is_valid(f_try):
                        # Check if this would improve hypervolume contribution
                        # by seeing if it's non-dominated and adds hypervolume
                        dominated = False
                        for _, af in archive:
                            if np.all(af <= f_try) and np.any(af < f_try):
                                dominated = True
                                break

                        if not dominated:
                            # Compute potential contribution
                            potential_contrib = marginal_hypervolume(f_try, archive_points, ref)
                            if potential_contrib > current_contrib * 0.5:  # some improvement
                                x_best = x_try
                                f_best = f_try
                                current_contrib = potential_contrib
                                improved = True
                                break  # restart coordinate loop

                    # Try negative perturbation
                    x_try = x_best.copy()
                    x_try[d] = x_best[d] - step_size * span_d
                    x_try[d] = np.clip(x_try[d], lo[d], hi[d])

                    f_try = evaluate(x_try)
                    if f_try is None:
                        break

                    if is_valid(f_try):
                        dominated = False
                        for _, af in archive:
                            if np.all(af <= f_try) and np.any(af < f_try):
                                dominated = True
                                break

                        if not dominated:
                            potential_contrib = marginal_hypervolume(f_try, archive_points, ref)
                            if potential_contrib > current_contrib * 0.5:
                                x_best = x_try
                                f_best = f_try
                                current_contrib = potential_contrib
                                improved = True
                                break

                if not improved:
                    break

                # Decrease step size
                step_size *= 0.7

            return x_best, f_best

        # Phase 1: Stratified initialization
        n_init = min(40, self.budget)
        for i in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            x = np.zeros(self.dim)
            for d in range(self.dim):
                x[d] = lo[d] + (hi[d] - lo[d]) * ((i + np.random.random()) / n_init)
            f = evaluate(x)
            if f is not None and is_valid(f):
                add_to_archive(x, f)

        # Phase 2: HypE-style evolutionary loop with coordinate descent polish
        sigma = 0.15  # initial relative step size

        while used < self.budget and time.time() < deadline:
            if len(archive) == 0:
                # If archive is empty, do random search
                x = np.random.uniform(lo, hi)
                f = evaluate(x)
                if f is not None and is_valid(f):
                    add_to_archive(x, f)
                continue

            # Compute marginal hypervolume contributions
            archive_points = [f for _, f in archive]
            contributions = []
            for i, (x_i, f_i) in enumerate(archive):
                contrib = marginal_hypervolume(f_i, archive_points, ref)
                contributions.append(contrib)

            # Select parent with probability proportional to contribution
            total_contrib = sum(contributions)
            if total_contrib <= 0:
                parent_idx = np.random.randint(len(archive))
            else:
                probs = np.array(contributions) / total_contrib
                parent_idx = np.random.choice(len(archive), p=probs)

            parent_x, parent_f = archive[parent_idx]

            # Generate offspring via Gaussian mutation
            x_off = parent_x.copy()
            for d in range(self.dim):
                step = sigma * (hi[d] - lo[d])
                x_off[d] += np.random.normal(0, step)
                x_off[d] = np.clip(x_off[d], lo[d], hi[d])

            f_off = evaluate(x_off)

            if f_off is not None and is_valid(f_off):
                # Check if offspring is non-dominated
                dominated = False
                for _, af in archive:
                    if np.all(af <= f_off) and np.any(af < f_off):
                        dominated = True
                        break

                if not dominated:
                    # Apply coordinate descent polish
                    x_polished, f_polished = coordinate_descent_polish(x_off, f_off, max_steps=4)
                    add_to_archive(x_polished, f_polished)
                    sigma *= 0.95  # slight decay on success
                else:
                    sigma *= 1.05  # slight increase on failure
            else:
                sigma *= 1.05

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 0.5)

        # Phase 3: Final polish pass on all archive points
        if time.time() < deadline and used < self.budget:
            for i in range(len(archive)):
                if time.time() > deadline or used >= self.budget:
                    break
                x_i, f_i = archive[i]
                x_polished, f_polished = coordinate_descent_polish(x_i, f_i, max_steps=3)
                if is_valid(f_polished):
                    add_to_archive(x_polished, f_polished)

        return archive

# EVOLVE-BLOCK-END
