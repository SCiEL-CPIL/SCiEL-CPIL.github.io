import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

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
            full_hv = compute_hv_2d(archive_f, [3000.0, 0.06])
            reduced_f = [f for i, f in enumerate(archive_f) if i != idx]
            reduced_hv = compute_hv_2d(reduced_f, [3000.0, 0.06])
            return full_hv - reduced_hv

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

        # 1. Initialization: Halton sequence for better coverage
        init_size = 25
        for i in range(1, init_size + 1):
            if not time_ok():
                break
            p = halton_point(i, self.dim)
            x = lo + range_ * p
            f = problem(x)
            evals += 1
            add_to_archive(x, f)

        if not time_ok():
            return archive_x

        # 2. Main search: adaptive step size + HV-based parent selection
        # Track best HV and adapt step size
        best_hv = compute_hv_2d(archive_f, [3000.0, 0.06])
        base_step = 0.15 * range_  # Start with moderate step
        current_step = base_step.copy()
        improvement_count = 0
        no_improvement_count = 0
        max_no_improvement = 15

        while evals < self.budget and time_ok():
            if len(archive_x) == 0:
                break

            # Parent selection: HV-based when archive is substantial
            if len(archive_x) > 8:
                # Use HV contribution for selection
                contributions = [hv_contribution(i) for i in range(len(archive_x))]
                total = sum(contributions)
                if total > 0:
                    probs = [c / total for c in contributions]
                else:
                    probs = [1.0 / len(archive_x)] * len(archive_x)
                idx = np.random.choice(len(archive_x), p=probs)
            else:
                idx = np.random.randint(len(archive_x))

            parent = archive_x[idx]

            # Generate multiple children with varying step sizes for diversity
            n_children = 1
            if len(archive_x) < 15 and evals < self.budget - 10:
                n_children = 2  # More exploration early on

            for c in range(n_children):
                if evals >= self.budget or not time_ok():
                    break

                # Adaptive step: mix of current step and random perturbation
                if c == 0:
                    step = current_step
                else:
                    step = current_step * (0.5 + np.random.random())

                child = parent + step * np.random.normal(size=self.dim)
                child = np.clip(child, lo, hi)

                f_child = problem(child)
                evals += 1

                added = add_to_archive(child, f_child)

                if added:
                    new_hv = compute_hv_2d(archive_f, [3000.0, 0.06])
                    if new_hv > best_hv:
                        best_hv = new_hv
                        improvement_count += 1
                        no_improvement_count = 0
                        # Slightly reduce step on improvement (exploit more)
                        current_step *= 0.95
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1

            # Adapt step size based on improvement history
            if no_improvement_count >= max_no_improvement:
                # Increase step to explore more
                current_step = np.minimum(current_step * 1.3, base_step)
                no_improvement_count = 0
            elif improvement_count > 0 and no_improvement_count == 0:
                # Continue shrinking on sustained improvement
                current_step *= 0.97

            # Ensure step doesn't get too small
            current_step = np.maximum(current_step, 0.02 * range_)

        # 3. Final polish: local refinement around archive points
        if time_ok() and evals < self.budget:
            polish_budget = min(self.budget - evals, 30)
            if polish_budget > 0 and len(archive_x) > 0:
                # Focus on points with high HV contribution
                if len(archive_x) > 5:
                    contributions = [hv_contribution(i) for i in range(len(archive_x))]
                    # Sort by contribution, take top half
                    sorted_idx = np.argsort(contributions)[::-1]
                    top_n = max(3, len(archive_x) // 2)
                    top_indices = sorted_idx[:top_n]
                else:
                    top_indices = list(range(len(archive_x)))

                # Adaptive local step for polish
                polish_step = 0.03 * range_

                for idx in top_indices:
                    if evals >= self.budget or not time_ok():
                        break

                    parent = archive_x[idx]
                    n_local = min(5, self.budget - evals)

                    for _ in range(n_local):
                        if evals >= self.budget or not time_ok():
                            break

                        # Decreasing step for local search
                        local_step = polish_step * (0.5 + 0.5 * np.random.random())
                        child = parent + local_step * np.random.normal(size=self.dim)
                        child = np.clip(child, lo, hi)

                        f_child = problem(child)
                        evals += 1
                        add_to_archive(child, f_child)

        return archive_x
