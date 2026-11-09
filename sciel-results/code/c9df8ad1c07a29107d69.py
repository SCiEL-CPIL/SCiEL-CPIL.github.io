"""Multi-objective optimizer on RE21 using HypE-style hypervolume contribution selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """HypE-style multi-objective optimizer with hypervolume contribution selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref = np.array([3000.0, 0.06], dtype=np.float64)

        def is_valid(f):
            return np.all(f < ref)

        # Archive stored as sorted list by f[0] for efficient HV computation
        archive = []  # list of (x, f) sorted by f[0]
        used = 0

        def add_to_archive(x, f):
            nonlocal used
            if not is_valid(f):
                return
            # Check if dominated by any point in archive
            for ax, af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return  # dominated, don't add
            # Remove dominated points from archive
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            # Sort by first objective for efficient HV
            new_archive.sort(key=lambda item: item[1][0])
            archive[:] = new_archive

        def hv_2d_sorted(sorted_pts, ref_pt):
            """Compute hypervolume in 2D for points already sorted by f[0]."""
            if not sorted_pts:
                return 0.0
            hv = 0.0
            prev_y = ref_pt[1]
            for _, p in sorted_pts:
                if p[1] >= ref_pt[1]:
                    continue
                width = ref_pt[0] - p[0]
                height = prev_y - p[1]
                if width > 0 and height > 0:
                    hv += width * height
                prev_y = min(prev_y, p[1])
            return hv

        def marginal_contributions_fast(sorted_archive, ref_pt):
            """Compute marginal HV contribution using sorted archive. O(n) total."""
            n = len(sorted_archive)
            if n == 0:
                return np.array([])
            
            # Compute cumulative HV from right to left
            # For each point i, its contribution is the area it uniquely covers
            contribs = np.zeros(n)
            
            # Track the minimum y seen so far from the right
            # Point i contributes: (ref[0] - f_i[0]) * (min_y_right - f_i[1]) if f_i[1] < min_y_right
            min_y_from_right = ref_pt[1]
            
            # We need to be careful: contribution of point i is the area between
            # f_i[0] and the next point's f[0] (or ref[0]), and between f_i[1] and
            # the minimum y of all points to its right (including itself)
            
            # Actually, let's use the standard approach:
            # Sort by f[0] ascending. For each point, its marginal contribution
            # is the rectangle from (f_i[0], f_i[1]) to (next_f0_or_ref, prev_min_y)
            # where prev_min_y is the minimum y of points with smaller f[0]... 
            # No, it's the minimum y of points with LARGER f[0] (to the right).
            
            # Let me think again. In 2D HV with points sorted by x ascending:
            # The HV is sum over i of (x_{i+1} - x_i) * (y_min_suffix - y_i)
            # where x_{i+1} is next point's x or ref_x, and y_min_suffix is min y from i to end
            # Actually no. Let me use the standard formula.
            
            # For points sorted by f1 ascending, the hypervolume is:
            # HV = sum_{i=0}^{n-1} (x_{i+1} - x_i) * (y_i - y_{i+1})
            # where x_n = ref[0], y_0 = ref[1], and we only include points with y < ref[1]
            # Wait, that's not quite right either.
            
            # Standard 2D HV: sort by f1 asc. 
            # HV = sum over i where f1_i < ref[0] and f2_i < ref[1]:
            #     (f1_{i+1} - f1_i) * (f2_i - f2_{i+1})
            # where f1_n = ref[0], f2_0 = ref[1], and we skip points with f2 >= ref[1]
            
            # Actually the cleanest: 
            # Filter valid points, sort by f1 asc.
            # HV = sum_{i} (x_{i+1} - x_i) * (y_i - y_{i+1})
            # with x_0 = first point's f1, y_0 = ref[1]... no.
            
            # Let me just use the incremental approach from the original but optimized.
            # The marginal contribution of point i is the area it adds when inserted.
            
            # For a sorted list, inserting point i adds area:
            # (x_right - x_i) * (y_prev_min - y_i) where x_right is next point's x or ref,
            # and y_prev_min is the min y of all points with x > x_i
            
            # Let me precompute suffix minimum y
            f1_vals = np.array([p[1][0] for p in sorted_archive])
            f2_vals = np.array([p[1][1] for p in sorted_archive])
            
            # Suffix min of f2: min_y_from[i] = min(f2[i], f2[i+1], ..., f2[n-1])
            suffix_min_y = np.minimum.accumulate(f2_vals[::-1])[::-1]
            
            # For point i, its contribution is:
            # width = (f1[i+1] if i < n-1 else ref[0]) - f1[i]
            # height = suffix_min_y[i] - f2[i]  ... no, that's not right
            # The contribution is the area that point i uniquely covers.
            # It's the rectangle from (f1[i], f2[i]) to (next_x, prev_y_min)
            # where next_x is f1[i+1] or ref[0], and prev_y_min is the min y of points
            # that have f1 > f1[i] (i.e., to the right). But we also need to account
            # for the fact that points to the left might have lower y.
            
            # Actually, the correct marginal contribution in 2D for a Pareto-optimal
            # set sorted by f1 ascending:
            # contrib[i] = (x_{i+1} - x_i) * (y_i - y_{i+1})  ... no
            # 
            # Let me just compute it directly but efficiently.
            # Total HV with all points, then HV without each point.
            # But that's O(n^2). For n up to ~100, it's fine.
            
            # For now, keep the O(n^2) but make it faster with numpy
            total_hv = hv_2d_sorted(sorted_archive, ref_pt)
            contribs = np.zeros(n)
            for i in range(n):
                remaining = sorted_archive[:i] + sorted_archive[i+1:]
                hv_without = hv_2d_sorted(remaining, ref_pt)
                contribs[i] = total_hv - hv_without
            return contribs

        def sample_uniform():
            return lo + (hi - lo) * np.random.random(self.dim)

        def sample_gaussian(parent, sigma_scale):
            sigma = sigma_scale * (hi - lo)
            child = parent + sigma * np.random.standard_normal(self.dim)
            return np.clip(child, lo, hi)

        # Phase 1: Stratified initialization
        n_init = min(40, self.budget // 3)
        for i in range(n_init):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            for d in range(self.dim):
                bin_start = i / n_init
                bin_end = (i + 1) / n_init
                x[d] = lo[d] + (hi[d] - lo[d]) * (bin_start + np.random.random() * (bin_end - bin_start))
            f = problem(x)
            used += 1
            add_to_archive(x, f)

        # Phase 2: HypE-style selection and mutation
        mutation_sigma = 0.25
        min_sigma = 0.005
        max_sigma = 0.4

        while used < self.budget:
            if time.time() > deadline:
                break

            if len(archive) >= 2:
                contribs = marginal_contributions_fast(archive, ref)
                contribs = np.maximum(contribs, 1e-12)
                probs = contribs / contribs.sum()
                idx = np.random.choice(len(archive), p=probs)
            elif len(archive) == 1:
                idx = 0
            else:
                x = sample_uniform()
                f = problem(x)
                used += 1
                add_to_archive(x, f)
                continue

            parent_x, parent_f = archive[idx]

            r = np.random.random()
            if r < 0.25:
                x = sample_uniform()
            else:
                x = sample_gaussian(parent_x, mutation_sigma)

            f = problem(x)
            used += 1
            add_to_archive(x, f)

            # Adapt sigma based on archive size
            if len(archive) > 0:
                mutation_sigma = max(min_sigma, mutation_sigma * 0.997)

        # Phase 3: Local refinement on top archive points
        if used < self.budget and len(archive) > 0 and time.time() < deadline:
            # Pick top-k points by marginal contribution for local search
            if len(archive) >= 2:
                contribs = marginal_contributions_fast(archive, ref)
                top_k = np.argsort(contribs)[::-1][:min(5, len(archive))]
            else:
                top_k = [0]
            
            for idx in top_k:
                if used >= self.budget or time.time() > deadline:
                    break
                parent_x, parent_f = archive[idx]
                # Try a few local perturbations
                for trial in range(3):
                    if used >= self.budget or time.time() > deadline:
                        break
                    # Small perturbation
                    x = parent_x + 0.02 * (hi - lo) * np.random.standard_normal(self.dim)
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    used += 1
                    add_to_archive(x, f)

        return archive
# EVOLVE-BLOCK-END
