# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Decomposition-based multi-objective optimizer with local scalarized refinement for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _gen_reference_directions(self, n_obj, n_dirs):
        """Generate reference directions (weight vectors) on the unit simplex."""
        if n_obj == 1:
            return np.array([[1.0]])
        if n_obj == 2:
            weights = []
            for i in range(n_dirs):
                w0 = i / (n_dirs - 1) if n_dirs > 1 else 0.5
                w1 = 1.0 - w0
                weights.append([w0, w1])
            return np.array(weights)
        # n_obj == 3: use a systematic grid on the 2-simplex
        points = []
        n = max(2, int(np.ceil(np.sqrt(n_dirs))))
        for i in range(n):
            for j in range(n - i):
                k = n - i - j
                if i + j + k == n - 1:
                    points.append([i / (n - 1), j / (n - 1), k / (n - 1)])
        pts = np.array(points)
        if len(pts) < n_dirs:
            # Pad with random points on simplex
            rng = np.random.default_rng(42)
            while len(pts) < n_dirs:
                r = rng.random(n_obj)
                pts = np.vstack([pts, r / r.sum()])
        elif len(pts) > n_dirs:
            # Subsample
            idx = np.linspace(0, len(pts) - 1, n_dirs).astype(int)
            pts = pts[idx]
        return pts[:n_dirs]

    def _tchebycheff_offspring(self, parent, weights, ideal, nadir, lo, hi, rng, sigma):
        """Generate an offspring using Tchebycheff-based mutation toward a reference direction."""
        child = parent.copy()
        # Adaptive step size
        step = sigma * (hi - lo)
        child = child + step * rng.standard_normal(self.dim)
        child = np.clip(child, lo, hi)
        return child

    def _is_dominated(self, f, archive_F):
        """Check if f is dominated by any point in archive_F."""
        return np.any(np.all(archive_F <= f, axis=1) & np.any(archive_F < f, axis=1))

    def _update_archive(self, x, f, archive_X, archive_F, max_archive):
        """Add (x, f) to archive if non-dominated, removing dominated points."""
        if self._is_dominated(f, archive_F):
            return archive_X, archive_F
        # Remove points dominated by f
        keep = ~np.all(f <= archive_F, axis=1) & ~np.any(f < archive_F, axis=1)
        # Actually: keep points that are NOT dominated by f
        dominated_by_f = np.all(f <= archive_F, axis=1) & np.any(f < archive_F, axis=1)
        keep = ~dominated_by_f
        archive_X = archive_X[keep]
        archive_F = archive_F[keep]
        # Add new point
        archive_X = np.vstack([archive_X, x.reshape(1, -1)])
        archive_F = np.vstack([archive_F, f.reshape(1, -1)])
        # Enforce max size: remove the point with worst hypervolume contribution (approx by sum)
        if len(archive_X) > max_archive:
            scores = archive_F.sum(axis=1)
            worst_idx = np.argmax(scores)
            archive_X = np.delete(archive_X, worst_idx, axis=0)
            archive_F = np.delete(archive_F, worst_idx, axis=0)
        return archive_X, archive_F

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(0)

        # Phase 1: Decomposition-based exploration
        n_dirs = min(30, budget // 6)
        n_dirs = max(5, n_dirs)
        weights = self._gen_reference_directions(n_obj, n_dirs)

        # Initialize population with reference-direction-based sampling
        pop_size = n_dirs
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        for i in range(pop_size):
            X[i] = lo + (hi - lo) * rng.random(dim)
            F[i] = problem(X[i])
        used = pop_size

        # Estimate ideal and nadir
        ideal = F.min(axis=0)
        nadir = F.max(axis=0)
        f_range = nadir - ideal
        f_range[f_range == 0] = 1.0

        # Archive
        archive_X = X.copy()
        archive_F = F.copy()

        # Phase 1 loop: generate offspring guided by reference directions
        sigma = 0.15
        while used < budget:
            if time.time() > deadline:
                break
            # Pick a random parent and a reference direction
            p_idx = rng.integers(pop_size)
            w_idx = rng.integers(n_dirs)
            w = weights[w_idx]

            # Tchebycheff-based mutation: move toward the region indicated by w
            child = self._tchebycheff_offspring(X[p_idx], w, ideal, nadir, lo, hi, rng, sigma)
            fc = problem(child)
            used += 1

            # Update population: replace the point with worst Tchebycheff value for this direction
            # Compute Tchebycheff values for all points with respect to w
            norm_F = (F - ideal) / f_range
            tcheb_vals = np.max(w * norm_F, axis=1)
            worst_idx = np.argmax(tcheb_vals)

            # Only replace if child is better or similar
            if np.max(w * (fc - ideal) / f_range) <= tcheb_vals[worst_idx] + 1e-9:
                X[worst_idx] = child
                F[worst_idx] = fc

            # Update archive
            archive_X, archive_F = self._update_archive(child, fc, archive_X, archive_F, max_archive=50)

            # Adaptive sigma: reduce if stagnation
            if used % 30 == 0:
                sigma *= 0.95

        # Phase 2: Local scalarized refinement from best archive points
        if used < budget and len(archive_X) > 0:
            # Sort archive by hypervolume-like score (sum of normalized objectives)
            norm_AF = (archive_F - ideal) / f_range
            af_scores = norm_AF.sum(axis=1)
            order = np.argsort(af_scores)
            top_k = min(10, len(order))
            top_indices = order[:top_k]

            for idx in top_indices:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                # Local refinement: try small perturbations around this point
                base = archive_X[idx].copy()
                base_f = archive_F[idx].copy()
                best_local = base_f.copy()
                best_local_x = base.copy()

                # Try a few random perturbations
                for trial in range(5):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    pert = base + 0.02 * (hi - lo) * rng.standard_normal(dim)
                    pert = np.clip(pert, lo, hi)
                    pf = problem(pert)
                    used += 1
                    if np.sum(pf) < np.sum(best_local):
                        best_local = pf
                        best_local_x = pert

                # Update archive with best local result
                archive_X, archive_F = self._update_archive(best_local_x, best_local, archive_X, archive_F, max_archive=50)

        return archive_X
# EVOLVE-BLOCK-END
