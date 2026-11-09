"""Multi-objective optimizer for RE21 using hypervolume-based selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using HypE-style marginal hypervolume selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref_point = np.array([3000.0, 0.06])
        rng = np.random.default_rng(42)

        # Archive size limit
        max_archive = 100
        initial_pop = 20

        # Stratified initialization for better spread
        X_archive = self._stratified_sample(rng, lo, hi, initial_pop, self.dim)
        F_archive = np.array([problem(x) for x in X_archive], dtype=np.float64)
        used = initial_pop

        # Filter to non-dominated
        X_archive, F_archive = self._prune_archive(X_archive, F_archive, ref_point, max_archive)

        # Adaptive sigma: start wide, decay over time
        base_sigma = 0.15
        sigma_decay = 0.98

        while used < self.budget:
            if time.time() > deadline - 5:
                break

            # Generate offspring
            n_offspring = 20
            offspring_X = np.empty((n_offspring, self.dim))
            offspring_F = np.empty((n_offspring, self.n_obj))

            for i in range(n_offspring):
                if time.time() > deadline - 5:
                    break
                if used >= self.budget:
                    break

                # Mix of exploration and exploitation
                if rng.random() < 0.25:
                    # Exploration: random point
                    child = lo + (hi - lo) * rng.random(self.dim)
                else:
                    # Exploitation: select parent by marginal hypervolume contribution
                    parent_idx = self._select_parent_hype(F_archive, ref_point, rng)
                    parent = X_archive[parent_idx]
                    # Gaussian mutation with adaptive sigma
                    sigma = base_sigma * (hi - lo) * (sigma_decay ** (used / max(1, self.budget)))
                    child = parent + rng.normal(0, 1, self.dim) * sigma
                    child = np.clip(child, lo, hi)

                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1

            # Combine
            combined_X = np.vstack([X_archive, offspring_X])
            combined_F = np.vstack([F_archive, offspring_F])

            # Prune to non-dominated and limit size
            X_archive, F_archive = self._prune_archive(combined_X, combined_F, ref_point, max_archive)

        return X_archive

    def _stratified_sample(self, rng, lo, hi, n, dim):
        """Generate stratified samples for better initial spread."""
        points = np.empty((n, dim))
        for d in range(dim):
            # Divide range into n bins and sample one from each
            bins = np.linspace(lo[d], hi[d], n + 1)
            for i in range(n):
                points[i, d] = rng.uniform(bins[i], bins[i + 1])
        return points

    def _select_parent_hype(self, F, ref_point, rng):
        """Select parent based on marginal hypervolume contribution (HypE-style)."""
        n = len(F)
        if n == 0:
            return 0
        if n == 1:
            return 0

        # Compute marginal hypervolume contribution for each point
        # For 2D: contribution of point i is the area it uniquely covers
        # Sort by first objective
        sort_idx = np.argsort(F[:, 0])
        F_sorted = F[sort_idx]

        contributions = np.zeros(n)
        for i in range(n):
            # Find position in sorted order
            pos = np.where(sort_idx == i)[0][0]
            if pos == 0:
                # Leftmost point: covers from its f1 to next point's f1
                next_f1 = F_sorted[pos + 1, 0] if pos + 1 < n else ref_point[0]
                w = max(0, next_f1 - F_sorted[pos, 0])
                h = max(0, ref_point[1] - F_sorted[pos, 1])
                contributions[i] = w * h
            elif pos == n - 1:
                # Rightmost point
                prev_f1 = F_sorted[pos - 1, 0]
                w = max(0, ref_point[0] - F_sorted[pos, 0])
                h = max(0, ref_point[1] - F_sorted[pos, 1])
                contributions[i] = w * h
            else:
                # Middle point: only contributes if it's not dominated in the sorted chain
                # Check if f2 is lower than all points to its left (which have lower f1)
                # In a proper Pareto front sorted by f1, f2 should be decreasing
                # Contribution is the unique area it covers
                next_f1 = F_sorted[pos + 1, 0]
                prev_f1 = F_sorted[pos - 1, 0]
                # The point contributes if its f2 is lower than the previous point's f2
                # and higher than the next point's f2 (in the Pareto sense)
                w = max(0, next_f1 - prev_f1)
                # Only count the portion not covered by neighbors
                h_above_prev = max(0, F_sorted[pos - 1, 1] - F_sorted[pos, 1])
                h_below_next = max(0, F_sorted[pos + 1, 1] - F_sorted[pos, 1])
                # Unique contribution is the minimum of these
                h = min(h_above_prev, h_below_next)
                contributions[i] = w * h

        # Normalize to probabilities
        total = contributions.sum()
        if total <= 0:
            return rng.integers(n)
        probs = contributions / total
        return rng.choice(n, p=probs)

    def _prune_archive(self, X, F, ref_point, max_size):
        """Prune archive to non-dominated points and limit size by hypervolume contribution."""
        n = len(X)
        if n == 0:
            return X, F

        # 1. Remove dominated points
        non_dom_mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not non_dom_mask[i]:
                continue
            for j in range(n):
                if i == j or not non_dom_mask[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    non_dom_mask[i] = False
                    break

        X_nd = X[non_dom_mask]
        F_nd = F[non_dom_mask]

        if len(X_nd) <= max_size:
            return X_nd, F_nd

        # 2. If too many, prune based on hypervolume contribution
        # Iteratively remove point with smallest marginal contribution
        keep_indices = np.arange(len(F_nd))
        current_F = F_nd.copy()
        current_X = X_nd.copy()

        while len(keep_indices) > max_size:
            # Compute marginal contributions
            n_curr = len(current_F)
            sort_idx = np.argsort(current_F[:, 0])
            F_sorted = current_F[sort_idx]

            contributions = np.zeros(n_curr)
            for i in range(n_curr):
                pos = np.where(sort_idx == i)[0][0]
                if pos == 0:
                    next_f1 = F_sorted[pos + 1, 0] if pos + 1 < n_curr else ref_point[0]
                    w = max(0, next_f1 - F_sorted[pos, 0])
                    h = max(0, ref_point[1] - F_sorted[pos, 1])
                    contributions[i] = w * h
                elif pos == n_curr - 1:
                    w = max(0, ref_point[0] - F_sorted[pos, 0])
                    h = max(0, ref_point[1] - F_sorted[pos, 1])
                    contributions[i] = w * h
                else:
                    next_f1 = F_sorted[pos + 1, 0]
                    prev_f1 = F_sorted[pos - 1, 0]
                    w = max(0, next_f1 - prev_f1)
                    h_above_prev = max(0, F_sorted[pos - 1, 1] - F_sorted[pos, 1])
                    h_below_next = max(0, F_sorted[pos + 1, 1] - F_sorted[pos, 1])
                    h = min(h_above_prev, h_below_next)
                    contributions[i] = w * h

            # Remove point with smallest contribution
            min_idx = np.argmin(contributions)
            keep_indices = np.delete(keep_indices, min_idx)
            current_F = current_F[keep_indices]
            current_X = current_X[keep_indices]

        return current_X, current_F
# EVOLVE-BLOCK-END
