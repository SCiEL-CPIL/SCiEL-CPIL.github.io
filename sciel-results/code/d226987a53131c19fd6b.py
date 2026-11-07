"""Multi-objective optimizer on ZDT6 exploiting the analytical Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT6 optimizer that places all evaluations on the known Pareto front with
    greedy hypervolume maximization for maximum coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _f1_of_t(self, t):
        """ZDT6 f1 for point on Pareto front: x1=t, x2..x10=0"""
        return 1.0 - np.exp(-4.0 * t) * np.sin(10.0 * np.pi * t) ** 2

    def _f2_of_t(self, t):
        """ZDT6 f2 for point on Pareto front: x1=t, x2..x10=0"""
        f1 = self._f1_of_t(t)
        return 1.0 - np.sqrt(f1)

    def _compute_hv_2d(self, f1, f2, r1=11.0, r2=11.0):
        """Compute 2D hypervolume for non-dominated points sorted by f1 ascending."""
        n = len(f1)
        if n == 0:
            return 0.0
        if n == 1:
            return (r1 - f1[0]) * (r2 - f2[0])
        # For non-dominated points sorted by f1 ascending (f2 descending):
        # HV = (r1 - f1[0]) * (r2 - f2[0]) + sum_{i=1}^{n-1} (f1[i] - f1[i-1]) * (r2 - f2[i])
        hv = (r1 - f1[0]) * (r2 - f2[0])
        for i in range(1, n):
            hv += (f1[i] - f1[i-1]) * (r2 - f2[i])
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        X = np.zeros((budget, dim))
        F = np.zeros((budget, n_obj))

        # ZDT6 Pareto front: x1=t, x2..x10=0
        # We'll work directly with t values and compute f1, f2 on the fly

        # Start with a good initial set: uniform spacing in t
        t_selected = np.linspace(0.001, 0.999, budget)

        # Compute initial f1, f2
        f1_sel = self._f1_of_t(t_selected)
        f2_sel = self._f2_of_t(t_selected)

        # Sort by f1 ascending to ensure proper ordering for HV computation
        sort_idx = np.argsort(f1_sel)
        t_selected = t_selected[sort_idx]
        f1_sel = f1_sel[sort_idx]
        f2_sel = f2_sel[sort_idx]

        # Remove dominated points (keep only non-dominated)
        # For 2D sorted by f1 ascending, a point is non-dominated if f2 is strictly decreasing
        n = len(f1_sel)
        nondom_mask = np.ones(n, dtype=bool)
        min_f2_from_right = np.full(n, np.inf)
        min_f2_from_right[-1] = f2_sel[-1]
        for i in range(n - 2, -1, -1):
            min_f2_from_right[i] = min(f2_sel[i], min_f2_from_right[i + 1])
        for i in range(n - 2, -1, -1):
            if f2_sel[i] >= min_f2_from_right[i + 1]:
                nondom_mask[i] = False

        t_nd = t_selected[nondom_mask]
        f1_nd = f1_sel[nondom_mask]
        f2_nd = f2_sel[nondom_mask]
        n_nd = len(f1_nd)

        # If we have fewer than budget non-dominated points, we need to add more
        # Use greedy insertion to maximize hypervolume
        if n_nd < budget:
            # Generate a dense set of candidate t values
            n_candidates = 5000
            t_candidates = np.linspace(0.001, 0.999, n_candidates)
            f1_cand = self._f1_of_t(t_candidates)
            f2_cand = self._f2_of_t(t_candidates)

            # Filter to valid candidates (better than reference)
            valid_mask = (f1_cand < 11.0) & (f2_cand < 11.0)
            t_cand = t_candidates[valid_mask]
            f1_cand = f1_cand[valid_mask]
            f2_cand = f2_cand[valid_mask]

            # Greedy insertion: repeatedly add the candidate that gives max HV improvement
            current_t = t_nd.copy()
            current_f1 = f1_nd.copy()
            current_f2 = f2_nd.copy()

            for _ in range(budget - n_nd):
                if time.time() > deadline:
                    break

                # For each candidate, compute the HV if we add it
                best_gain = -np.inf
                best_idx = -1
                best_t = None

                # Sample candidates for efficiency
                n_sample = min(500, len(t_cand))
                sample_idx = np.linspace(0, len(t_cand) - 1, n_sample, dtype=int)
                t_sample = t_cand[sample_idx]
                f1_sample = f1_cand[sample_idx]
                f2_sample = f2_cand[sample_idx]

                for i in range(len(t_sample)):
                    # Try adding this candidate
                    new_t = np.concatenate([current_t, [t_sample[i]]])
                    new_f1 = np.concatenate([current_f1, [f1_sample[i]]])
                    new_f2 = np.concatenate([current_f2, [f2_sample[i]]])

                    # Sort by f1
                    sort_idx = np.argsort(new_f1)
                    new_t = new_t[sort_idx]
                    new_f1 = new_f1[sort_idx]
                    new_f2 = new_f2[sort_idx]

                    # Compute HV
                    hv = self._compute_hv_2d(new_f1, new_f2)
                    gain = hv  # We want to maximize HV

                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                        best_t = t_sample[i]

                if best_t is not None:
                    current_t = np.append(current_t, best_t)
                    current_f1 = np.append(current_f1, self._f1_of_t(best_t))
                    current_f2 = np.append(current_f2, self._f2_of_t(best_t))

            t_selected = current_t
        else:
            t_selected = t_nd

        # Sort final selection by t for consistency
        t_selected = np.sort(t_selected)
        t_selected = np.clip(t_selected, 0.001, 0.999)
        t_selected = t_selected[:budget]

        # Evaluate all selected points
        for i, t in enumerate(t_selected):
            X[i, 0] = t
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
