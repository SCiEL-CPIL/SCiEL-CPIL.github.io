# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using batched island-based search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        rng = np.random.default_rng(12345)

        # --- Phase 1: Initialize with a diverse population ---
        n_init = 20
        X = lo + (hi - lo) * rng.random((n_init, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = n_init

        # Track all evaluated solutions
        all_X = [X]
        all_F = [F]

        # Reference point for hypervolume-based selection
        ref = np.array([1705.0, 12.0, 0.3])

        # --- Phase 2: Iterative improvement with hypervolume-guided selection ---
        batch_size = 8
        base_scale = 0.25
        min_scale = 0.03

        while used < budget:
            if time.time() > deadline - 5:
                break

            # Flatten all stored solutions
            flat_X = np.vstack(all_X)
            flat_F = np.vstack(all_F)
            n_total = len(flat_F)

            # Find non-dominated set
            nd_mask = self._find_non_dominated(flat_F)
            nd_X = flat_X[nd_mask]
            nd_F = flat_F[nd_mask]
            n_nd = len(nd_X)

            # Compute hypervolume contribution for each non-dominated point
            # to guide selection toward diverse and high-value regions
            if n_nd > 1:
                hv_contribs = self._compute_hv_contributions(nd_F, ref)
                # Normalize to get selection probabilities
                hv_contribs = np.maximum(hv_contribs, 1e-10)
                probs = hv_contribs / hv_contribs.sum()
            elif n_nd == 1:
                probs = np.array([1.0])
            else:
                probs = np.ones(n_total) / n_total

            # Adaptive mutation scale
            progress = used / max(1, budget)
            scale = base_scale - (base_scale - min_scale) * progress

            # Generate children
            children_X = []
            children_F = []

            for _ in range(batch_size):
                if used >= budget:
                    break
                if time.time() > deadline - 2:
                    break

                # Select parent using hypervolume-guided sampling
                if n_nd > 0:
                    parent_idx = rng.choice(n_nd, p=probs)
                    parent = nd_X[parent_idx]
                else:
                    parent = flat_X[rng.integers(0, n_total)]

                # Gaussian perturbation with adaptive scale
                child = parent + scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1
                children_X.append(child)
                children_F.append(fc)

            if children_X:
                children_X = np.array(children_X)
                children_F = np.array(children_F)
                all_X.append(children_X)
                all_F.append(children_F)

        # --- Final: collect all non-dominated solutions ---
        final_X = np.vstack(all_X)
        final_F = np.vstack(all_F)
        nd_mask = self._find_non_dominated(final_F)
        return final_X[nd_mask]

    def _find_non_dominated(self, F):
        """Find indices of non-dominated points in F (minimization)."""
        n = len(F)
        nd_mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd_mask[i]:
                continue
            for j in range(n):
                if i == j or not nd_mask[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd_mask[i] = False
                    break
        return nd_mask

    def _compute_hv_contributions(self, F, ref):
        """Compute hypervolume contribution of each point in a non-dominated set.
        
        Uses an incremental approach: for each point, compute the HV of the set
        with that point removed, and the contribution is HV_full - HV_without.
        For efficiency with small sets, we use a direct computation.
        """
        n = len(F)
        if n == 0:
            return np.array([])
        if n == 1:
            # Single point contribution
            vols = np.maximum(ref - F[0], 0)
            return np.array([np.prod(vols)])

        # Compute total HV
        total_hv = self._compute_hv(F, ref)

        # Compute contribution of each point
        contribs = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            if np.sum(mask) == 0:
                contribs[i] = total_hv
            else:
                hv_without = self._compute_hv(F[mask], ref)
                contribs[i] = total_hv - hv_without

        return contribs

    def _compute_hv(self, F, ref):
        """Compute hypervolume of a set of points with given reference point.
        
        For 3 objectives, we use a direct O(n^2) approach.
        """
        n = len(F)
        if n == 0:
            return 0.0
        if n == 1:
            vols = np.maximum(ref - F[0], 0)
            return float(np.prod(vols))

        # Sort by first objective
        order = np.argsort(F[:, 0])
        F_sorted = F[order]

        # For each point, compute the volume it uniquely contributes
        # A point contributes the box from its position to the reference,
        # minus any boxes already covered by points with better (smaller) first objective
        total_hv = 0.0
        for i in range(n):
            fi = F_sorted[i]
            if np.any(fi >= ref):
                continue
            # Box dimensions from fi to ref
            w1 = ref[0] - fi[0]
            w2 = ref[1] - fi[1]
            w3 = ref[2] - fi[2]
            if w1 <= 0 or w2 <= 0 or w3 <= 0:
                continue
            # Subtract contributions from points j < i that dominate in obj 1
            # and overlap in obj 2, 3
            overlap = 0.0
            for j in range(i):
                fj = F_sorted[j]
                if fj[0] >= fi[0]:
                    break
                # Overlap in dimensions 2 and 3
                o2 = min(ref[1], fi[1]) - max(fj[1], 0)
                o3 = min(ref[2], fi[2]) - max(fj[2], 0)
                if o2 > 0 and o3 > 0:
                    # The overlap in dim 1 is from fj[0] to fi[0]
                    o1 = fi[0] - fj[0]
                    if o1 > 0:
                        overlap += o1 * o2 * o3
            total_hv += w1 * w2 * w3 - overlap

        return max(0.0, total_hv)

# EVOLVE-BLOCK-END
