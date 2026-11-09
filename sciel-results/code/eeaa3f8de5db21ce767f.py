"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-based multi-objective search with local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_contribution(self, F, ref):
        """Compute hypervolume contribution of each point (2D case)."""
        n = len(F)
        if n == 0:
            return np.array([])
        # Filter points that dominate reference
        valid = np.all(F < ref, axis=1)
        if not np.any(valid):
            return np.zeros(n)
        
        # For 2D, hypervolume contribution can be computed by sorting
        # and calculating the area each point uniquely contributes
        contribs = np.zeros(n)
        
        # Sort by first objective
        order = np.argsort(F[:, 0])
        
        # For each point, its contribution is the rectangle it uniquely covers
        # In 2D: point i contributes (f0[i+1] - f0[i]) * (ref[1] - f1[i]) if valid
        # But we need to handle the full hypervolume properly
        
        # Better approach: compute total HV of all points, then HV without each point
        # This is O(n^2) but n is small (archive size)
        
        def _hv_2d(points, ref):
            """Compute 2D hypervolume for a set of points with reference."""
            if len(points) == 0:
                return 0.0
            # Filter valid points
            valid = np.all(points < ref, axis=1)
            if not np.any(valid):
                return 0.0
            pts = points[valid]
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            hv = 0.0
            prev_f1 = ref[1]
            for i in range(len(pts) - 1, -1, -1):
                if pts[i, 1] < prev_f1:
                    if i == 0:
                        f0_next = ref[0]
                    else:
                        f0_next = pts[i - 1, 0]
                    hv += (f0_next - pts[i, 0]) * (prev_f1 - pts[i, 1])
                    prev_f1 = pts[i, 1]
            return hv

        total_hv = _hv_2d(F, ref)
        
        for i in range(n):
            if not valid[i]:
                contribs[i] = 0.0
                continue
            # Remove point i and compute HV
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            hv_without = _hv_2d(F[mask], ref)
            contribs[i] = total_hv - hv_without
        
        return contribs

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([3000.0, 0.06])

        # Archive size
        archive_size = 30

        # Latin hypercube-style initial sampling
        init_size = min(archive_size, self.budget)
        X = np.zeros((init_size, dim))
        for d in range(dim):
            perm = np.random.permutation(init_size)
            for i in range(init_size):
                X[perm[i], d] = lo[d] + (hi[d] - lo[d]) * (i + np.random.random()) / init_size

        F = np.array([problem(x) for x in X])
        used = init_size

        # Build archive: keep only non-dominated points, limited to archive_size
        def _build_archive(X, F, max_size):
            """Build a hypervolume-maximizing archive."""
            if len(X) == 0:
                return X, F
            
            # Get non-dominated points
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
            
            nd_idx = [i for i in range(n) if not dominated[i]]
            if len(nd_idx) <= max_size:
                return X[nd_idx], F[nd_idx]
            
            # Select top max_size by hypervolume contribution
            X_nd = X[nd_idx]
            F_nd = F[nd_idx]
            
            # Greedy selection by HV contribution
            selected = []
            remaining = list(range(len(F_nd)))
            
            while len(selected) < max_size and remaining:
                # Compute HV contribution of each remaining point given current selected
                if not selected:
                    # First point: just use its individual contribution
                    contribs = self._hv_contribution(F_nd[remaining], ref)
                else:
                    # Compute contribution relative to already selected
                    F_current = np.vstack([F_nd[selected], F_nd[remaining]])
                    contribs_all = self._hv_contribution(F_current, ref)
                    contribs = contribs_all[len(selected):]
                
                best_local = np.argmax(contribs)
                best_global = remaining[best_local]
                selected.append(best_global)
                remaining.remove(best_global)
            
            return X_nd[selected], F_nd[selected]

        X, F = _build_archive(X, F, archive_size)

        # Main search loop
        range_ = hi - lo

        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Select parent: prefer points with high HV contribution
            if len(F) > 0:
                contribs = self._hv_contribution(F, ref)
                # Add small noise for diversity
                probs = contribs + 1e-6
                probs = probs / (probs.sum() + 1e-12)
                parent_idx = np.random.choice(len(F), p=probs)
                parent = X[parent_idx]
            else:
                parent = np.random.uniform(lo, hi)

            # Mutation: adaptive step size
            spread = np.std(X, axis=0) if len(X) > 1 else range_ * 0.5
            base_scale = 0.2
            scale = base_scale * (1.0 + 0.5 * (spread / (range_ + 1e-12)).mean())
            mutation = np.random.standard_normal(dim) * scale * range_
            child = np.clip(parent + mutation, lo, hi)

            fc = problem(child)
            used += 1

            # Try to add to archive
            # Check if child is dominated by any archive point
            dominated_by_archive = False
            if len(F) > 0:
                for i in range(len(F)):
                    if np.all(F[i] <= fc) and np.any(F[i] < fc):
                        dominated_by_archive = True
                        break
            
            if not dominated_by_archive:
                # Child is not dominated; add it and remove dominated archive points
                X_new = np.vstack([X, child.reshape(1, -1)])
                F_new = np.vstack([F, fc.reshape(1, -1)])
                
                # Remove points dominated by child
                keep = []
                for i in range(len(F_new)):
                    if i == len(F):  # The new point
                        keep.append(i)
                        continue
                    if np.all(fc <= F_new[i]) and np.any(fc < F_new[i]):
                        continue  # Child dominates this point
                    keep.append(i)
                
                X = X_new[keep]
                F = F_new[keep]
                
                # If archive too large, trim by HV contribution
                if len(F) > archive_size:
                    X, F = _build_archive(X, F, archive_size)

        # Local search polish on top HV contributors
        if time.time() < deadline - 5:
            if len(F) > 0:
                contribs = self._hv_contribution(F, ref)
                # Polish top 5 contributors
                top_k = min(5, len(F))
                top_indices = np.argsort(contribs)[::-1][:top_k]
                
                for idx in top_indices:
                    if time.time() > deadline:
                        break
                    if used >= self.budget:
                        break
                    
                    # Small perturbations around this point
                    x0 = X[idx].copy()
                    f0 = F[idx].copy()
                    
                    for step in range(3):
                        if used >= self.budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        # Try a small random perturbation
                        small_step = 0.05 * range_
                        x_trial = x0 + np.random.standard_normal(dim) * small_step
                        x_trial = np.clip(x_trial, lo, hi)
                        
                        f_trial = problem(x_trial)
                        used += 1
                        
                        # Check if this improves the point
                        if np.all(f_trial <= f0):
                            if np.any(f_trial < f0):
                                x0 = x_trial
                                f0 = f_trial
                    
                    # Update archive with improved point if better
                    if np.all(f0 <= F[idx]) and np.any(f0 < F[idx]):
                        X[idx] = x0
                        F[idx] = f0

        return X
# EVOLVE-BLOCK-END
