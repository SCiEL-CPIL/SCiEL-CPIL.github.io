"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, front, ref):
        """Compute exact 2D hypervolume for a list of points (Pareto front)."""
        if len(front) == 0:
            return 0.0
        pts = np.array(front, dtype=float)
        # Filter strictly better than ref
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            return 0.0
        # Sort by obj0 ascending
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        # Keep only non-dominated points (obj1 must be strictly decreasing)
        front = [pts[0]]
        for p in pts[1:]:
            if p[1] < front[-1][1]:
                front.append(p)
        if len(front) == 0:
            return 0.0
        # Compute HV: sum of rectangles
        hv = 0.0
        for i in range(len(front)):
            x_next = front[i + 1][0] if i + 1 < len(front) else ref[0]
            y_curr = front[i][1]
            if y_curr >= ref[1]:
                continue
            width = x_next - front[i][0]
            height = ref[1] - y_curr
            hv += width * height
        return hv

    def _non_dominated_front(self, F):
        """Return indices of non-dominated points in F (2D minimization)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        F = np.array(F, dtype=float)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def _hv_contribution_2d(self, point, front, ref):
        """Compute HV contribution of a single point in a 2D front."""
        if len(front) == 0:
            return 0.0
        # Filter front to only points better than ref
        pts = np.array(front, dtype=float)
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            return 0.0
        p = np.array(point, dtype=float)
        if not np.all(p < ref):
            return 0.0
        
        # Sort by obj0
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        # Build non-dominated front
        nd_front = [pts[0]]
        for pt in pts[1:]:
            if pt[1] < nd_front[-1][1]:
                nd_front.append(pt)
        
        if len(nd_front) == 0:
            return 0.0
        
        # Find where p fits in the sorted order
        # p's contribution is the area it adds when inserted
        # Sort nd_front + [p] by obj0
        all_pts = np.vstack([np.array(nd_front), p.reshape(1, -1)])
        order = np.argsort(all_pts[:, 0])
        all_pts = all_pts[order]
        
        # Build non-dominated front of all_pts
        nd_all = [all_pts[0]]
        for pt in all_pts[1:]:
            if pt[1] < nd_all[-1][1]:
                nd_all.append(pt)
        
        # Compute HV of nd_all and nd_front
        hv_with = 0.0
        for i in range(len(nd_all)):
            x_next = nd_all[i + 1][0] if i + 1 < len(nd_all) else ref[0]
            y_curr = nd_all[i][1]
            if y_curr >= ref[1]:
                continue
            width = x_next - nd_all[i][0]
            height = ref[1] - y_curr
            hv_with += width * height
        
        hv_without = 0.0
        for i in range(len(nd_front)):
            x_next = nd_front[i + 1][0] if i + 1 < len(nd_front) else ref[0]
            y_curr = nd_front[i][1]
            if y_curr >= ref[1]:
                continue
            width = x_next - nd_front[i][0]
            height = ref[1] - y_curr
            hv_without += width * height
        
        return hv_with - hv_without

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        rng = np.random

        # Initial population: diverse random points + structured
        pop_size = min(40, self.budget // 5)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        # Add structured points
        structured = np.array([
            lo + (hi - lo) * [0.0, 0.0, 0.0, 0.0],
            lo + (hi - lo) * [1.0, 1.0, 1.0, 1.0],
            lo + (hi - lo) * [0.5, 0.5, 0.5, 0.5],
            lo + (hi - lo) * [0.1, 0.9, 0.1, 0.9],
            lo + (hi - lo) * [0.9, 0.1, 0.9, 0.1],
            lo + (hi - lo) * [0.2, 0.2, 0.8, 0.8],
            lo + (hi - lo) * [0.8, 0.8, 0.2, 0.2],
            lo + (hi - lo) * [0.3, 0.7, 0.3, 0.7],
            lo + (hi - lo) * [0.7, 0.3, 0.7, 0.3],
        ])
        X = np.vstack([X, structured])
        F = np.array([problem(x) for x in X])
        used = len(X)

        all_X = list(X)
        all_F = list(F)

        # Adaptive parameters
        mut_scale = 0.12

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated front
            front_idx = self._non_dominated_front(all_F)
            if len(front_idx) == 0:
                front_idx = np.array([0])

            front_points = [all_F[i] for i in front_idx]
            
            # Compute HV contributions for each front point
            contributions = np.zeros(len(front_idx))
            for k, idx in enumerate(front_idx):
                contributions[k] = self._hv_contribution_2d(
                    all_F[idx], front_points, ref
                )
            
            # Add small epsilon to avoid zero probabilities
            contributions = contributions + 1e-10
            probs = contributions / contributions.sum()

            # Select parents via weighted sampling
            parent_indices = []
            for _ in range(2):
                if len(front_idx) == 1:
                    parent_indices.append(front_idx[0])
                else:
                    parent_indices.append(front_idx[rng.choice(len(front_idx), p=probs)])

            p1 = all_X[parent_indices[0]].copy()
            p2 = all_X[parent_indices[1]].copy()

            # Crossover
            if np.random.random() < 0.75:
                # Blend crossover with random weight
                alpha = np.random.random()
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = p1.copy()

            # Mutate with adaptive scale
            mutation = np.random.standard_normal(self.dim) * mut_scale * (hi - lo)
            child = child + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            all_X.append(child)
            all_F.append(fc)

            # Adapt mutation scale: decrease over time
            progress = used / self.budget
            mut_scale = 0.12 * (1.0 - 0.7 * progress) + 0.01

            # Periodic random restarts for exploration
            if used % 40 == 0 and used < self.budget - 2:
                for _ in range(2):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    random_child = lo + (hi - lo) * rng.random(self.dim)
                    rc = problem(random_child)
                    used += 1
                    all_X.append(random_child)
                    all_F.append(rc)

        # Final polish: local refinement around best front points
        remaining = self.budget - used
        if remaining > 0:
            front_idx = self._non_dominated_front(all_F)
            if len(front_idx) > 0:
                # Sort front by HV contribution
                front_points = [all_F[i] for i in front_idx]
                contributions = np.zeros(len(front_idx))
                for k, idx in enumerate(front_idx):
                    contributions[k] = self._hv_contribution_2d(
                        all_F[idx], front_points, ref
                    )
                
                # Take top contributors for local search
                top_k = min(5, len(front_idx))
                top_indices = front_idx[np.argsort(contributions)[-top_k:]]
                
                # Local refinement with small steps
                local_steps = min(remaining, 30)
                for step in range(local_steps):
                    if time.time() > deadline:
                        break
                    parent_idx = top_indices[step % len(top_indices)]
                    parent = all_X[parent_idx].copy()
                    
                    # Small local perturbation
                    small_mut = 0.02 * (hi - lo)
                    child = parent + rng.standard_normal(self.dim) * small_mut
                    child = np.clip(child, lo, hi)
                    
                    fc = problem(child)
                    used += 1
                    all_X.append(child)
                    all_F.append(fc)
                    
                    # Update front if child is non-dominated
                    new_front = self._non_dominated_front(all_F)
                    if len(new_front) > 0:
                        new_front_points = [all_F[i] for i in new_front]
                        new_contribs = np.zeros(len(new_front))
                        for k, idx in enumerate(new_front):
                            new_contribs[k] = self._hv_contribution_2d(
                                all_F[idx], new_front_points, ref
                            )
                        top_indices = new_front[np.argsort(new_contribs)[-top_k:]]

        return all_X
# EVOLVE-BLOCK-END
