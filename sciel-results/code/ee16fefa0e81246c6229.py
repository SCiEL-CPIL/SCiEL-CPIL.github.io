# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE21 (2 objectives, 4 vars) using a
    hypervolume-guided population-based search with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3000.0, 0.06])

        # Population size
        pop_size = min(80, max(20, self.budget // 4))

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Track all evaluated points
        all_points = list(F)

        def get_non_dominated_front(F_arr):
            """Return indices of non-dominated points (O(n^2) but n is small)."""
            n = len(F_arr)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        def compute_hv_2d(front, ref_point):
            """Compute hypervolume of a 2D front."""
            if len(front) == 0:
                return 0.0
            valid = [f for f in front if f[0] < ref_point[0] and f[1] < ref_point[1]]
            if len(valid) == 0:
                return 0.0
            valid.sort(key=lambda x: x[0])
            hv = 0.0
            prev_f1 = ref_point[1]
            for f in valid:
                if f[1] < prev_f1:
                    hv += (ref_point[0] - f[0]) * (prev_f1 - f[1])
                    prev_f1 = f[1]
            return hv

        def hv_contribution(point, front, ref_point):
            """Compute hypervolume contribution of a single point."""
            if point[0] >= ref_point[0] or point[1] >= ref_point[1]:
                return 0.0
            
            # Check if dominated
            for fp in front:
                if fp[0] <= point[0] and fp[1] <= point[1] and (fp[0] < point[0] or fp[1] < point[1]):
                    return 0.0
            
            # Sort front by f0
            sorted_front = sorted(front, key=lambda x: x[0])
            f0, f1 = point[0], point[1]
            
            # Find insertion position
            idx = 0
            for i, fp in enumerate(sorted_front):
                if fp[0] < f0:
                    idx = i + 1
                else:
                    break
            
            f1_prev = ref_point[1] if idx == 0 else sorted_front[idx - 1][1]
            contrib = (ref_point[0] - f0) * (f1_prev - f1)
            return max(0.0, contrib)

        # Main optimization loop
        step_scale = 1.0
        no_improve_count = 0
        best_hv = 0.0

        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Get non-dominated front
            nd_indices = get_non_dominated_front(F)
            front = [F[i] for i in nd_indices]

            # Compute current HV
            current_hv = compute_hv_2d(front, ref)
            if current_hv > best_hv:
                best_hv = current_hv
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Adaptive step size
            if no_improve_count > 15:
                step_scale = max(step_scale * 0.8, 0.1)
                no_improve_count = 0

            # Select parent based on HV contribution
            if len(nd_indices) > 0:
                contributions = np.array([hv_contribution(F[i], front, ref) for i in nd_indices])
                if np.all(contributions <= 0):
                    parent_idx = nd_indices[rng.integers(len(nd_indices))]
                else:
                    max_c = np.max(contributions)
                    weights = np.exp(contributions - max_c)
                    weights /= np.sum(weights)
                    parent_idx = nd_indices[rng.choice(len(nd_indices), p=weights)]
            else:
                parent_idx = rng.integers(len(F))

            # Generate child via mutation
            child = X[parent_idx].copy()
            spread = np.std(F, axis=0)
            step = np.maximum(0.05 * (hi - lo), 0.1 * spread) * step_scale

            for d in range(self.dim):
                if rng.random() < 0.8:  # 80% mutation rate
                    child[d] += step[d] * rng.standard_normal()

            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            all_points.append(fc)

            # Replace worst point in population
            pop_nd = get_non_dominated_front(F)
            if len(pop_nd) > 0:
                pop_contribs = [hv_contribution(F[i], [F[j] for j in pop_nd], ref) for i in pop_nd]
                min_c = min(pop_contribs)
                replace_idx = pop_nd[np.argmin(pop_contribs)]
            else:
                replace_idx = rng.integers(len(F))

            X[replace_idx] = child
            F[replace_idx] = fc

        return X
# EVOLVE-BLOCK-END
