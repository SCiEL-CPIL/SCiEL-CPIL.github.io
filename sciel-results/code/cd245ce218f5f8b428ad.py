"""Multi-objective optimizer for RE21 with hypervolume-driven selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-contribution-based multi-objective optimizer for RE21 (2 objectives, 4 vars)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_contribution_2d(self, point, front, ref):
        """Compute hypervolume contribution of a single 2D point given a non-dominated front.
        
        For 2D minimization, the contribution of point p is:
        (ref[0] - p[0]) * (ref[1] - neighbor_below_p[1])
        where neighbor_below_p is the point with the largest f1 that is still <= p[1],
        or 0 if none exists (meaning p is the best in f1 among those with f2 <= p[1]).
        
        Actually, for 2D: sort front by f0 ascending. The contribution of point i is:
        (ref[0] - f0_i) * (f1_{i-1} - f1_i) where f1_{i-1} is the f1 of the previous point
        (with larger f0), or ref[1] if it's the first.
        Wait, let me think more carefully.
        
        In 2D minimization with reference point ref:
        Sort points by f0 ascending (best first).
        The hypervolume is sum over i of (ref[0] - f0_i) * (f1_prev - f1_i)
        where f1_prev starts at ref[1] and f1_i is the f1 of point i.
        But we need f1 to be non-increasing for the non-dominated front sorted by f0.
        
        Actually for a non-dominated front sorted by f0 ascending, f1 must be descending.
        The contribution of point i (0-indexed) is:
        (ref[0] - f0_i) * (f1_{i-1} - f1_i)
        where f1_{-1} = ref[1].
        
        But if the point is dominated by another in the front, its contribution is 0.
        """
        if len(front) == 0:
            return (ref[0] - point[0]) * (ref[1] - point[1]) if (point[0] < ref[0] and point[1] < ref[1]) else 0.0
        
        # Check if point is dominated by any point in front
        for fp in front:
            if fp[0] <= point[0] and fp[1] <= point[1] and (fp[0] < point[0] or fp[1] < point[1]):
                return 0.0
        
        # Point is not dominated. Find its position in the sorted front.
        # Sort by f0 ascending
        sorted_front = sorted(front, key=lambda x: x[0])
        
        # Find where point would be inserted
        f0 = point[0]
        f1 = point[1]
        
        # Find the index where f0 would be inserted
        idx = 0
        for i, fp in enumerate(sorted_front):
            if fp[0] < f0:
                idx = i + 1
            else:
                break
        
        # The contribution is (ref[0] - f0) * (f1_prev - f1)
        # where f1_prev is the f1 of the point at idx-1 in sorted order, or ref[1] if idx==0
        if idx == 0:
            f1_prev = ref[1]
        else:
            f1_prev = sorted_front[idx - 1][1]
        
        # Also need to make sure the next point doesn't affect this
        # Actually in 2D, the contribution of a non-dominated point is simply:
        # (ref[0] - f0) * (min(f1_prev, f1) - f1) ... no wait.
        
        # Let me reconsider. For 2D non-dominated front sorted by f0 asc (f1 desc):
        # HV = sum_i (ref[0] - f0_i) * (f1_{i-1} - f1_i)
        # where f1_{-1} = ref[1]
        # The contribution of point i is (ref[0] - f0_i) * (f1_{i-1} - f1_i)
        # But only if f1_{i-1} > f1_i (which should hold for non-dominated sorted front)
        
        contrib = (ref[0] - f0) * (f1_prev - f1)
        if contrib < 0:
            contrib = 0.0
        return contrib

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3000.0, 0.06])
        
        # Population size
        pop_size = min(100, max(20, self.budget // 5))
        
        # Initialize population with random points
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points (for recording)
        all_points = list(F)
        
        def get_non_dominated_front(F_arr):
            """Return indices of non-dominated points."""
            n = len(F_arr)
            nd = np.zeros(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    for j in range(n):
                        if i != j:
                            if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                                nd[i] = False
                                break
                    else:
                        nd[i] = True
            return nd
        
        def compute_hv(front, ref_point):
            """Compute hypervolume of a 2D front."""
            if len(front) == 0:
                return 0.0
            # Filter points that are better than reference
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
        
        # Main loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Get non-dominated front
            nd_mask = get_non_dominated_front(F)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) == 0:
                # All dominated, pick random
                nd_indices = np.random.choice(len(F), size=min(5, len(F)), replace=False)
            
            # Compute hypervolume contribution for each non-dominated point
            front_points = [F[i] for i in nd_indices]
            contributions = []
            for i in nd_indices:
                # Remove point i from front, compute HV of remaining
                remaining = [F[j] for j in nd_indices if j != i]
                hv_with = compute_hv(front_points, ref)
                hv_without = compute_hv(remaining, ref)
                contributions.append(hv_with - hv_without)
            
            # Select parent with highest contribution (or random if all zero)
            contributions = np.array(contributions)
            if np.all(contributions <= 0):
                parent_idx = nd_indices[rng.integers(len(nd_indices))]
            else:
                # Softmax selection
                max_c = np.max(contributions)
                weights = np.exp(contributions - max_c)
                weights /= np.sum(weights)
                parent_idx = nd_indices[rng.choice(len(nd_indices), p=weights)]
            
            # Adaptive mutation
            # Estimate step size from current spread
            spread = np.std(F, axis=0)
            step = np.maximum(0.05 * (hi - lo), 0.1 * spread)
            
            # SBX-like crossover with a second parent
            if len(nd_indices) >= 2:
                # Pick second parent
                second_idx = nd_indices[rng.integers(len(nd_indices))]
                if second_idx == parent_idx:
                    second_idx = nd_indices[(nd_indices.tolist().index(parent_idx) + 1) % len(nd_indices)]
                
                x1 = X[parent_idx]
                x2 = X[second_idx]
                
                # SBX crossover
                eta_c = 15.0
                child = np.zeros(self.dim)
                for d in range(self.dim):
                    if x1[d] != x2[d]:
                        if rng.random() < 0.5:
                            beta = 1.0 / (2.0 * rng.random())
                            if beta > 1.0:
                                beta = beta ** (1.0 / (eta_c + 1.0))
                                child[d] = 0.5 * ((1.0 + beta) * x1[d] + (1.0 - beta) * x2[d])
                            else:
                                beta = 1.0 / (2.0 - 2.0 * rng.random())
                                if beta > 1.0:
                                    beta = beta ** (1.0 / (eta_c + 1.0))
                                    child[d] = 0.5 * ((1.0 + beta) * x2[d] + (1.0 - beta) * x1[d])
                                else:
                                    beta = 1.0 / (2.0 - 2.0 * rng.random())
                                    beta = beta ** (1.0 / (eta_c + 1.0))
                                    child[d] = 0.5 * ((1.0 + beta) * x2[d] + (1.0 - beta) * x1[d])
                        else:
                            beta = 1.0 / (2.0 * rng.random())
                            if beta > 1.0:
                                beta = beta ** (1.0 / (eta_c + 1.0))
                                child[d] = 0.5 * ((1.0 + beta) * x2[d] + (1.0 - beta) * x1[d])
                            else:
                                beta = 1.0 / (2.0 - 2.0 * rng.random())
                                if beta > 1.0:
                                    beta = beta ** (1.0 / (eta_c + 1.0))
                                    child[d] = 0.5 * ((1.0 + beta) * x1[d] + (1.0 - beta) * x2[d])
                                else:
                                    beta = 1.0 / (2.0 - 2.0 * rng.random())
                                    beta = beta ** (1.0 / (eta_c + 1.0))
                                    child[d] = 0.5 * ((1.0 + beta) * x1[d] + (1.0 - beta) * x2[d])
                    else:
                        child[d] = x1[d]
                
                # Mutation
                eta_m = 20.0
                for d in range(self.dim):
                    if rng.random() < 0.5:
                        if rng.random() < 0.5:
                            delta1 = (1.0 / (2.0 * rng.random()))
                            if delta1 > 1.0:
                                delta1 = delta1 ** (1.0 / (eta_m + 1.0))
                                child[d] = child[d] * (2.0 - delta1) + x1[d] * (delta1 - 1.0)
                            else:
                                delta2 = 1.0 / (2.0 - 2.0 * rng.random())
                                delta2 = delta2 ** (1.0 / (eta_m + 1.0))
                                child[d] = child[d] * (1.0 + delta2) + x1[d] * (-delta2)
                        else:
                            delta2 = 1.0 / (2.0 * rng.random())
                            if delta2 > 1.0:
                                delta2 = delta2 ** (1.0 / (eta_m + 1.0))
                                child[d] = child[d] * (1.0 + delta2) + x2[d] * (-delta2)
                            else:
                                delta1 = 1.0 / (2.0 - 2.0 * rng.random())
                                delta1 = delta1 ** (1.0 / (eta_m + 1.0))
                                child[d] = child[d] * (2.0 - delta1) + x2[d] * (delta1 - 1.0)
            else:
                # Just mutate the single parent
                child = X[parent_idx].copy()
                for d in range(self.dim):
                    if rng.random() < 0.5:
                        child[d] = child[d] + step[d] * rng.standard_normal()
            
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            all_points.append(fc)
            
            # Replace worst point in population (by HV contribution)
            # Compute contribution of each point in population
            pop_front = get_non_dominated_front(F)
            pop_front_indices = np.where(pop_front)[0]
            
            if len(pop_front_indices) > 0:
                # Find point with lowest contribution
                min_contrib = float('inf')
                min_idx = 0
                for i in pop_front_indices:
                    remaining = [F[j] for j in pop_front_indices if j != i]
                    hv_with = compute_hv([F[j] for j in pop_front_indices], ref)
                    hv_without = compute_hv(remaining, ref)
                    contrib = hv_with - hv_without
                    if contrib < min_contrib:
                        min_contrib = contrib
                        min_idx = i
                X[min_idx] = child
                F[min_idx] = fc
            else:
                # All dominated, replace random
                idx = rng.integers(len(F))
                X[idx] = child
                F[idx] = fc
        
        return X
# EVOLVE-BLOCK-END
