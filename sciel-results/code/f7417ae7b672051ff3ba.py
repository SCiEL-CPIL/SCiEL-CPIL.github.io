"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using NSGA-II-style selection with tournament
    selection, crowding distance for diversity, adaptive mutation with jumps,
    and hypervolume-guided refinement.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation
        ref_point = np.array([1.1, 1.3, 1.1])

        # Phase 1: Initial random population
        init_pop = min(30, budget)
        X_init = lo + (hi - lo) * np.random.random((init_pop, dim))
        F_init = np.array([problem(x) for x in X_init])
        evals_used = init_pop

        # Store all evaluated points
        all_X = [x.copy() for x in X_init]
        all_F = [f.copy() for f in F_init]

        # Helper: compute non-dominated front indices using vectorized approach
        def get_non_dominated_indices(F_list):
            n = len(F_list)
            if n == 0:
                return []
            F_arr = np.array(F_list)
            # leq[j,i] = True if F[j] <= F[i] component-wise
            leq = np.all(F_arr[:, None, :] <= F_arr[None, :, :], axis=2)
            # lt[j,i] = True if F[j] strictly dominates F[i]
            lt = leq & np.any(F_arr[:, None, :] < F_arr[None, :, :], axis=2)
            # dominated[i] = True if any j strictly dominates i
            dominated = np.any(lt, axis=0)
            nd_indices = np.where(~dominated)[0].tolist()
            return nd_indices

        # Helper: compute crowding distance for a set of points
        def crowding_distance(F_list):
            n = len(F_list)
            if n <= 2:
                return [float('inf')] * n
            F_arr = np.array(F_list)
            cd = np.zeros(n)
            for m in range(n_obj):
                sorted_idx = np.argsort(F_arr[:, m])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                for i in range(1, n - 1):
                    denom = F_arr[sorted_idx[i + 1], m] - F_arr[sorted_idx[i - 1], m]
                    if denom > 1e-10:
                        cd[sorted_idx[i]] += 1.0 / denom
            return cd

        # Helper: compute hypervolume contribution of a single point
        def hv_contribution(point_F, nd_F_arr):
            """
            Approximate the hypervolume contribution of point_F given the
            current non-dominated front nd_F_arr. Returns a scalar estimate.
            Only points strictly better than ref_point in every objective count.
            """
            # Filter points that are strictly better than ref_point
            valid_mask = np.all(nd_F_arr < ref_point, axis=1)
            if not np.any(valid_mask):
                return 0.0
            valid_F = nd_F_arr[valid_mask]
            
            # Check if point_F is valid
            if not np.all(point_F < ref_point):
                return 0.0
            
            # Simple approximation: compute the volume of the box that point_F
            # would add if it were on the front. Use a Monte Carlo-like estimate
            # based on the spacing of neighboring points.
            # For simplicity, estimate contribution as the product of gaps to
            # nearest neighbors in each objective.
            n_valid = len(valid_F)
            if n_valid == 0:
                # Point is the only valid one; contribution is the full box
                return np.prod(ref_point - point_F)
            
            contributions = np.zeros(n_obj)
            for m in range(n_obj):
                # Find the nearest neighbor in objective m
                diffs = np.abs(valid_F[:, m] - point_F[m])
                min_diff = np.min(diffs)
                # Also consider distance to reference point
                dist_to_ref = ref_point[m] - point_F[m]
                # Contribution in this dimension is roughly min(min_diff, dist_to_ref)
                contributions[m] = min(min_diff, dist_to_ref)
            
            return np.prod(contributions)

        # Phase 2: NSGA-II-style evolutionary refinement with jumps
        remaining = budget - evals_used
        mutation_scale_start = 0.2
        mutation_scale_end = 0.015
        tournament_size = 3
        jump_probability = 0.2
        jump_scale = 0.4
        random_inject_every = 15

        while evals_used < budget:
            if time.time() > deadline:
                break

            # Get non-dominated front from all evaluated points
            nd_idx = get_non_dominated_indices(all_F)

            if len(nd_idx) == 0:
                break

            # Compute crowding distance for the non-dominated front
            nd_F = [all_F[i] for i in nd_idx]
            cd = crowding_distance(nd_F)

            # Tournament selection: pick tournament_size random non-dominated points,
            # choose the one with the largest crowding distance
            if len(nd_idx) > 1:
                candidates = np.random.choice(len(nd_idx), size=min(tournament_size, len(nd_idx)), replace=False)
                best_candidate = candidates[np.argmax([cd[c] for c in candidates])]
                parent_local = nd_idx[best_candidate]
            else:
                parent_local = nd_idx[0]

            parent_X = all_X[parent_local]

            # Adaptive mutation step size
            progress = 1.0 - (budget - evals_used) / max(1, remaining)
            scale = mutation_scale_start + (mutation_scale_end - mutation_scale_start) * progress
            scale = max(scale, mutation_scale_end)

            # Decide mutation type
            r = np.random.random()
            if r < jump_probability:
                # Global jump: large step to explore distant regions
                mutation = jump_scale * (hi - lo) * np.random.standard_normal(dim)
                child_X = np.clip(parent_X + mutation, lo, hi)
            elif evals_used % random_inject_every == 0 and np.random.random() < 0.3:
                # Random injection: fully random point to maintain diversity
                child_X = lo + (hi - lo) * np.random.random(dim)
            else:
                # Standard Gaussian mutation
                mutation = scale * (hi - lo) * np.random.standard_normal(dim)
                child_X = np.clip(parent_X + mutation, lo, hi)

            # Evaluate child
            child_F = problem(child_X)
            evals_used += 1

            # Add to archive
            all_X.append(child_X)
            all_F.append(child_F)

        return None
# EVOLVE-BLOCK-END
