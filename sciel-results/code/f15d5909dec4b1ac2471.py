import numpy as np
import time

class Optimizer:
    """
    Direct Hierarchical Multi-Objective Optimizer with Adaptive Local Search.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        bounds_range = hi - lo

        all_X = []
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_X.append(x)
            all_F.append(f)
            eval_count += 1
            return f

        # Helper: Check if point A dominates point B (minimization)
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Helper: Get non-dominated indices from current lists
        def get_non_dominated_indices():
            if not all_F:
                return []
            F = np.array(all_F)
            n = len(F)
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if not nd_mask[j]:
                        continue
                    if dominates(F[j], F[i]):
                        nd_mask[i] = False
                        break
            return np.where(nd_mask)[0]

        # Helper: Calculate Hypervolume Contribution in 2D
        # Points must be non-dominated and strictly better than ref
        def hv_contribution_2d(point, nd_points, ref):
            # nd_points: list of tuples (f1, f2)
            # ref: tuple (r1, r2)
            # We assume nd_points are sorted by f1 ascending
            # Contribution is the area of the rectangle defined by point and the "corner"
            # formed by the next point with higher f1 and lower f2 (or ref).
            
            p1, p2 = point
            r1, r2 = ref
            
            # If point is not better than ref, contribution is 0
            if p1 >= r1 or p2 >= r2:
                return 0.0
            
            # Find the next point in the front with f1 > p1
            # Since we are in 2D and non-dominated, if f1 increases, f2 must decrease.
            # We iterate through the sorted list.
            # The "width" of the contribution is determined by the next point's f1 (or ref f1 if none).
            # The "height" is determined by the current point's f2 (or ref f2 if it's the last).
            
            # Actually, simpler logic for 2D HV contribution of a single point to the set:
            # The contribution is (next_f1 - current_f1) * (current_f2 - ref_f2) ? No.
            # Let's use the standard definition.
            # Hypervolume of a set is the volume of the union of boxes [f_i, ref].
            # In 2D, if sorted by f1 ascending (f2 descending):
            # HV = sum_{i} (f_{i+1,1} - f_{i,1}) * (ref_2 - f_{i,2}) where f_{n+1,1} = ref_1.
            # Wait, that's if we sum over the "steps".
            # The contribution of point i is the part of the HV that disappears if we remove point i.
            # If we remove point i, the box [f_i, ref] loses the area that is not covered by other boxes.
            # In 2D, if points are sorted by f1 asc:
            # The area contributed by point i is (f_{i+1,1} - f_{i,1}) * (f_{i,2} - ref_2) ? No.
            # Let's visualize.
            # Points: A(1, 4), B(2, 3), C(3, 2). Ref(5, 5).
            # Box A: [1,5]x[4,5] -> Area 4*1=4. But part is covered by B?
            # B: [2,5]x[3,5] -> Area 3*2=6.
            # C: [3,5]x[2,5] -> Area 2*3=6.
            # Union:
            # x in [1,2]: y in [4,5] -> 1*1 = 1
            # x in [2,3]: y in [3,5] -> 1*2 = 2
            # x in [3,5]: y in [2,5] -> 2*3 = 6
            # Total HV = 1 + 2 + 6 = 9.
            
            # Contribution of A: If removed, we lose x in [1,2] y in [4,5]. Area = 1.
            # Formula: (f_{i+1,1} - f_{i,1}) * (f_{i,2} - f_{i+1,2})? No.
            # For A (i=0): next is B. Width = 2-1=1. Height = 5-4=1? No, height is ref_2 - f_i_2 = 5-4=1.
            # So Contrib(A) = (f_{next,1} - f_i,1) * (ref_2 - f_i,2) ?
            # Let's check B. Next is C. Width = 3-2=1. Height = 5-3=2. Contrib(B) = 2.
            # Let's check C. Next is Ref. Width = 5-3=2. Height = 5-2=3. Contrib(C) = 6.
            # Sum = 1+2+6=9. Correct.
            
            # So for a point P_i at index i in sorted list (by f1 asc):
            # next_f1 = f_{i+1,1} if i < n-1 else ref_1
            # contrib = (next_f1 - p1) * (ref_2 - p2)
            
            # We need to find the index of this point in the sorted list.
            # Since we are calling this for selection, we can pass the sorted list and the index.
            # But here we just have the point. We should pass the sorted list and the index.
            # Let's refactor to accept index.
            
            # For now, this function is a placeholder for the logic, 
            # but in the main loop we will compute contributions for all ND points at once.
            return 0.0

        # Phase 1: Reference-point-aligned deterministic grid initialization
        # The reference point is [3000.0, 0.06]. Objective 1 dominates HV contribution.
        # We want to explore the space to find points with low f1 and low f2.
        # Since we don't know the mapping, we use a stratified grid in the input space.
        # We allocate more budget to initialization to ensure a good initial front.
        
        n_global = 80
        
        # Create a stratified grid in 4D
        # 4^4 = 256 points is too many if budget is small, but 200 is enough.
        # Let's use 3^4 = 81 points for a dense grid.
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        x = np.array([
                            lo[0] + bounds_range[0]*(i+0.5)/3.0,
                            lo[1] + bounds_range[1]*(j+0.5)/3.0,
                            lo[2] + bounds_range[2]*(k+0.5)/3.0,
                            lo[3] + bounds_range[3]*(l+0.5)/3.0
                        ])
                        grid_points.append(x)
        
        # Add some random points to fill the rest of the initial budget
        n_random = n_global - len(grid_points)
        if n_random > 0:
            random_points = lo + bounds_range * np.random.random((n_random, self.dim))
            initial_points = np.vstack([grid_points, random_points])
        else:
            initial_points = np.array(grid_points)
        
        for x in initial_points:
            if time.time() > deadline:
                break
            if eval_count >= self.budget:
                break
            evaluate(x)

        if eval_count >= self.budget or not all_F:
            return all_X

        # Phase 2: Hypervolume-guided Adaptive Differential Evolution
        # Use DE to explore the neighborhood of the current Pareto front.
        # This allows for better exploration than simple random walks.
        
        # Reference point for HV calculation
        ref_point = np.array([3000.0, 0.06])
        
        # Dynamically shrinking mutation factor
        F = 0.8
        CR = 0.9
        
        # Number of offspring to generate per round
        # We want to use the remaining budget efficiently.
        # Let's generate a population of size N_DE
        N_DE = 20
        
        while eval_count < self.budget and time.time() < deadline:
            nd_indices = get_non_dominated_indices()
            if len(nd_indices) == 0:
                break
            
            nd_indices = np.array(nd_indices)
            nd_X = np.array(all_X)[nd_indices]
            nd_F = np.array(all_F)[nd_indices]
            
            # Filter points strictly better than ref
            valid_mask = (nd_F[:, 0] < ref_point[0]) & (nd_F[:, 1] < ref_point[1])
            valid_nd_X = nd_X[valid_mask]
            valid_nd_F = nd_F[valid_mask]
            
            if len(valid_nd_F) == 0:
                # Fallback to random search if no valid points
                x = lo + bounds_range * np.random.random(self.dim)
                evaluate(x)
                continue
            
            # Select parents based on hypervolume contribution
            sort_idx = np.argsort(valid_nd_F[:, 0])
            sorted_F = valid_nd_F[sort_idx]
            sorted_X = valid_nd_X[sort_idx]
            
            n_valid = len(sorted_F)
            contributions = np.zeros(n_valid)
            
            for i in range(n_valid):
                curr_f1 = sorted_F[i, 0]
                curr_f2 = sorted_F[i, 1]
                
                if i < n_valid - 1:
                    next_f1 = sorted_F[i+1, 0]
                else:
                    next_f1 = ref_point[0]
                
                width = next_f1 - curr_f1
                height = ref_point[1] - curr_f2
                contributions[i] = width * height
            
            # If no contributions, just pick best sum
            if np.sum(contributions) < 1e-10:
                sums = np.sum(valid_nd_F, axis=1)
                contrib_sort_idx = np.argsort(sums)
            else:
                contrib_sort_idx = np.argsort(contributions)[::-1]
            
            # Select top parents
            n_parents = min(n_valid, 5)
            parent_indices = contrib_sort_idx[:n_parents]
            
            # Generate offspring
            n_offspring = min(N_DE, self.budget - eval_count)
            
            for _ in range(n_offspring):
                if eval_count >= self.budget or time.time() > deadline:
                    break
                
                # Select 3 distinct parents
                if n_parents >= 3:
                    p1, p2, p3 = np.random.choice(n_parents, 3, replace=False)
                    x1 = sorted_X[p1]
                    x2 = sorted_X[p2]
                    x3 = sorted_X[p3]
                else:
                    # If not enough parents, use random points from the whole set
                    idx1 = np.random.randint(0, len(all_X))
                    idx2 = np.random.randint(0, len(all_X))
                    idx3 = np.random.randint(0, len(all_X))
                    x1 = np.array(all_X[idx1])
                    x2 = np.array(all_X[idx2])
                    x3 = np.array(all_X[idx3])
                
                # DE mutation
                v = x1 + F * (x2 - x3)
                
                # Crossover
                trial = v.copy()
                r = np.random.random(self.dim)
                j_rand = np.random.randint(0, self.dim)
                for j in range(self.dim):
                    if r[j] <= CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = x1[j]
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                
                f_trial = evaluate(trial)
                if f_trial is None:
                    break
                
                # If the trial point is non-dominated, we might want to update parents
                # but for simplicity, we just rely on the next iteration's ND calculation.
            
            # Adapt F and CR
            # Shrink F to exploit local regions
            F *= 0.95
            if F < 0.1:
                F = 0.1

        return all_X
