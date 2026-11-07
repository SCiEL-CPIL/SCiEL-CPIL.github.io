"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution with adaptive parameters and crowding distance for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_3d(self, points, ref):
        """Compute exact hypervolume for 3D points using inclusion-exclusion or sorting method."""
        # Filter points strictly better than reference
        points = np.array(points)
        ref = np.array(ref)
        mask = np.all(points < ref, axis=1)
        points = points[mask]
        if len(points) == 0:
            return 0.0
        
        # Sort by first objective
        points = points[points[:, 0].argsort()]
        
        # Use a simple O(n^2) or O(n^3) method for 3D
        # For small n, we can use the standard algorithm
        # Here we use a simple approach: for each point, compute the volume it uniquely contributes
        # Actually, let's use a more robust method for 3D
        
        n = len(points)
        if n == 0:
            return 0.0
        
        # Sort by f1
        points = points[np.argsort(points[:, 0])]
        
        # Alternative: use the standard 3D hypervolume algorithm
        # Sort by first dimension
        points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
        
        # For 3D, we can use the following:
        # hv = sum over all points of (ref[0]-p0[i]) * (ref[1]-p1[i]) * (ref[2]-p2[i]) 
        # minus overlaps... this is complex.
        
        # Let's use a simple Monte Carlo or grid-based approximation? No, we need exact.
        # For 3D with small n (pop size ~50-100), we can use the following:
        
        # Implementation of 3D hypervolume calculation
        # Sort points by first coordinate
        points = points[np.argsort(points[:, 0])]
        
        hv = 0.0
        for i in range(n):
            # For point i, calculate the volume it contributes
            # This is the volume of the box from point i to reference, 
            # minus volumes already counted by better points
            
            # Find all points that dominate point i in dimensions 1 and 2
            # while having smaller or equal value in dimension 0
            dominated = points[:i+1]
            
            # Filter to points that are better in dim 0 and dim 1
            # Actually, let's use a simpler approach for 3D:
            # For each point, the contribution is the volume of the region
            # where this point is the minimum in all dimensions
            
            # Get the box defined by this point and reference
            box_vol = (ref[0] - points[i, 0]) * (ref[1] - points[i, 1]) * (ref[2] - points[i, 2])
            
            if box_vol <= 0:
                continue
            
            # Subtract volumes of boxes that overlap and are from points 
            # that are better in at least one dimension
            overlap_vol = 0.0
            for j in range(i):
                # Point j is better if it's smaller in at least one dimension
                # and not worse in all dimensions
                if points[j, 0] <= points[i, 0] and points[j, 1] <= points[i, 1] and points[j, 2] <= points[i, 2]:
                    if points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1] or points[j, 2] < points[i, 2]:
                        # Calculate overlap volume
                        ox = min(ref[0], points[i, 0]) - max(points[j, 0], points[i, 0])
                        oy = min(ref[1], points[i, 1]) - max(points[j, 1], points[i, 1])
                        oz = min(ref[2], points[i, 2]) - max(points[j, 2], points[i, 2])
                        
                        if ox > 0 and oy > 0 and oz > 0:
                            overlap_vol += ox * oy * oz
            
            hv += box_vol - overlap_vol
        
        return hv
        
        # Actually, let's just return 0 for now and rely on the evaluator's hypervolume computation.
        # The evaluator computes the hypervolume, not us. We just need to return a good set of points.
        return 0.0

    def _crowding_distance(self, F):
        """Compute crowding distance for a set of points in objective space."""
        n = len(F)
        if n <= 2:
            return np.ones(n)
        
        cd = np.zeros(n)
        for m in range(F.shape[1]):
            # Sort by objective m
            order = np.argsort(F[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for i in range(1, n - 1):
                cd[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / (F[:, m].max() - F[:, m].min() + 1e-10)
        
        return cd

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return np.where(~is_dominated)[0]

    def _nondominated_sort(self, F):
        """Perform non-dominated sorting on F. Returns (ranks, fronts)."""
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        fronts = []
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_doms = np.zeros(n, dtype=int)  # n_doms[i] = count of points dominating i
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_doms[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_doms[i] += 1
        
        # First front
        front_idx = 0
        current_front = [i for i in range(n) if n_doms[i] == 0]
        
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                ranks[i] = front_idx
                for j in S[i]:
                    n_doms[j] -= 1
                    if n_doms[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front_idx += 1
            
        return ranks, fronts

    def __call__(self, problem, bounds):
        # Use global numpy random state to respect evaluator's seeding
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size: smaller allows more generations
        pop_size = 30
        
        # Initialize population
        # Bias towards 0 for DTLZ4? Standard uniform is safer for "keep approach"
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Differential Evolution parameters
        F_factor = 0.8  # Scaling factor
        CR = 0.9  # Crossover rate
        
        # Main loop
        while used + pop_size <= budget:
            # Generate offspring
            X_offspring = np.empty_like(X)
            F_offspring = np.empty_like(F)
            
            for i in range(pop_size):
                # Select 3 distinct parents
                p1_idx, p2_idx, p3_idx = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = X[p1_idx] + F_factor * (X[p2_idx] - X[p3_idx])
                
                # Crossover
                jrand = np.random.randint(dim)
                y = np.empty(dim)
                mask = np.random.random(dim) < CR
                mask[jrand] = True
                y[mask] = v[mask]
                y[~mask] = X[p1_idx][~mask]
                
                # Clip to bounds
                y = np.clip(y, lo, hi)
                
                # Evaluate
                fy = problem(y)
                used += 1
                
                X_offspring[i] = y
                F_offspring[i] = fy
            
            # Combine and Select (NSGA-II style)
            X_all = np.vstack([X, X_offspring])
            F_all = np.vstack([F, F_offspring])
            
            # Non-dominated sorting
            ranks, fronts = self._nondominated_sort(F_all)
            
            # Select best pop_size individuals
            X_new = np.empty_like(X)
            F_new = np.empty_like(F)
            
            count = 0
            for front in fronts:
                if count + len(front) <= pop_size:
                    # Take all from this front
                    X_new[count:count+len(front)] = X_all[front]
                    F_new[count:count+len(front)] = F_all[front]
                    count += len(front)
                else:
                    # Need to select subset of this front using crowding distance
                    needed = pop_size - count
                    cd = self._crowding_distance(F_all[front])
                    # Sort by crowding distance descending
                    sorted_indices = np.argsort(-cd)
                    selected_local = sorted_indices[:needed]
                    
                    # Map back to global indices
                    global_indices = front[selected_local]
                    
                    X_new[count:pop_size] = X_all[global_indices]
                    F_new[count:pop_size] = F_all[global_indices]
                    break
            
            X = X_new
            F = F_new
        
        # If there are remaining evaluations, use them to refine?
        # Or just return. The loop condition `used + pop_size <= budget` stops when we can't do a full generation.
        # We might have 0-29 evals left. Let's use them for local search or just ignore.
        # Given the small budget, ignoring the remainder is acceptable, or we can do a few more DE steps.
        # Let's do a few more single-point DE steps if budget allows.
        while used < budget:
             p1_idx, p2_idx, p3_idx = np.random.choice(pop_size, 3, replace=False)
             v = X[p1_idx] + F_factor * (X[p2_idx] - X[p3_idx])
             jrand = np.random.randint(dim)
             y = np.empty(dim)
             mask = np.random.random(dim) < CR
             mask[jrand] = True
             y[mask] = v[mask]
             y[~mask] = X[p1_idx][~mask]
             y = np.clip(y, lo, hi)
             fy = problem(y)
             used += 1
             
             # Simple selection: if fy is non-dominated, replace worst CD point
             is_dominated = False
             for i in range(pop_size):
                 if np.all(F[i] <= fy) and np.any(F[i] < fy):
                     is_dominated = True
                     break
             
             if not is_dominated:
                 cd = self._crowding_distance(F)
                 replace_idx = np.argmax(cd)
                 X[replace_idx] = y
                 F[replace_idx] = fy

        return X
# EVOLVE-BLOCK-END
