"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hv(self, points, ref):
        """
        Compute hypervolume for a set of points in 3D.
        points: (N, 3) array
        ref: (3,) array
        Returns scalar hypervolume.
        """
        if len(points) == 0:
            return 0.0
        
        # Filter points strictly better than reference
        mask = np.all(points < ref, axis=1)
        points = points[mask]
        if len(points) == 0:
            return 0.0
            
        # Sort by first objective
        points = points[np.argsort(points[:, 0])]
        
        hv = 0.0
        prev_h = 0.0
        prev_w = 0.0
        
        for i in range(len(points)):
            f1, f2, f3 = points[i]
            
            # Width in dimension 2 (f2)
            # We need the minimum f2 among points j where f1_j <= f1_i
            # Since sorted by f1, all previous points have f1 <= current f1
            # But we also need to consider the current point's f2
            # Actually, the standard 3D HV algorithm sorts by f1, then for each point,
            # it computes the contribution based on the envelope of f2 and f3.
            
            # Let's use a simpler, robust O(N^2) or O(N log N) approach for N~100.
            # Given small N (population size ~20-40), O(N^3) is too slow if done naively, 
            # but O(N^2) is fine.
            
            pass
        
        # Efficient 3D HV algorithm (Saxena et al. / simple sweep)
        # Sort by f1 ascending
        idx = np.argsort(points[:, 0])
        pts = points[idx]
        
        hv = 0.0
        # For each point i, the contribution is:
        # (f1_{i+1} - f1_i) * HV_2D(points[i+1:], ref[1:]) 
        # where HV_2D is the hypervolume in the (f2, f3) plane for points with index > i
        # bounded by ref[1], ref[2].
        
        # Precompute 2D HV for suffixes? 
        # Simpler: Iterate i from N-1 down to 0.
        # Maintain a list of "active" points in (f2, f3) space that are not dominated by any point with smaller f1 (which we've already processed in reverse? No, we process from large f1 to small f1).
        
        # Let's stick to a verified O(N^2) method for N < 100.
        # HV = sum of boxes.
        # Sort by f1.
        # For each point i, its box extends to the next point in f1, or ref[0].
        # The cross-section in (f2, f3) is the area under the Pareto front of points j > i (in f1 order) plus point i itself?
        # Actually, point i dominates points with larger f1 only if it has smaller f2 and f3.
        
        # Correct 3D HV Algorithm (sweep line):
        # 1. Sort points by f1.
        # 2. Iterate i from N-1 down to 0.
        # 3. Maintain a structure of points j > i.
        # 4. The contribution of point i is (f1_{i+1} - f1_i) * Area_2D({points j >= i} projected on f2, f3).
        #    Wait, the "next" point in f1 is i+1. The box for i is from f1_i to f1_{i+1}.
        #    The height in f2, f3 is determined by the Pareto front of points with index >= i.
        
        # Let's just use a simple recursive or iterative 2D HV calculator for the suffix.
        
        def hv_2d(pts_2d, ref_2d):
            # pts_2d: (M, 2), ref_2d: (2,)
            if len(pts_2d) == 0:
                return 0.0
            # Sort by first obj
            pts_2d = pts_2d[np.argsort(pts_2d[:, 0])]
            area = 0.0
            max_y = ref_2d[1]
            for i in range(len(pts_2d) - 1, -1, -1):
                x, y = pts_2d[i]
                if y < max_y:
                    width = (pts_2d[i+1][0] if i+1 < len(pts_2d) else ref_2d[0]) - x
                    if width > 0:
                        area += width * (max_y - y)
                    max_y = y
            return area

        # Apply to 3D
        n = len(pts)
        # Suffix points for 2D HV
        # We can compute this in O(N^2)
        total_hv = 0.0
        for i in range(n - 1, -1, -1):
            # The box width in f1
            f1_next = pts[i+1][0] if i+1 < n else ref[0]
            width_f1 = f1_next - pts[i][0]
            if width_f1 <= 0:
                continue
                
            # The cross section is the 2D HV of points[i:] in (f2, f3)
            # Note: Points with index < i have smaller f1, so they don't contribute to the volume "behind" i in the f1 direction?
            # Actually, the volume associated with point i is the volume of the box [f1_i, f1_{i+1}] x [f2_i, ref[1]] x [f3_i, ref[2]] 
            # MINUS the parts dominated by other points.
            # The sweep line approach:
            # Volume = sum_{i} (f1_{i+1} - f1_i) * HV_2D( {p_j | j >= i}, ref[1:])
            
            # Extract 2D points for indices i to n-1
            pts_2d = pts[i:, [1, 2]]
            ref_2d = ref[1:]
            
            # Filter 2D points strictly better than ref_2d
            mask_2d = np.all(pts_2d < ref_2d, axis=1)
            pts_2d = pts_2d[mask_2d]
            
            area = hv_2d(pts_2d, ref_2d)
            total_hv += width_f1 * area
            
        return total_hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3.0, 5.0, 7.0])
        
        dim = self.dim
        n_obj = self.n_obj
        
        # MO-CMA-ES Parameters
        pop_size = 20  # Lambda
        mu = pop_size // 2  # Number of parents
        
        # Initialize mean and covariance
        # Start near center
        mean = (lo + hi) / 2.0
        # Initial sigma
        sigma = 0.1 * np.max(hi - lo)
        C = np.eye(dim)
        
        # Archive of non-dominated solutions found so far (for final return, but we only need to return X)
        # We will keep the population X and F.
        
        # Initial population
        X = mean + sigma * np.random.multivariate_normal(np.zeros(dim), C, size=pop_size)
        X = np.clip(X, lo, hi)
        F = np.array([problem(x) for x in X])
        evaluations = pop_size
        
        # Strategy Parameters
        # We will use a simple hypervolume-based selection.
        
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Select parents based on hypervolume contribution
            # Compute HV contribution for each point in current population
            # To save time, we can just use non-dominated sort or a simple HV approximation.
            # Given small pop size, exact HV per point is feasible if we do it smartly, 
            # but O(N^3) might be slow. Let's use a simple non-dominated sort + crowding 
            # or just random selection if HV is too slow.
            
            # Actually, MO-CMA-ES typically uses the hypervolume indicator.
            # Let's compute the marginal HV contribution of each point.
            # This requires computing HV of the set and HV of the set minus point i.
            # O(N^2 * Cost_HV). With N=20, this is 20 * 20^2 = 8000 ops * HV_cost.
            # HV_cost for N=20 is small.
            
            # Alternative: Use non-dominated sorting.
            # Let's stick to a robust, fast heuristic: 
            # Select mu parents. 
            # 1. Identify non-dominated front.
            # 2. If front size > mu, select from front based on crowding distance or HV.
            # 3. If front size < mu, add next front.
            
            # For simplicity and speed, let's use a simple non-dominated sort.
            # It's O(N^2).
            
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True
                        
            front_indices = np.where(~dominated)[0]
            
            # If front is too large, we need to reduce it.
            # If front is too small, we add more.
            selected_indices = []
            
            if len(front_indices) >= mu:
                # Select mu from front
                # Use crowding distance or random
                # Simple: Random
                selected_indices = np.random.choice(front_indices, size=mu, replace=False)
            else:
                selected_indices = list(front_indices)
                # Need more parents from dominated points
                # Pick random dominated points
                dominated_indices = np.where(dominated)[0]
                remaining = mu - len(selected_indices)
                if len(dominated_indices) >= remaining:
                    selected_indices.extend(np.random.choice(dominated_indices, size=remaining, replace=False))
                else:
                    selected_indices.extend(dominated_indices)
                    # If still not enough, repeat from front
                    while len(selected_indices) < mu:
                        selected_indices.append(np.random.choice(front_indices))
                        
            parents_X = X[selected_indices]
            parents_F = F[selected_indices]
            
            # 2. Recombine (CMA-ES step)
            # Weighted mean
            # Standard CMA-ES weights
            w = np.log(mu) - np.arange(mu)
            w = w / w.sum()
            
            mean_new = np.sum(w[:, np.newaxis] * parents_X, axis=0)
            
            # Update covariance
            # C = (1 - c1 - cm) * C + c1 * (mean_new - mean) @ (mean_new - mean).T + cm * sum(w_i * (x_i - mean) @ (x_i - mean).T)
            # Simplified: Just use the standard CMA-ES update
            # We need to track the history or use the standard algorithm.
            # For a quick implementation, we can just do:
            # 1. Compute new mean
            # 2. Compute new sigma
            # 3. Generate offspring from N(mean_new, sigma^2 * C)
            
            # Standard CMA-ES updates C and sigma.
            # Let's implement a basic version.
            
            # Transformation
            # y_i = (x_i - mean) / sigma
            Y = (parents_X - mean) / sigma
            
            # Update C
            # c1 = 2 / ((dim+1)^2 + 10)
            # cm = 2 * (mu_eff - 2 + 1/mu_eff) / ((dim+2)^2 + 20)
            # mu_eff = (sum(w)^2) / (sum(w^2))
            
            mu_eff = (np.sum(w)**2) / np.sum(w**2)
            c1 = 2 / ((dim + 1)**2 + 10)
            cm = 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + 20)
            
            # C = (1 - c1 - cm) * C + c1 * (mean_new - mean) @ (mean_new - mean).T / sigma^2 + cm * sum(w_i * Y_i @ Y_i.T)
            # Note: (mean_new - mean) is in x-space. Y is in standardized space.
            # The term (mean_new - mean)/sigma is the "evolution path" approximation?
            # Standard CMA-ES uses evolution paths p_sigma and p_c.
            # For simplicity, we can use the "rank-one" and "rank-mu" updates directly on C.
            
            # Rank-one update
            diff = (mean_new - mean) / sigma
            C += c1 * np.outer(diff, diff)
            
            # Rank-mu update
            C -= cm * C  # This is part of (1 - c1 - cm) * C
            # Wait, the formula is C_new = (1 - c1 - cm) C_old + c1 * ... + cm * ...
            # So:
            C = (1 - c1 - cm) * C + c1 * np.outer(diff, diff)
            for i in range(mu):
                C += cm * w[i] * np.outer(Y[i], Y[i])
                
            # Update sigma
            # sigma = sigma * exp( (c_sigma / d_sigma) * (||Y|| / chi_n - 1) )
            # chi_n = sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim^2))
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 2) / (dim + mu_eff - 2)) + c_sigma
            
            norm_Y = np.sqrt(np.mean(np.sum(Y**2, axis=1)))
            sigma *= np.exp((c_sigma / d_sigma) * (norm_Y / chi_n - 1))
            
            mean = mean_new
            
            # 3. Generate Offspring
            # Ensure C is symmetric and positive definite (approximate)
            C = (C + C.T) / 2
            # Add small regularization if needed
            C += 1e-6 * np.eye(dim)
            
            # Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to random if C is not PD
                L = np.eye(dim)
                
            Z = np.random.randn(pop_size, dim)
            X_off = mean + sigma * (Z @ L.T)
            X_off = np.clip(X_off, lo, hi)
            
            F_off = np.array([problem(x) for x in X_off])
            evaluations += pop_size
            
            # 4. Update Population
            # In MO-CMA-ES, we often maintain an archive or just replace the population.
            # A common strategy is to keep the best non-dominated points from the union of parents and offspring.
            # Let's combine parents and offspring, and select the best pop_size points based on non-domination.
            
            all_X = np.vstack([X, X_off])
            all_F = np.vstack([F, F_off])
            
            # Non-dominated sort on combined set
            n_all = len(all_F)
            dominated_all = np.zeros(n_all, dtype=bool)
            for i in range(n_all):
                if dominated_all[i]:
                    continue
                for j in range(n_all):
                    if i == j:
                        continue
                    if np.all(all_F[i] <= all_F[j]) and np.any(all_F[i] < all_F[j]):
                        dominated_all[j] = True
                        
            front_all = np.where(~dominated_all)[0]
            
            # Select next population
            if len(front_all) >= pop_size:
                # Select pop_size from front
                # Use crowding distance for diversity
                # Simple crowding: for each dimension, sort and assign distance
                dists = np.zeros(len(front_all))
                for k in range(n_obj):
                    sorted_idx = np.argsort(all_F[front_all, k])
                    dists[sorted_idx[0]] = np.inf
                    dists[sorted_idx[-1]] = np.inf
                    f_min = all_F[front_all[sorted_idx[0]], k]
                    f_max = all_F[front_all[sorted_idx[-1]], k]
                    if f_max > f_min:
                        for j in range(1, len(sorted_idx) - 1):
                            dists[sorted_idx[j]] += (all_F[front_all[sorted_idx[j+1]], k] - all_F[front_all[sorted_idx[j-1]], k]) / (f_max - f_min)
                    
                # Select top pop_size by distance
                top_idx = np.argsort(dists)[::-1][:pop_size]
                new_indices = front_all[top_idx]
            else:
                # Take all non-dominated, fill with random dominated
                new_indices = list(front_all)
                dom_idx = np.where(dominated_all)[0]
                need = pop_size - len(new_indices)
                if need > 0 and len(dom_idx) > 0:
                    new_indices.extend(np.random.choice(dom_idx, size=min(need, len(dom_idx)), replace=False))
                    
            X = all_X[new_indices]
            F = all_F[new_indices]
            
        return X
# EVOLVE-BLOCK-END
