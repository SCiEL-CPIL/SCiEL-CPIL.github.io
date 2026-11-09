import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # 20s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # HypE strategy:
        # 1. Initialize with Sobol sequence for better coverage.
        # 2. Iteratively select the point from the current population that maximizes
        #    the expected hypervolume improvement (or a proxy like distance to a random point
        #    in the objective space, weighted by hypervolume contribution).
        #    Actually, standard HypE samples a point y from a distribution proportional to
        #    hypervolume contribution, and minimizes distance to y.
        #    For 2 objectives, we can compute exact hypervolume contribution.
        
        # We will maintain a list of all evaluated points (X, F).
        # To keep memory and time manageable, we might prune dominated points from the
        # "active" set used for selection, but we must keep all for the evaluator record.
        # The evaluator records every call to problem(). So we just need to be careful
        # with the number of calls.
        
        # Budget: 200
        # Let's use a population size for the "archive" of non-dominated points.
        # HypE typically works with a small archive (e.g., 10-50).
        # Let's start with 20 points.
        # Init: 20 points.
        # Remaining: 180 points.
        
        # Strategy:
        # 1. Generate 20 initial points using Sobol.
        # 2. Evaluate them.
        # 3. For each remaining eval:
        #    a. Compute the non-dominated set from the current archive.
        #    b. Sample a point y in the objective space (weighted by HV contribution).
        #    c. Find the point x in the archive closest to y (in objective space).
        #    d. Generate a new point x_new by perturbing x (e.g., Gaussian noise).
        #    e. Evaluate x_new.
        #    f. Add to archive.
        
        # This is a "memetic" or "local search" flavor of HypE, or just standard HypE
        # where the "surrogate" is the actual objective function (expensive, but we have budget).
        # Actually, standard HypE assumes a surrogate. Here we can just use the actual function.
        # But generating x_new is the tricky part. In standard HypE, you minimize distance to y
        # using a single-objective optimizer. Here, we have limited budget.
        
        # Alternative simpler HypE-like approach for 2D:
        # Maintain a list of non-dominated points.
        # For each new point, choose a parent from the non-dominated set.
        # The probability of choosing a parent is proportional to its hypervolume contribution.
        # Then, apply a mutation to the parent.
        # This is essentially a "Hypervolume-guided" mutation.
        
        # Let's try this:
        # 1. Init: 30 points (Sobol).
        # 2. Loop:
        #    - Identify non-dominated points in the current population.
        #    - Compute hypervolume contribution of each non-dominated point.
        #    - Select one parent with probability proportional to HV contribution.
        #    - Mutate the parent (Gaussian noise, sigma decreases over time?).
        #    - Evaluate new point.
        #    - Add to population.
        
        # This is different from NSGA-II (which uses crowding distance) and different from
        # simple random search. It directly optimizes the scoring metric (HV).
        
        # Implementation details:
        # Sobol sequence generation in numpy? No built-in.
        # We can use a simple Latin Hypercube or just random with stratification.
        # Let's use a simple stratified random sampling for init.
        
        def stratified_sample(n, dim, lo, hi):
            # Divide each dimension into n bins, pick one random point in each bin.
            # For dim=4, this is a bit complex to do perfectly in a grid.
            # Let's just use random for now, or a simple grid.
            # With 30 points and 4 dims, a grid is 30^(1/4) ~ 2.3, so 3x3x3x3 = 81 points.
            # We can't afford a full grid.
            # Let's use random. Sobol is better but hard to implement quickly without scipy.stats.qmc (available in scipy 1.7+).
            # Let's try scipy.stats.qmc.Sobol if available.
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=dim, scramble=True, seed=np.random.get_state()[1][0])
                samples = sampler.random(n)
                # Map [0,1] to [lo, hi]
                samples = lo + (hi - lo) * samples
                return samples
            except ImportError:
                # Fallback to random
                return lo + (hi - lo) * np.random.random((n, dim))

        n_init = 30
        X_init = stratified_sample(n_init, self.dim, lo, hi)
        F_init = np.array([problem(x) for x in X_init])
        
        # Archive of all evaluated points
        X_all = list(X_init)
        F_all = list(F_init)
        
        # Helper: Compute hypervolume contribution for 2 objectives
        # Reference point: [3000.0, 0.06]
        ref = np.array([3000.0, 0.06])
        
        def hypervolume_contribution_2d(points_f, ref):
            # points_f: list of 2D vectors
            # Returns list of HV contributions
            # Sort by f1
            if not points_f:
                return []
            
            # Filter points strictly better than ref
            valid = [f for f in points_f if f[0] < ref[0] and f[1] < ref[1]]
            if not valid:
                return [0.0] * len(points_f)
            
            # Sort by f1
            valid_sorted = sorted(valid, key=lambda x: x[0])
            
            # Calculate contribution
            # For 2D, the HV of a point (f1, f2) relative to the front is:
            # (f1_next - f1) * (f2_ref - f2) if it's the leftmost
            # Generally, sort by f1 ascending.
            # Contribution of point i: (f1_{i+1} - f1_i) * (ref[1] - f2_i)
            # But we need to handle the "envelope".
            
            # Let's compute the total HV of the set of valid points.
            # Then, for each point, remove it and compute HV again? Too slow (O(N^2)).
            # For 2D, there's an O(N log N) way to compute HV, but contribution is harder.
            
            # Approximation for selection:
            # A point's contribution is roughly the area it "covers" that no other point covers.
            # For 2D, if we sort by f1, the contribution of point i is:
            # (max(f1_{i+1}, ref[0]) - f1_i) * (ref[1] - min(f2_i, f2_{i-1}, ...))
            # This is getting complicated.
            
            # Simpler proxy for 2D:
            # Distance to the "ideal" corner (0,0) or similar? No.
            # Let's use the area of the rectangle defined by the point and the reference point,
            # minus the area covered by points that dominate it?
            
            # Actually, for selection in HypE, we can just sample a random point y in the
            # objective space (uniformly in the box defined by the current front and ref),
            # and pick the point in the front that is closest to y.
            # This is the standard HypE sampling strategy.
            
            # Let's implement that.
            return None # Not using this function directly for contribution, using sampling instead.

        def select_parent(F_nd):
            # F_nd: list of non-dominated objective vectors
            # 1. Find the "envelope" bounds.
            # f1_min = min(f[0] for f in F_nd)
            # f1_max = max(f[0] for f in F_nd)
            # f2_min = min(f[1] for f in F_nd)
            # f2_max = max(f[1] for f in F_nd)
            
            # 2. Sample y uniformly in [f1_min, ref[0]] x [f2_min, ref[1]]?
            # No, the region of interest is the union of rectangles from each point to ref.
            # A simpler approximation: Sample y uniformly in the bounding box of the front.
            # Or, sample y uniformly in the box [0, ref[0]] x [0, ref[1]]?
            
            # Standard HypE:
            # 1. Compute the total HV of the front.
            # 2. Sample y uniformly in the region that contributes to HV.
            # 3. Find the point in the front that is closest to y (Euclidean distance).
            
            # For 2D, the region is the area under the front.
            # Let's approximate by sampling y in [min_f1, max_f1] x [min_f2, max_f2]
            # and finding the closest point.
            
            if not F_nd:
                return None
            
            f1s = [f[0] for f in F_nd]
            f2s = [f[1] for f in F_nd]
            
            # To be more accurate, we can sample in the union of rectangles.
            # But let's start with the bounding box of the non-dominated points.
            y1 = np.random.uniform(min(f1s), ref[0])
            y2 = np.random.uniform(min(f2s), ref[1])
            
            # Find closest point
            min_dist = float('inf')
            best_idx = 0
            for i, f in enumerate(F_nd):
                dist = (f[0] - y1)**2 + (f[1] - y2)**2
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
                    
            return best_idx

        def get_non_dominated_indices(F):
            nd = []
            for i in range(len(F)):
                dominated = False
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        # Main loop
        # We have 30 evals used.
        # Budget is 200.
        # We have 170 left.
        
        # We will keep a "population" of the last, say, 50 points for selection?
        # Or just the non-dominated set?
        # HypE usually maintains a small archive of non-dominated points.
        # Let's maintain the non-dominated set from ALL evaluated points.
        # But checking all 200 points for non-domination every step is O(N^2) = 40000 ops.
        # 170 steps * 40000 = 6.8M ops. This is fast in Python? Maybe 1-2 seconds.
        # So it's feasible.
        
        # However, we need to map the index in the non-dominated list back to the X coordinate.
        
        for step in range(self.budget - n_init):
            if time.time() > deadline:
                break
                
            # 1. Get non-dominated indices from F_all
            nd_indices = get_non_dominated_indices(F_all)
            
            if not nd_indices:
                # Should not happen if we have valid points
                break
            
            # 2. Extract non-dominated F values
            F_nd = [F_all[i] for i in nd_indices]
            
            # 3. Select a parent
            parent_idx_in_nd = select_parent(F_nd)
            if parent_idx_in_nd is None:
                break
            
            global_parent_idx = nd_indices[parent_idx_in_nd]
            x_parent = X_all[global_parent_idx]
            
            # 4. Mutate
            # Gaussian noise. Sigma should be adaptive?
            # Let's use a fixed sigma relative to the range.
            # sigma = 0.1 * (hi - lo)
            sigma = 0.1 * (hi - lo)
            noise = np.random.normal(0, sigma)
            x_new = x_parent + noise
            x_new = np.clip(x_new, lo, hi)
            
            # 5. Evaluate
            f_new = problem(x_new)
            X_all.append(x_new)
            F_all.append(f_new)
            
        return None
