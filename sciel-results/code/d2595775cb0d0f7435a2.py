"""Multi-objective optimizer for RE21 using Expected Hypervolume Improvement (EHVI) Bayesian Optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # 300s limit, use 280s to be safe
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Reference point for hypervolume calculation
        # The task says reference point [3000.0, 0.06] for scoring.
        # For EHVI, we need a reference point that is worse than any expected point.
        # We can use the observed worst or a fixed large value. 
        # Let's use [3000.0, 0.06] as the reference for the EHVI calculation as well, 
        # or slightly worse to ensure positive contributions. 
        # Actually, EHVI is defined relative to a reference point. 
        # If we use the scoring reference, it aligns the optimization with the score.
        ref_point = np.array([3000.0, 0.06])
        
        # State
        X = []
        F = []
        eval_count = 0
        
        def add_point(x, f):
            nonlocal eval_count
            X.append(x)
            F.append(f)
            eval_count += 1

        # 1. Initialization: Sobol Sequence
        # Use a quasi-random sequence for better coverage
        n_init = 50
        # Simple Sobol-like sampling using Halton or just stratified random
        # Since scipy.stats.qmc might be version dependent, we use stratified random for robustness
        # Or just random if qmc is not available. Let's try qmc.
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim, seed=0)
            # Generate points in [0,1]
            points_unit = sampler.random(n=n_init)
            X_init = lo + (hi - lo) * points_unit
        except ImportError:
            # Fallback to random
            X_init = lo + (hi - lo) * np.random.random((n_init, dim))
            
        for x in X_init:
            if time.time() > deadline:
                break
            f = problem(x)
            add_point(x, f)
            
        if eval_count == 0:
            return None

        X = np.array(X)
        F = np.array(F)
        
        # Helper: Calculate Hypervolume of non-dominated set
        # For 2 objectives, HV is easy to compute.
        def calc_hv(F_arr, ref):
            # F_arr: (N, 2)
            # Filter dominated points? 
            # For 2D, sort by f1, then iterate.
            if len(F_arr) == 0:
                return 0.0
            
            # Only consider points better than ref in all objectives
            valid = np.all(F_arr < ref, axis=1)
            F_valid = F_arr[valid]
            if len(F_valid) == 0:
                return 0.0
            
            # Sort by first objective (ascending)
            idx = np.argsort(F_valid[:, 0])
            F_sorted = F_valid[idx]
            
            hv = 0.0
            # The hypervolume is the sum of rectangles
            # For each point i in sorted order, the width is (f1[i+1] - f1[i]) or (ref[0] - f1[i]) for the last
            # The height is (ref[1] - f2[i])
            # But we must ensure we only count the non-dominated part.
            # Since we sorted by f1, we need to track the minimum f2 seen so far?
            # No, for non-dominated set, f2 decreases as f1 increases.
            # If the set is not non-dominated, dominated points will have higher f2 for same/higher f1.
            # Standard 2D HV algorithm:
            # Sort by f1 ascending.
            # Keep track of min_f2_so_far? No, we want the area under the curve of the Pareto front.
            # The Pareto front in 2D minimization is a decreasing curve.
            # We can filter to non-dominated first to be safe, or just compute the union of boxes.
            
            # Efficient 2D HV:
            # Sort by f1.
            # Iterate, maintaining the minimum f2 value seen (which represents the "envelope").
            # Actually, if we sort by f1, and for each point we take the min f2 from that point to the end?
            # Let's just use the standard iterative method for non-dominated sets.
            # First, filter to non-dominated.
            # N is small (max 200), so O(N^2) filter is fine.
            
            leq = np.all(F_valid[None, :, :] <= F_valid[:, None, :], axis=2)
            lt = np.any(F_valid[None, :, :] < F_valid[:, None, :], axis=2)
            dominated_mask = np.any(leq & lt, axis=1)
            F_nd = F_valid[~dominated_mask]
            
            if len(F_nd) == 0:
                return 0.0
            
            idx = np.argsort(F_nd[:, 0])
            F_nd_sorted = F_nd[idx]
            
            hv = 0.0
            for i in range(len(F_nd_sorted)):
                f1 = F_nd_sorted[i, 0]
                f2 = F_nd_sorted[i, 1]
                
                # Width of the box
                if i == len(F_nd_sorted) - 1:
                    width = ref[0] - f1
                else:
                    width = F_nd_sorted[i+1, 0] - f1
                
                if width <= 0:
                    continue
                    
                # Height of the box
                # The height is determined by the min f2 in the non-dominated set from i onwards?
                # In a non-dominated set sorted by f1, f2 is strictly decreasing.
                # So the height for the box starting at f1[i] is ref[1] - f2[i].
                # Wait, is that correct?
                # Yes, for the rectangle [f1[i], f1[i+1]] x [f2[i], ref[1]], 
                # the area is (f1[i+1]-f1[i]) * (ref[1]-f2[i]).
                # But wait, if f2[i] is the height, does it cover the whole width?
                # Yes, because for any x in [f1[i], f1[i+1]], the Pareto front has f2 <= f2[i] is NOT true.
                # The Pareto front has f2 >= f2[i] for x < f1[i] and f2 <= f2[i] for x > f1[i]?
                # Minimization: Lower f1 and Lower f2 is better.
                # Pareto front: As f1 increases, f2 decreases.
                # So for x in [f1[i], f1[i+1]], the "worst" (highest) f2 on the front is f2[i].
                # The area contributed by this segment is width * (ref[1] - f2[i]).
                
                hv += width * (ref[1] - f2)
                
            return hv

        # Helper: EHVI Acquisition Function
        # We need to compute the expected improvement in hypervolume.
        # E[HV(F ∪ {y}) - HV(F)]
        # Where y ~ GP(x)
        
        # GP Implementation (RBF Kernel)
        class SimpleGP:
            def __init__(self, length_scale=0.5, sigma=1.0, noise=1e-6):
                self.length_scale = length_scale
                self.sigma = sigma
                self.noise = noise
                self.X_train = None
                self.Y_train = None
                self.K_inv = None
                self.alpha = None
                self.L = None

            def rbf_kernel(self, X1, X2):
                # X1: (N, D), X2: (M, D)
                # Returns (N, M)
                X1_sq = np.sum(X1**2, axis=1)[:, None]
                X2_sq = np.sum(X2**2, axis=1)[None, :]
                dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
                dist_sq = np.maximum(dist_sq, 0)
                return self.sigma**2 * np.exp(-0.5 * dist_sq / (self.length_scale**2))

            def fit(self, X, Y):
                self.X_train = X
                self.Y_train = Y
                N = len(X)
                K = self.rbf_kernel(X, X)
                K[np.diag_indices(N)] += self.noise
                try:
                    self.L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y.T))
                    self.alpha = alpha
                except np.linalg.LinAlgError:
                    # Add jitter if Cholesky fails
                    K[np.diag_indices(N)] += 1e-4
                    self.L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y.T))
                    self.alpha = alpha

            def predict(self, X):
                K_star = self.rbf_kernel(X, self.X_train)
                mu = K_star @ self.alpha
                # Variance
                # v = L^-1 K_star^T
                # var = sigma^2 - sum(v^2, axis=0)
                v = np.linalg.solve(self.L, K_star.T)
                var = self.sigma**2 - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-8)
                return mu.T, var.T

        # Main Loop
        # We have 50 evals used. 150 remaining.
        # GP fitting is O(N^3). With N=150, this is fast.
        
        # Pre-compute current HV
        current_hv = calc_hv(F, ref_point)
        
        # We will use 150 iterations of BO.
        # Each iteration:
        # 1. Fit GP
        # 2. Sample candidates (e.g., 200 random points)
        # 3. Compute EHVI for each candidate
        # 4. Pick best
        # 5. Evaluate
        
        n_bo_iter = budget - eval_count
        
        # To speed up EHVI calculation, we can use a grid of candidates.
        # Since dim=4, we can use a Latin Hypercube or random sample of size M=500.
        
        for i in range(n_bo_iter):
            if time.time() > deadline:
                break
            if eval_count >= budget:
                break
            
            # Fit GP
            X_curr = np.array(X)
            F_curr = np.array(F)
            
            gp = SimpleGP(length_scale=0.5, sigma=1.0, noise=1e-5)
            gp.fit(X_curr, F_curr)
            
            # Generate candidate points
            # Use random sampling for candidates
            n_candidates = 200
            X_cand = lo + (hi - lo) * np.random.random((n_candidates, dim))
            
            # Predict mean and variance
            mu, var = gp.predict(X_cand)
            # mu: (n_candidates, 2), var: (n_candidates, 2)
            
            # Compute EHVI
            # EHVI(x) = E[ max(0, HV(F ∪ {y}) - HV(F)) ]
            # Approximation: Sample from the predictive distribution?
            # Or use the analytic EHVI for 2D?
            # Analytic EHVI for 2D is complex but possible.
            # Monte Carlo EHVI is simpler:
            # For each candidate x, sample K values from N(mu, var).
            # For each sample y, compute HV(F ∪ {y}) - HV(F).
            # Average over K samples.
            
            # To be fast, we can use a small K, e.g., 20 samples per candidate.
            # 200 candidates * 20 samples = 4000 HV calculations.
            # Each HV calc is O(N^2) ~ 150^2 = 22500 ops.
            # 4000 * 22500 = 90M ops. This might be too slow in Python.
            
            # Optimization: 
            # We can pre-compute the "contribution" of a new point more efficiently?
            # Or reduce n_candidates and K.
            # Let's try n_candidates=100, K=10.
            # 100 * 10 = 1000 HV calcs. 1000 * 22500 = 22.5M ops. ~0.1s. Acceptable.
            
            n_cand_bo = 100
            n_mc_samples = 10
            
            # Re-sample candidates if n_cand_bo < n_candidates
            if n_cand_bo < len(X_cand):
                X_cand = X_cand[:n_cand_bo]
                mu = mu[:n_cand_bo]
                var = var[:n_cand_bo]
            
            ehvi_values = np.zeros(n_cand_bo)
            
            for j in range(n_cand_bo):
                # Sample from GP
                # y ~ N(mu[j], sqrt(var[j]))
                # var[j] is (2,)
                std = np.sqrt(var[j])
                samples = mu[j] + np.random.randn(n_mc_samples, 2) * std
                
                # For each sample, compute HV(F ∪ {sample})
                # F_curr is (N, 2)
                # We can compute the difference efficiently?
                # HV(F ∪ {y}) - HV(F)
                
                # For 2D, the improvement is the area of the rectangle added by y
                # if y is non-dominated, minus any area removed? No, HV is monotonic.
                # If y is dominated by F, improvement is 0.
                # If y is non-dominated, it adds a box and might "cut off" parts of existing boxes?
                # No, in minimization, adding a point can only increase HV.
                # The increase is the area of the region [f1_y, ref1] x [f2_y, ref2] that was NOT covered by F.
                
                # Efficient check:
                # Find points in F that dominate y. If any, improvement = 0.
                # Otherwise, the improvement is the area of the box defined by y and ref,
                # minus the area of that box already covered by F.
                
                # Let's just compute HV(F ∪ {y}) - HV(F) using the calc_hv function 
                # but that's slow.
                
                # Alternative: 
                # Improvement = Area of box(y) - Area(box(y) ∩ Union of boxes(F))
                
                # Let's use a faster approximate EHVI or just use the MC with a faster HV diff.
                # We can compute the "new" area contributed by y.
                # y = (y1, y2)
                # Box: [y1, ref1] x [y2, ref2]
                # This box is partially covered by existing points.
                # A point f in F covers part of this box if f1 < y1 and f2 < y2? No.
                # If f1 < y1 and f2 < y2, then f dominates y? No, f is better.
                # If f1 < y1 and f2 < y2, then f is better than y in both objectives.
                # Then y is dominated, improvement 0.
                
                # If y is not dominated, it means for all f in F, either f1 > y1 or f2 > y2.
                # The area added is the area of the rectangle [y1, ref1] x [y2, ref2] 
                # that is not covered by any f in F.
                
                # Actually, for 2D, the hypervolume improvement of a single point y
                # relative to a set F can be computed as:
                # Let R_y be the rectangle [y1, ref1] x [y2, ref2].
                # The area of R_y is (ref1 - y1) * (ref2 - y2).
                # The part of R_y already covered by F is the union of rectangles [f1, ref1] x [f2, ref2] 
                # intersected with R_y.
                # Since we are in 2D, we can sort F by f1.
                
                # For simplicity and speed, let's just use the MC with the full HV calc 
                # but optimized for the small number of points added.
                # We can compute HV of F ∪ {y} by checking if y is dominated.
                
                # Let's try a faster MC:
                total_improvement = 0.0
                for k in range(n_mc_samples):
                    y = samples[k]
                    # Check if y is dominated
                    # If any f in F_curr has f <= y and f != y, then y is dominated.
                    # If dominated, improvement = 0.
                    # Else, compute HV(F ∪ {y}) - HV(F).
                    
                    # Fast check:
                    # Find min f2 in F where f1 < y1? No.
                    # Let's just compute the HV difference directly for 2D.
                    
                    # If y is not dominated:
                    # The new HV is HV(F) + (ref1 - y1) * (ref2 - y2) - (area of overlap)
                    # This is getting complicated to code fast in Python loop.
                    
                    # Let's stick to the full calc_hv but it's called 1000 times.
                    # Is 1000 * 0.01s = 10s too slow? 
                    # calc_hv with N=150 is very fast in NumPy.
                    # Let's assume it's fast enough.
                    
                    F_new = np.vstack([F_curr, y])
                    hv_new = calc_hv(F_new, ref_point)
                    total_improvement += max(0, hv_new - current_hv)
                
                ehvi_values[j] = total_improvement / n_mc_samples
            
            # Pick best candidate
            best_idx = np.argmax(ehvi_values)
            x_new = X_cand[best_idx]
            
            # Evaluate
            f_new = problem(x_new)
            add_point(x_new, f_new)
            
            # Update state
            X = np.array(X)
            F = np.array(F)
            current_hv = calc_hv(F, ref_point)

        return None

# EVOLVE-BLOCK-END
