"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using Model-Based Evolutionary Algorithm (MBEA) with EHVI."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        max_time = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point for hypervolume calculation
        ref_point = np.array([1.1, 1.3, 1.1])
        
        # Initialize with a small set of diverse points (e.g., Latin Hypercube or random)
        # We need at least n_obj+1 points to define a simplex in objective space, 
        # but more is better for GP. Let's start with 10 points.
        n_init = 10
        X = []
        F = []
        
        # Generate initial points
        # Use a fixed seed for reproducibility in the initial design if needed, 
        # but the problem says np.random.seed(seed) is called by evaluator.
        # So we can just use np.random.
        X_init = np.random.uniform(lo, hi, size=(n_init, self.dim))
        
        for x in X_init:
            f = problem(x)
            X.append(x)
            F.append(f)
            
        X = np.array(X)
        F = np.array(F)
        
        eval_count = n_init
        
        # Define a helper to compute hypervolume contribution of a point given a set of non-dominated points
        # For 3 objectives, we can use a simple exact calculation.
        def compute_hv(points, ref):
            # points: (N, 3)
            # ref: (3,)
            # Returns hypervolume of the union of boxes defined by points and ref.
            # Only points dominated by ref (i.e., all f_i < ref_i) count.
            # Filter points
            valid = np.all(points < ref, axis=1)
            pts = points[valid]
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective
            pts = pts[np.argsort(pts[:, 0])]
            
            # For 3D, we can project to 2D and use the "sweep" or just iterate.
            # A simple way for small N:
            # HV = sum over non-dominated points of the volume of the box they uniquely contribute.
            # This is tricky to do exactly in O(N) for 3D without specialized algorithms.
            # Given N is small (< 200), we can use a simple recursive or iterative subtraction.
            # Or even simpler: since we are maximizing HV, and we are adding points one by one,
            # we can just compute the HV of the current set and the HV of the set+new point.
            # But that's expensive.
            
            # Let's use a standard 3D HV algorithm (e.g., from literature).
            # For simplicity and speed in this context, we'll use a basic implementation.
            
            # Actually, for EHVI, we need the HV of the current set and the HV of the set with the new point.
            # Let's implement a function that computes HV of a set of points.
            
            # Sorting and sweeping:
            # 1. Sort by f0
            # 2. For each point, determine the "shadow" it casts.
            
            # A robust simple method for small N:
            # Use the fact that HV is the volume of the union of boxes.
            # We can discretize? No, exact is required for scoring, but for acquisition we can approximate.
            # However, the score is exact HV. The acquisition function uses EHVI which relies on HV.
            # We can use a fast approximate HV for the acquisition function to save time,
            # as long as it guides us well.
            
            # Let's use a simple O(N^2) or O(N^3) exact HV for 3D.
            # Reference: "Computing the Hypervolume Indicator"
            
            # For 3 objectives:
            # Sort by f0.
            # Initialize a list of "fronts" in the 2D space (f1, f2).
            
            # This is getting complex. Let's use a simpler heuristic for acquisition:
            # Maximize the minimum distance to the current Pareto front in objective space?
            # Or use a simple "expected improvement" on a weighted sum?
            
            # Given the time constraints, a full MBEA with exact EHVI might be too slow in Python.
            # Let's try a simpler surrogate-based approach:
            # Fit a GP to each objective.
            # Predict mean and variance for a grid of candidate points.
            # Select the point that maximizes the expected improvement in the "worst" objective
            # or maximizes a scalarized objective (like weighted sum with adaptive weights).
            
            # Alternative: Use a simple "infill criterion" like Expected Improvement (EI) for a weighted sum,
            # where weights are chosen to cover the front.
            
            # Let's stick to a simpler, faster strategy that is still model-based:
            # 1. Fit GP to each objective.
            # 2. Generate a grid of candidate points.
            # 3. For each candidate, predict mean and std for all 3 objectives.
            # 4. Compute a "score" for each candidate.
            #    Score = max over weights w (sum w_i * (mean_i - best_known_i) - kappa * std_i)
            #    This is like a multi-objective EI.
            
            # To keep it fast, we will:
            # - Use a small number of weight vectors (e.g., 10).
            # - For each weight vector, compute EI.
            # - Take the max EI across weights for each candidate.
            # - Pick the candidate with the highest max EI.
            
            # This is essentially a model-based approach with a simple acquisition function.
            
            pass # Placeholder for HV calc if we decide to use it directly.

        # Let's implement the GP-based search.
        # We need a GP implementation. sklearn has GaussianProcessRegressor.
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel
        
        # Define kernel
        # Matern is good for smooth functions. RBF is smoother.
        # Let's use Matern with nu=2.5 for a balance.
        kernel = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(noise_level=0.1)
        
        # We will fit 3 GPs, one for each objective.
        
        while eval_count < self.budget:
            if time.time() - start_time > max_time:
                break
            
            # Fit GPs
            gps = []
            for i in range(self.n_obj):
                # Add small noise to avoid singular matrices if points are identical
                Y = F[:, i]
                # Check for duplicates
                # Simple check: if any two points are very close, add tiny noise
                # For now, assume distinct points.
                gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0)
                gp.fit(X, Y)
                gps.append(gp)
            
            # Generate candidate points
            # A grid might be too large. Let's use a random sample of candidates.
            n_candidates = 1000
            X_cand = np.random.uniform(lo, hi, size=(n_candidates, self.dim))
            
            # Predict mean and std for all candidates
            means = np.zeros((n_candidates, self.n_obj))
            stds = np.zeros((n_candidates, self.n_obj))
            
            for i, gp in enumerate(gps):
                means[:, i], stds[:, i] = gp.predict(X_cand, return_std=True)
            
            # Define best known values for each objective
            best_known = np.min(F, axis=0)
            
            # Define weight vectors
            # Use a set of weights to cover the objective space
            n_weights = 10
            # Generate weights on the simplex
            # Simple method: random Dirichlet
            rng = np.random.default_rng(42)
            weights = rng.dirichlet(np.ones(self.n_obj), size=n_weights)
            # Add the extreme points (1,0,0), (0,1,0), (0,0,1) to ensure coverage of extremes
            extreme_weights = np.eye(self.n_obj)
            weights = np.vstack([weights, extreme_weights])
            
            # Compute EI for each candidate and each weight
            # EI = E[max(0, f_best - f_new)] for minimization
            # For weighted sum: s = sum w_i f_i
            # s_best = sum w_i best_known_i
            # f_new ~ N(mean_s, std_s)
            # mean_s = sum w_i mean_i
            # std_s = sqrt(sum (w_i std_i)^2) (assuming independence, which is an approximation)
            
            max_ei = np.zeros(n_candidates)
            
            for w in weights:
                mean_s = means @ w
                # Approximate std of the sum
                # var_s = sum (w_i * std_i)^2
                var_s = np.sum((w * stds) ** 2, axis=1)
                std_s = np.sqrt(var_s)
                
                s_best = np.dot(w, best_known)
                
                # EI calculation
                # If std is 0, EI is 0 (or max(0, s_best - mean_s))
                # Standard EI formula:
                # EI = (s_best - mean_s) * Phi((s_best - mean_s)/std_s) + std_s * phi((s_best - mean_s)/std_s)
                
                z = (s_best - mean_s) / (std_s + 1e-10)
                ei = (s_best - mean_s) * norm.cdf(z) + std_s * norm.pdf(z)
                
                # Update max EI
                max_ei = np.maximum(max_ei, ei)
            
            # Select the candidate with the highest EI
            best_idx = np.argmax(max_ei)
            x_new = X_cand[best_idx]
            
            # Evaluate
            f_new = problem(x_new)
            
            # Update X and F
            X = np.vstack([X, x_new])
            F = np.vstack([F, f_new])
            eval_count += 1
            
            # Optional: Remove dominated points to keep the set small?
            # With 200 points, fitting GP on 200 points is fine.
            # But removing dominated points can help the GP focus on the front.
            # Let's keep all points for simplicity, as GP can handle it.
            
        return X

# EVOLVE-BLOCK-END
