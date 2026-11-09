"""Multi-Objective Bayesian Optimization for RE21 using EHVI acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point for Hypervolume
        ref_point = np.array([3000.0, 0.06])
        
        # Data storage
        X = []
        Y = []
        
        eval_count = 0
        max_evals = self.budget

        def safe_problem(x):
            nonlocal eval_count
            if eval_count >= max_evals or time.time() > deadline:
                return None
            x = np.array(x)
            fx = problem(x)
            eval_count += 1
            return fx

        # 1. Initialization: Sobol Quasi-Random Sequence
        # Use a Sobol sequence for low-discrepancy coverage.
        # We need to generate points in [0,1]^d and map to bounds.
        # Since we don't have scipy.stats.qmc.Sobol in older versions sometimes, 
        # and to be safe, we can use a simple Halton or just random if Sobol is tricky.
        # But scipy is available. Let's try to use a deterministic low-discrepancy set.
        # Actually, the prompt says "Sobol quasi-random sequence". 
        # scipy.stats.qmc is available in recent SciPy.
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=False)
            # Generate 2^10 = 1024 points, take first 20
            sobol_samples = sampler.random(1024)[:20, :]
        except Exception:
            # Fallback to random if Sobol fails
            sobol_samples = np.random.rand(20, self.dim)

        # Map to bounds
        X_init = sobol_samples * (hi - lo) + lo
        
        for x in X_init:
            fx = safe_problem(x)
            if fx is None:
                continue
            X.append(x)
            Y.append(fx)
        
        if not X:
            return np.array([])
            
        X = np.array(X)
        Y = np.array(Y)
        
        # Helper: Exact Hypervolume for 2D
        def compute_hypervolume(points, ref):
            """
            points: Nx2 array
            ref: 2D array
            Returns hypervolume dominated by points w.r.t ref.
            Only points strictly better than ref in all objectives count.
            """
            # Filter points
            valid = []
            for p in points:
                if np.all(p < ref):
                    valid.append(p)
            if not valid:
                return 0.0
            
            valid = np.array(valid)
            
            # Remove dominated points (keep Pareto front)
            # In 2D, sort by obj1 ascending
            valid = valid[np.argsort(valid[:, 0])]
            
            # Keep only non-dominated
            pareto = []
            min_obj2 = np.inf
            for p in valid:
                if p[1] < min_obj2:
                    pareto.append(p)
                    min_obj2 = p[1]
            
            if not pareto:
                return 0.0
                
            pareto = np.array(pareto)
            
            # Calculate HV
            # Add reference point to the front to close the polygon
            # The front is sorted by f1 asc, f2 desc (after filtering)
            # Points: (f1_i, f2_i)
            # HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (ref2 - f2_i)
            # Where f1_{n+1} = ref1
            
            hv = 0.0
            for i in range(len(pareto)):
                f1 = pareto[i, 0]
                f2 = pareto[i, 1]
                if i < len(pareto) - 1:
                    next_f1 = pareto[i+1, 0]
                else:
                    next_f1 = ref[0]
                
                width = next_f1 - f1
                height = ref[1] - f2
                
                if width > 0 and height > 0:
                    hv += width * height
            return hv

        # Helper: Non-dominated set
        def get_nondominated(Y):
            if len(Y) == 0:
                return np.array([])
            nd_mask = np.ones(len(Y), dtype=bool)
            for i in range(len(Y)):
                if not nd_mask[i]:
                    continue
                for j in range(len(Y)):
                    if i == j:
                        continue
                    if np.all(Y[j] <= Y[i]) and np.any(Y[j] < Y[i]):
                        nd_mask[i] = False
                        break
            return Y[nd_mask]

        # Gaussian Process Surrogate (Simple implementation using RBF kernel)
        class GP2D:
            def __init__(self, alpha=1e-6, sigma_f=1.0, sigma_l=1.0):
                self.alpha = alpha
                self.sigma_f = sigma_f
                self.sigma_l = sigma_l
                self.X_train = None
                self.Y_train = None
                self.K_inv = None
                self.f_mean = None
                self.f_std = None
                self.y_mean = 0
                self.y_std = 1
                self.is_trained = False

            def _rbf_kernel(self, X1, X2):
                # Squared exponential kernel
                # Compute squared distances
                # (x1 - x2)^2 = x1^2 + x2^2 - 2 x1.x2
                X1_sq = np.sum(X1**2, axis=1)
                X2_sq = np.sum(X2**2, axis=1)
                dist_sq = X1_sq[:, None] + X2_sq[None, :] - 2 * np.dot(X1, X2.T)
                dist_sq = np.maximum(dist_sq, 0)
                K = self.sigma_f**2 * np.exp(-0.5 * dist_sq / self.sigma_l**2)
                return K

            def fit(self, X, Y):
                self.X_train = X
                self.Y_train = Y
                self.y_mean = np.mean(Y)
                self.y_std = np.std(Y) + 1e-8
                Y_norm = (Y - self.y_mean) / self.y_std
                
                K = self._rbf_kernel(X, X)
                K += self.alpha * np.eye(len(X))
                
                try:
                    self.K_inv = np.linalg.inv(K)
                except np.linalg.LinAlgError:
                    # Fallback if singular
                    self.K_inv = np.linalg.pinv(K)
                
                self.f_mean = np.dot(self.K_inv, Y_norm)
                self.is_trained = True

            def predict(self, X_new):
                if not self.is_trained:
                    return np.zeros(len(X_new)), np.ones(len(X_new))
                
                K_s = self._rbf_kernel(X_new, self.X_train)
                K_ss = self._rbf_kernel(X_new, X_new)
                
                # Mean
                mu_norm = np.dot(K_s, self.f_mean)
                mu = mu_norm * self.y_std + self.y_mean
                
                # Variance
                v = np.dot(K_s, self.K_inv)
                var_norm = np.diag(K_ss) - np.sum(v * K_s.T, axis=1)
                var = np.maximum(var_norm, 0) * self.y_std**2
                std = np.sqrt(var)
                
                return mu, std

        # EHVI Acquisition Function
        def ehvi_acquisition(X_candidates, Y_candidates, current_front, ref_point):
            """
            Calculate EHVI for a set of candidate points.
            current_front: list of non-dominated points (N x 2)
            ref_point: 2D array
            Returns array of EHVI values for each candidate.
            """
            n_cand = len(X_candidates)
            ehvi_vals = np.zeros(n_cand)
            
            # Current HV
            hv_current = compute_hypervolume(current_front, ref_point)
            
            # For each candidate, estimate the expected HV improvement
            # We approximate the distribution of Y at each candidate using the GP
            # Since we have 2 objectives, we have 2 GPs.
            # To keep it fast, we sample from the joint distribution?
            # Or use a deterministic approximation?
            # Given the tight budget and 200 evals, we can afford a Monte Carlo estimate
            # or a deterministic grid search in the objective space?
            # Let's use a simple Monte Carlo with 50 samples per candidate?
            # 50 * 200 * 100 candidates = 1M operations. Might be slow in Python loops.
            # Let's try a deterministic approach: 
            # Assume the mean of the GP is the best guess.
            # Then EHVI approx = max(0, HV(front U {mean}) - HV(front))
            # This is "Expected Improvement" but deterministic.
            # To be safer, we can sample 10 points from the GP distribution for each candidate.
            
            # Let's optimize:
            # 1. Get GP predictions for all candidates at once.
            # 2. For each candidate, sample M times from N(mu, sigma) for both objectives.
            # 3. Compute HV for each sample.
            # 4. Average improvement.
            
            M = 10 # Number of MC samples per candidate
            
            # Pre-compute GP means and stds for candidates
            # We need to train GPs for each objective first.
            
            pass # Placeholder, will be implemented in the loop below for clarity/speed tradeoff
            
            # Actually, training 2 GPs and predicting is fast.
            # The slow part is the HV calculation inside the MC loop.
            # Let's implement the MC loop efficiently.
            
            return ehvi_vals # Will be updated in the main loop logic

        # Main BO Loop
        n_init = len(X)
        n_bo_iters = max_evals - n_init
        
        # We will use a fixed set of candidate points for acquisition?
        # Or generate them dynamically?
        # Dynamic generation is better. We can use a Sobol sequence for candidates too.
        
        # Pre-generate a large set of candidate points from Sobol
        # We will reuse a subset of them or generate new ones each iter?
        # Generating new ones each iter is fine.
        
        for i in range(n_bo_iters):
            if time.time() > deadline - 5:
                break
            if eval_count >= max_evals:
                break
                
            # Get current Pareto front
            current_front = get_nondominated(Y)
            
            if len(current_front) == 0:
                # Fallback to random
                x_new = np.random.uniform(lo, hi)
                fx = safe_problem(x_new)
                if fx is not None:
                    X.append(x_new)
                    Y.append(fx)
                continue
            
            # Train GPs for each objective
            gp1 = GP2D()
            gp2 = GP2D()
            
            # Normalize inputs? GP is sensitive to scale.
            # Let's normalize X to [0,1] for GP training.
            X_norm = (X - lo) / (hi - lo)
            
            gp1.fit(X_norm, Y[:, 0])
            gp2.fit(X_norm, Y[:, 1])
            
            # Generate candidate points
            # Use 100 Sobol points
            n_cand = 100
            try:
                sampler = qmc.Sobol(d=self.dim, scramble=True)
                cand_samples = sampler.random(n_cand)
            except Exception:
                cand_samples = np.random.rand(n_cand, self.dim)
                
            X_cand = cand_samples * (hi - lo) + lo
            X_cand_norm = (X_cand - lo) / (hi - lo)
            
            # Predict means and stds
            mu1, std1 = gp1.predict(X_cand_norm)
            mu2, std2 = gp2.predict(X_cand_norm)
            
            # Calculate EHVI
            # We will use a Monte Carlo approach with M=5 samples for speed
            M = 5
            ehvi_scores = np.zeros(n_cand)
            
            # To speed up HV calculation, we can pre-compute the current front's contribution
            # But HV is not additive in a simple way.
            
            for j in range(n_cand):
                # Sample M points from the joint distribution
                # Assuming independence for simplicity (common in simple BO)
                samples1 = np.random.normal(mu1[j], std1[j], M)
                samples2 = np.random.normal(mu2[j], std2[j], M)
                
                hv_improvements = 0.0
                for k in range(M):
                    new_point = np.array([samples1[k], samples2[k]])
                    # Check if it's better than ref
                    if not np.all(new_point < ref_point):
                        continue
                    
                    # Add to front and compute HV
                    # Efficiently: Check if it dominates any existing front points?
                    # Or just compute HV of (front + new_point)
                    
                    # Simple HV calculation for 2D
                    # Front is sorted. New point might insert anywhere.
                    # Let's just concatenate and compute.
                    # For M=5 and 100 cands, this is 500 HV calcs.
                    # Each HV calc is O(N log N) or O(N^2) for dominance.
                    # With N < 50, it's fast.
                    
                    combined = np.vstack([current_front, new_point.reshape(1, -1)])
                    hv_new = compute_hypervolume(combined, ref_point)
                    hv_improvements += max(0, hv_new - hv_current)
                
                ehvi_scores[j] = hv_improvements / M
            
            # Select best candidate
            idx_best = np.argmax(ehvi_scores)
            x_new = X_cand[idx_best]
            
            # Evaluate
            fx = safe_problem(x_new)
            if fx is not None:
                X.append(x_new)
                Y.append(fx)
        
        # Final return
        # The evaluator records all calls to problem().
        # We just need to return something valid.
        # The contract says "return value of __call__ is ignored".
        # But we should return the best X or similar.
        return np.array(X)

# EVOLVE-BLOCK-END
