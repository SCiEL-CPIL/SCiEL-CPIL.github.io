import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim):
        """Generate n Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for i in range(1, n + 1):
            for d in range(dim):
                base = d + 2
                f = 1.0
                result = 0.0
                j = i
                while j > 0:
                    f /= base
                    result += f * (j % base)
                    j //= base
                points[i - 1, d] = result
        return points

    def _hypervolume_3d(self, points, ref):
        """
        Compute exact 3D hypervolume for a set of points relative to ref.
        Points must be strictly better than ref in all objectives to contribute.
        Uses slicing algorithm: O(N^2 log N) roughly.
        """
        if len(points) == 0:
            return 0.0
        
        # Filter points strictly better than reference
        valid = []
        for p in points:
            if np.all(p < ref):
                valid.append(p)
        
        if not valid:
            return 0.0
            
        valid = np.array(valid)
        n = len(valid)
        
        # Remove dominated points to speed up (optional but good practice)
        # Simple O(n^2) dominance check
        pareto_indices = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(valid[j] <= valid[i]) and np.any(valid[j] < valid[i]):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(i)
        
        if not pareto_indices:
            return 0.0
            
        valid = valid[pareto_indices]
        n = len(valid)
        
        # Sort by first objective (ascending)
        order = np.argsort(valid[:, 0])
        valid = valid[order]
        
        # Slicing:
        # HV = sum_{i=0}^{n-1} (x_{i+1} - x_i) * HV_2D(points[i:], ref_2, ref_3)
        # where x_{n} = ref[0]
        
        total_hv = 0.0
        ref_23 = ref[1:]
        
        for i in range(n):
            # Width in dimension 1
            x_next = valid[i+1, 0] if i < n - 1 else ref[0]
            width_1 = x_next - valid[i, 0]
            
            if width_1 <= 0:
                continue
                
            # Get the suffix of points from i to n-1
            suffix = valid[i:]
            
            # Compute 2D HV of suffix in dims 2 and 3 relative to ref[1], ref[2]
            hv_2d = self._hypervolume_2d(suffix[:, 1:], ref_23)
            
            total_hv += width_1 * hv_2d
            
        return total_hv

    def _hypervolume_2d(self, points, ref):
        """
        Compute exact 2D hypervolume for a set of points relative to ref.
        Points must be strictly better than ref.
        """
        if len(points) == 0:
            return 0.0
        
        # Filter
        valid = []
        for p in points:
            if np.all(p < ref):
                valid.append(p)
                
        if not valid:
            return 0.0
            
        valid = np.array(valid)
        
        # Remove dominated
        n = len(valid)
        pareto_indices = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(valid[j] <= valid[i]) and np.any(valid[j] < valid[i]):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(i)
                
        if not pareto_indices:
            return 0.0
            
        valid = valid[pareto_indices]
        
        # Sort by first dim (ascending)
        order = np.argsort(valid[:, 0])
        valid = valid[order]
        
        # Sweep line
        # Sort by x ascending.
        # Max y seen so far (from right to left? No, we want the area under the curve).
        # Actually, for minimization, the Pareto front is a decreasing curve.
        # Sort by x ascending. y should be descending for non-dominated points.
        # Area = sum (x_{i+1} - x_i) * y_i ? No.
        # Area is sum of rectangles.
        
        # Let's iterate from largest x to smallest x?
        # Or simply:
        # Sort by x ascending.
        # Keep track of min_y (which is the "front" in minimization).
        # Actually, for 2D HV with minimization:
        # Points are (x, y). Ref is (Rx, Ry).
        # Sort by x ascending.
        # The contribution of point i is (x_{i+1} - x_i) * (Ry - y_i) ?
        # No, because y_{i+1} might be smaller than y_i? No, if sorted by x asc and non-dominated, y must be desc.
        # So y_0 > y_1 > ...
        # The area is sum_{i} (x_{i+1} - x_i) * (Ry - y_i) ?
        # Let's check.
        # Box for point i: [x_i, x_{i+1}] x [y_i, Ry].
        # Yes, because for x in [x_i, x_{i+1}], the best y is y_i (since y is decreasing).
        
        total_hv = 0.0
        n = len(valid)
        for i in range(n):
            x_next = valid[i+1, 0] if i < n - 1 else ref[0]
            width = x_next - valid[i, 0]
            height = ref[1] - valid[i, 1]
            if width > 0 and height > 0:
                total_hv += width * height
                
        return total_hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n_budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        ref_point = np.array([1.1, 1.3, 1.1])
        
        X_all = []
        F_all = []
        eval_count = 0
        
        def evaluate(x):
            nonlocal eval_count
            if eval_count >= n_budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            X_all.append(x.copy())
            F_all.append(f.copy())
            eval_count += 1
            return f

        # Phase 1: Initialization
        # Use Sobol sequence for better coverage
        n_init = min(20, n_budget)
        try:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            X_init = sampler.random(n_init)
        except:
            X_init = self._halton(n_init, dim)
            
        X_init = lo + X_init * span
        
        for x in X_init:
            if time.time() > deadline:
                break
            evaluate(x)

        if eval_count >= n_budget:
            return np.array(X_all)

        # Phase 2: GP Surrogate Model-Based Optimization
        
        def matern_kernel(X, Y, length_scale=0.5, variance=1.0, nu=2.5):
            sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * X @ Y.T
            sq_dists = np.maximum(sq_dists, 0)
            dist = np.sqrt(sq_dists)
            if nu == 2.5:
                kernel = np.sqrt(3) * dist / length_scale
                return variance * (1 + kernel) * np.exp(-kernel)
            else:
                return variance * np.exp(-0.5 * dist**2 / length_scale**2)

        class SimpleGP:
            def __init__(self, length_scale=0.5, variance=1.0, nu=2.5):
                self.length_scale = length_scale
                self.variance = variance
                self.nu = nu
                self.X = None
                self.y = None
                self.K = None
                self.alpha = None

            def fit(self, X, y):
                self.X = X
                self.y = y
                self.K = matern_kernel(X, X, self.length_scale, self.variance, self.nu)
                self.K += 1e-6 * np.eye(len(X))
                try:
                    self.alpha = np.linalg.solve(self.K, y)
                except:
                    self.alpha = np.linalg.lstsq(self.K, y, rcond=None)[0]

            def predict_mean(self, X_new):
                if self.X is None:
                    return np.zeros(len(X_new))
                K_s = matern_kernel(X_new, self.X, self.length_scale, self.variance, self.nu)
                return K_s @ self.alpha

        # Initialize GP models
        X_elite = np.array(X_all)
        F_elite = np.array(F_all)
        
        gps = []
        for m in range(n_obj):
            gp = SimpleGP(length_scale=0.5, variance=1.0, nu=2.5)
            gp.fit(X_elite, F_elite[:, m])
            gps.append(gp)
            
        # Current Pareto Front (indices in X_all/F_all)
        # We will maintain a list of indices that form the non-dominated set
        def get_pareto_front_indices(X, F):
            n = len(F)
            if n == 0:
                return []
            fronts = []
            # Simple O(n^2) dominance
            nondom = [True] * n
            for i in range(n):
                if not nondom[i]:
                    continue
                for j in range(n):
                    if i == j or not nondom[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nondom[i] = False
                        break
            return [i for i in range(n) if nondom[i]]

        pareto_indices = get_pareto_front_indices(X_elite, F_elite)
        
        n_candidates = 100
        refit_interval = 5
        last_refit = eval_count
        
        while eval_count < n_budget and time.time() < deadline:
            # Generate candidates
            n_rand = n_candidates // 2
            n_local = n_candidates - n_rand
            
            candidates = []
            
            # Random
            candidates.append(lo + span * np.random.random((n_rand, dim)))
            
            # Local perturbations of current Pareto front
            if len(pareto_indices) > 0:
                # Pick a few from front
                sample_size = min(len(pareto_indices), 10)
                idxs = np.random.choice(len(pareto_indices), size=sample_size, replace=False)
                for idx in idxs:
                    real_idx = pareto_indices[idx]
                    x_base = X_elite[real_idx]
                    # Generate several perturbations
                    for _ in range(n_local // max(1, sample_size)):
                        pert = 0.1 * span * np.random.randn(dim)
                        candidates.append(x_base + pert)
            
            candidates = np.vstack(candidates)
            candidates = np.clip(candidates, lo, hi)
            
            # Predict means for all candidates
            means = np.zeros((len(candidates), n_obj))
            for m in range(n_obj):
                means[:, m] = gps[m].predict_mean(candidates)
                
            # Hypervolume-based selection
            # Current Pareto Front points
            current_pareto = F_elite[pareto_indices]
            
            best_candidate_idx = -1
            best_hv = -1.0
            
            # To save time, we can subsample candidates if n_candidates is large, 
            # but 100 is small. We calculate HV for each candidate added to the front.
            # Optimization: Pre-compute HV of current front.
            # HV_new = HV(current_pareto U {c})
            
            # We can just loop through candidates
            for k in range(len(candidates)):
                # Create new front: current_pareto + candidate
                # We need to check if candidate dominates any existing points?
                # For HV calculation, we just pass the set. The HV function handles dominance internally.
                new_front = np.vstack([current_pareto, means[k]])
                hv = self._hypervolume_3d(new_front, ref_point)
                
                if hv > best_hv:
                    best_hv = hv
                    best_candidate_idx = k
                    
            if best_candidate_idx == -1:
                break
                
            x_next = candidates[best_candidate_idx]
            f_next = evaluate(x_next)
            if f_next is None:
                break
            
            # Update elites
            X_elite = np.vstack([X_elite, x_next])
            F_elite = np.vstack([F_elite, f_next])
            
            # Update Pareto Front indices
            pareto_indices = get_pareto_front_indices(X_elite, F_elite)
            
            # Refit GP periodically
            if eval_count - last_refit >= refit_interval:
                for m in range(n_obj):
                    gps[m].fit(X_elite, F_elite[:, m])
                last_refit = eval_count

        # Phase 3: Final Polish
        while eval_count < n_budget and time.time() < deadline:
            x = lo + span * np.random.random(dim)
            evaluate(x)
            
        return np.array(X_all)
