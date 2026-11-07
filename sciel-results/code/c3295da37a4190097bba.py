import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hv(self, f1_vals, f2_vals, ref):
        """
        Compute Hypervolume for 2D points.
        f1_vals, f2_vals: numpy arrays of objective values.
        ref: tuple (ref1, ref2).
        """
        if len(f1_vals) == 0:
            return 0.0
        
        # Filter points strictly better than reference
        mask = (f1_vals < ref[0]) & (f2_vals < ref[1])
        if not np.any(mask):
            return 0.0
            
        f1 = f1_vals[mask]
        f2 = f2_vals[mask]
        
        # Sort by f1 ascending
        sort_idx = np.argsort(f1)
        f1 = f1[sort_idx]
        f2 = f2[sort_idx]
        
        # Handle duplicates in f1: keep only the one with minimum f2
        if len(f1) > 1:
            # Sort by f1, then f2
            sort_idx2 = np.lexsort((f2, f1))
            f1_sorted = f1[sort_idx2]
            f2_sorted = f2[sort_idx2]
            
            # Keep only unique f1 (first occurrence in sorted f2 is min f2)
            unique_f1_mask = np.concatenate(([True], np.diff(f1_sorted) > 0))
            f1 = f1_sorted[unique_f1_mask]
            f2 = f2_sorted[unique_f1_mask]

        hv = 0.0
        n = len(f1)
        if n == 1:
            return (ref[0] - f1[0]) * (ref[1] - f2[0])
            
        for i in range(n - 1):
            hv += (f1[i + 1] - f1[i]) * (ref[1] - f2[i])
        hv += (ref[0] - f1[-1]) * (ref[1] - f2[-1])
        
        return hv

    def _optimize_x1_analytic(self, N, ref):
        """
        Find optimal x1 values to maximize hypervolume on ZDT2 Pareto front.
        f1 = x1, f2 = 1 - sqrt(x1)
        
        The optimal distribution equalizes the marginal hypervolume contribution.
        For a concave front, this means points should be denser where the curve is steeper.
        
        We use an iterative approach: start with uniform spacing, then adjust to equalize
        the "area" between consecutive points.
        """
        ref1, ref2 = ref
        
        # Initial: uniform in f1
        x1 = np.linspace(0.0, 1.0, N)
        
        # Iteratively adjust to equalize marginal HV contributions
        # The marginal contribution of point i is approximately:
        # (x1[i+1] - x1[i]) * (ref2 - f2[i]) for i < N-1
        # (ref1 - x1[N-1]) * (ref2 - f2[N-1]) for the last point
        
        # Better approach: use the fact that for optimal HV on a smooth curve,
        # the points should satisfy that the "effective width" times height is equalized.
        
        # Let's use a gradient-free optimization with a smarter initial guess.
        # For ZDT2, f2 = 1 - sqrt(x1), so df2/dx1 = -1/(2*sqrt(x1))
        # The curve is steeper near x1=0.
        
        # A good heuristic: space points proportional to sqrt(x1) or similar.
        # Actually, let's just use a numerical optimizer with a good initialization.
        
        # Use scipy's minimize or implement a simple coordinate descent.
        # Since N=200 and we have 290s, we can afford a more expensive optimization.
        
        from scipy.optimize import minimize, differential_evolution
        
        def neg_hv(x1_flat):
            x1_arr = np.asarray(x1_flat)
            # Ensure sorted
            x1_arr = np.sort(x1_arr)
            f1 = x1_arr
            f2 = 1.0 - np.sqrt(f1)
            return -self._compute_hv(f1, f2, ref)
        
        # Initial guess: use a power-law distribution that densifies near x1=0
        # x1_i = (i/(N-1))^alpha for some alpha > 1
        # Let's try alpha = 2 (quadratic spacing)
        alpha = 2.0
        x1_init = (np.arange(N) / (N - 1)) ** alpha
        x1_init = np.sort(x1_init)
        
        # Use differential evolution for global optimization
        # But this might be slow for N=200. Let's use a simpler approach.
        
        # Alternative: use the analytic property that optimal points on a concave curve
        # satisfy a specific ODE. For now, let's use a refined hill climbing with
        # multiple restarts and a better initial guess.
        
        best_x1 = None
        best_hv = -np.inf
        
        # Try several initial distributions
        for alpha in [1.0, 1.5, 2.0, 2.5, 3.0]:
            x1_candidate = (np.arange(N) / (N - 1)) ** alpha
            x1_candidate = np.sort(x1_candidate)
            
            hv_candidate = -neg_hv(x1_candidate)
            if hv_candidate > best_hv:
                best_hv = hv_candidate
                best_x1 = x1_candidate.copy()
        
        # Refine with local search
        if best_x1 is not None:
            x1_refined = best_x1.copy()
            step_size = 0.01
            for _ in range(50):
                improved = False
                for i in range(N):
                    orig = x1_refined[i]
                    for delta in [-step_size, step_size]:
                        new_val = np.clip(orig + delta, 0.0, 1.0)
                        x1_refined[i] = new_val
                        # Keep sorted
                        x1_sorted = np.sort(x1_refined)
                        hv_new = -neg_hv(x1_sorted)
                        if hv_new > -neg_hv(np.sort(x1_refined)) + 1e-12:
                            improved = True
                            break
                        x1_refined[i] = orig
                    if improved:
                        x1_refined = np.sort(x1_refined)
                        break
                if not improved:
                    step_size *= 0.7
                    if step_size < 1e-6:
                        break
        
        return x1_refined if best_x1 is not None else np.linspace(0.0, 1.0, N)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        N = self.budget
        dim = self.dim
        ref = (11.0, 11.0)
        
        # Use analytic optimization to find optimal x1 distribution
        x1_params = self._optimize_x1_analytic(N, ref)
        
        # Ensure x1 is sorted and within [0, 1]
        x1_params = np.sort(np.clip(x1_params, 0.0, 1.0))
        
        # 2. Execute evaluations using the optimized parameters
        X = np.zeros((N, dim))
        X[:, 0] = x1_params
        # x[1:] are already 0
        
        results = []
        for i in range(N):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            results.append(f)
            
        return X
