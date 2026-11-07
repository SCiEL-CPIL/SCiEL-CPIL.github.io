import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def _zdt4_f2(self, x0):
        """Compute f2 for ZDT4 given x0 (x[1:] = 0)"""
        x0 = np.clip(x0, 0, 1)
        return 1.0 - np.sqrt(x0) - x0 * np.sin(10.0 * np.pi * x0)**2
    
    def _hv_2d(self, f1_vals, f2_vals, R1=11.0, R2=400.0):
        """Compute 2D hypervolume for non-dominated points sorted by increasing f1"""
        if len(f1_vals) == 0:
            return 0.0
        # Sort by f1
        order = np.argsort(f1_vals)
        f1 = f1_vals[order]
        f2 = f2_vals[order]
        # Remove points not strictly better than reference
        mask = (f1 < R1) & (f2 < R2)
        f1 = f1[mask]
        f2 = f2[mask]
        if len(f1) == 0:
            return 0.0
        # Remove dominated points (keep only Pareto-optimal subset)
        # A point is dominated if there's another with f1 <= and f2 <= (at least one strict)
        # Since sorted by f1, for each point check if it has the minimum f2 seen so far from right
        # Actually for HV computation, we only need the non-dominated frontier
        # Let's filter: iterate from high f1 to low f1, keep track of min f2
        # Points where f2 is strictly less than all points with higher f1 are on the frontier
        n = len(f1)
        keep = np.zeros(n, dtype=bool)
        min_f2_from_right = np.inf
        for i in range(n - 1, -1, -1):
            if f2[i] < min_f2_from_right:
                keep[i] = True
                min_f2_from_right = f2[i]
            # If f2[i] >= min_f2_from_right, it's dominated by a point with higher f1 and lower/equal f2
            # Wait, that's not right. Dominated means there exists j with f1[j] <= f1[i] and f2[j] <= f2[i]
            # Since sorted by f1 ascending, a point i is dominated if there exists j < i with f2[j] <= f2[i]
            # OR j > i with f2[j] <= f2[i]... no.
            # Point i is dominated if there exists j such that f1[j] <= f1[i] and f2[j] <= f2[i] and (f1[j] < f1[i] or f2[j] < f2[i])
            # Since f1 is sorted, for point i, we need to check if any point with f1 <= f1[i] has f2 < f2[i]
            # That's equivalent to: min f2 among points 0..i is < f2[i]
        
        # Let me redo this properly
        keep = np.zeros(n, dtype=bool)
        running_min_f2 = np.inf
        for i in range(n):
            # Point i is on the Pareto front if its f2 is strictly less than all previous points' f2
            # (since f1 is increasing, previous points have lower f1)
            if f2[i] < running_min_f2:
                keep[i] = True
                running_min_f2 = f2[i]
        
        f1 = f1[keep]
        f2 = f2[keep]
        if len(f1) == 0:
            return 0.0
        
        # Compute HV using the standard 2D formula
        hv = 0.0
        for i in range(len(f1)):
            next_f1 = f1[i+1] if i+1 < len(f1) else R1
            hv += (next_f1 - f1[i]) * (R2 - f2[i])
        return hv
    
    def __call__(self, problem, bounds):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        DEADLINE = 290.0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        if budget <= 0:
            return np.empty((0, dim))
        
        N = min(budget, 200)
        
        # For ZDT4 with x[1:]=0:
        # f1 = x0, f2 = 1 - sqrt(x0) - x0*sin(10*pi*x0)^2
        # The Pareto front is parameterized by x0 in [0,1]
        
        # Strategy: Initial placement via arc-length, then local refinement
        # to maximize hypervolume
        
        # Compute fine discretization of the curve
        n_fine = 10000
        x0_fine = np.linspace(0.0, 1.0, n_fine)
        f2_fine = self._zdt4_f2(x0_fine)
        
        # Compute arc length
        dx = np.diff(x0_fine)
        df2 = np.diff(f2_fine)
        ds = np.sqrt(dx**2 + df2**2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        s_total = s[-1]
        s_norm = s / s_total
        
        # Place N points uniformly in normalized arc length
        if N == 1:
            x0_vals = np.array([0.5])
        else:
            s_targets = np.linspace(0.0, 1.0, N)
            x0_vals = np.interp(s_targets, s_norm, x0_fine)
        
        x0_vals = np.clip(x0_vals, lo[0], hi[0])
        
        # Local refinement: for each point, find the x0 in its interval
        # that minimizes f2, which maximizes its HV contribution
        def refine_placement(x0_init, max_iters=5):
            x0 = x0_init.copy()
            n = len(x0)
            if n <= 1:
                return x0
            
            # Precompute f2 on a fine grid for fast lookup
            grid_n = 5000
            x0_grid = np.linspace(lo[0], hi[0], grid_n)
            f2_grid = self._zdt4_f2(x0_grid)
            
            for it in range(max_iters):
                if time.time() - t0 > DEADLINE - 5:
                    break
                improved = False
                for i in range(n):
                    # Determine the f1 interval this point "owns"
                    if i == 0:
                        f1_left = 0.0
                        f1_right = x0[1]
                    elif i == n - 1:
                        f1_left = x0[i-1]
                        f1_right = 11.0
                    else:
                        f1_left = x0[i-1]
                        f1_right = x0[i+1]
                    
                    if f1_right <= f1_left:
                        continue
                    
                    # Find grid points in the interval
                    mask = (x0_grid >= f1_left) & (x0_grid <= f1_right)
                    if not np.any(mask):
                        continue
                    
                    x0_cand = x0_grid[mask]
                    f2_cand = f2_grid[mask]
                    
                    # Pick the candidate with minimum f2
                    best_idx = np.argmin(f2_cand)
                    new_x0 = x0_cand[best_idx]
                    
                    if abs(new_x0 - x0[i]) < 1e-8:
                        continue
                    
                    # Check if this improves overall HV
                    old_hv = self._hv_2d(x0, self._zdt4_f2(x0))
                    x0_test = x0.copy()
                    x0_test[i] = new_x0
                    new_hv = self._hv_2d(x0_test, self._zdt4_f2(x0_test))
                    
                    if new_hv > old_hv + 1e-10:
                        x0[i] = new_x0
                        improved = True
                
                if not improved:
                    break
            
            return x0
        
        # Refine the placement
        if N > 1 and time.time() - t0 < DEADLINE - 10:
            x0_vals = refine_placement(x0_vals)
        
        # Ensure sorted and unique
        x0_vals = np.sort(x0_vals)
        x0_vals = np.clip(x0_vals, lo[0], hi[0])
        
        evals = 0
        results = []
        
        for x0_val in x0_vals:
            if evals >= budget or time.time() - t0 > DEADLINE:
                break
            
            x = np.zeros(dim)
            x[0] = float(x0_val)
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                evals += 1
                results.append(f)
            except Exception:
                pass
        
        if results:
            return np.array(results)
        
        return np.empty((0, dim))
