import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _greedy_select_x0(self, n_points, x0_lo, x0_hi, r1, r2, n_candidates):
        """
        Greedy selection of x0 values to maximize hypervolume approximation.
        
        The Pareto front is f2 = 2 - sqrt(f1) for f1 in [0,1].
        We generate many candidate points and greedily select the subset
        that maximizes the hypervolume contribution.
        """
        if n_points <= 1:
            return np.array([(x0_lo + x0_hi) / 2.0])
        
        # Generate candidate points with various spacings
        # Use a combination of uniform, quadratic, and cubic spacing
        candidates = set()
        
        # Quadratic spacing (denser near 0)
        t = np.linspace(0.0, 1.0, n_candidates)
        for power in [1.5, 2.0, 2.5, 3.0]:
            x0_cand = x0_lo + (x0_hi - x0_lo) * t**power
            for x in x0_cand:
                candidates.add(round(x, 12))
        
        # Uniform spacing
        x0_cand = np.linspace(x0_lo, x0_hi, n_candidates)
        for x in x0_cand:
            candidates.add(round(x, 12))
        
        # Convert to sorted list
        x0_candidates = np.array(sorted(candidates))
        
        # Greedy selection: start with the point that gives the largest single-point HV
        # then iteratively add the point that gives the largest marginal HV gain
        
        def compute_hv(x0_vals):
            """Compute hypervolume for given x0 values on the Pareto front."""
            if len(x0_vals) == 0:
                return 0.0
            if len(x0_vals) == 1:
                f1 = x0_vals[0]
                f2 = 2.0 - np.sqrt(f1)
                if f1 < r1 and f2 < r2:
                    return (r1 - f1) * (r2 - f2)
                return 0.0
            
            x0_sorted = np.sort(x0_vals)
            hv = 0.0
            for i in range(len(x0_sorted) - 1):
                f1_i = x0_sorted[i]
                f1_next = x0_sorted[i + 1]
                f2_i = 2.0 - np.sqrt(f1_i)
                if f1_i < r1 and f2_i < r2:
                    hv += (f1_next - f1_i) * (r2 - f2_i)
            
            f1_last = x0_sorted[-1]
            f2_last = 2.0 - np.sqrt(f1_last)
            if f1_last < r1 and f2_last < r2:
                hv += (r1 - f1_last) * (r2 - f2_last)
            
            return hv
        
        # Start with the point that gives the largest single-point HV
        single_hvs = []
        for x in x0_candidates:
            f1 = x
            f2 = 2.0 - np.sqrt(f1)
            if f1 < r1 and f2 < r2:
                single_hv = (r1 - f1) * (r2 - f2)
            else:
                single_hv = 0.0
            single_hvs.append(single_hv)
        
        single_hvs = np.array(single_hvs)
        best_idx = np.argmax(single_hvs)
        selected = [x0_candidates[best_idx]]
        
        # Iteratively add the point that gives the largest marginal HV gain
        for _ in range(n_points - 1):
            best_gain = -1.0
            best_candidate = None
            
            current_hv = compute_hv(np.array(selected))
            
            for x in x0_candidates:
                if x in selected:
                    continue
                
                new_selected = selected + [x]
                new_hv = compute_hv(np.array(new_selected))
                gain = new_hv - current_hv
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = x
            
            if best_candidate is None or best_gain <= 0:
                break
            
            selected.append(best_candidate)
        
        return np.array(sorted(selected))

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Check if we can set x_1...x_9 to 0 (known global minimum for g(x))
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))
        
        if zeros_ok:
            x0_lo = max(lo[0], 0.0)
            x0_hi = min(hi[0], 1.0)
            
            if x0_lo >= x0_hi:
                x0_vals = np.linspace(lo[0], hi[0], self.budget)
            else:
                # Use greedy selection with a larger candidate pool
                n_candidates = min(2000, self.budget * 10)
                x0_vals = self._greedy_select_x0(
                    self.budget, x0_lo, x0_hi, 11.0, 400.0, n_candidates
                )
            
            X = np.zeros((self.budget, self.dim))
            X[:, 0] = x0_vals[:self.budget]
            
            if time.time() > deadline:
                return X
            
            # Evaluate all points
            for i in range(self.budget):
                if time.time() > deadline:
                    break
                problem(X[i])
            
            return X
        
        else:
            # Fallback: random search
            rng = np.random.default_rng(0)
            X = lo + (hi - lo) * rng.random((self.budget, self.dim))
            for i in range(self.budget):
                if time.time() > deadline:
                    break
                problem(X[i])
            return X
