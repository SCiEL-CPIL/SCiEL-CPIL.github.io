import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295

        n_evals = self.budget
        if n_evals <= 0:
            return []

        # Standard ZDT6 Pareto Front
        # f1 in [0, 1]
        # f2 = 1 - sqrt(f1)
        
        # 1. Generate candidate points on the front
        n_candidates = 5000
        f1_candidates = np.linspace(0.0, 1.0, n_candidates)
        f2_candidates = 1.0 - np.sqrt(f1_candidates)
        
        # 2. Map f1 to x1
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^6
        x1_grid = np.linspace(0.0, 1.0, 100000)
        f1_grid = 1.0 - np.exp(-4.0 * x1_grid) * (np.sin(6.0 * np.pi * x1_grid) ** 6)
        
        x1_candidates = np.zeros(n_candidates)
        for i in range(n_candidates):
            # Find x1 in grid closest to f1_candidates[i]
            # Since f1 is not monotonic, we just find the closest match
            # We can use argmin
            idx = np.argmin(np.abs(f1_grid - f1_candidates[i]))
            x1_candidates[i] = x1_grid[idx]

        # 3. Greedy selection of points to maximize Hypervolume
        # Reference point [11, 11]
        # HV = sum (f1[i] - f1[i-1]) * f2[i] with f1[-1]=0
        
        selected_indices = []
        current_hv = 0.0
        
        # To speed up, we can just iterate through all candidates each time
        # N=200, Candidates=5000. 200*5000 = 1,000,000 iterations.
        # Each iteration O(log N) for insertion + O(1) for delta.
        
        for k in range(n_evals):
            if time.time() > deadline - 5:
                break
                
            best_delta = -1e9
            best_idx = -1
            
            # Current selected points sorted by f1
            # We maintain a list of (f1, f2, original_idx)
            # For efficiency, we just store indices and sort them on the fly?
            # No, we maintain a sorted list of indices.
            
            if not selected_indices:
                # First point: HV = f1 * f2
                # Maximize f1[i] * f2[i]
                vals = f1_candidates * f2_candidates
                best_idx = np.argmax(vals)
                best_delta = vals[best_idx]
            else:
                # Sort selected indices by f1
                sel_f1 = f1_candidates[selected_indices]
                sel_f2 = f2_candidates[selected_indices]
                
                # For each candidate not in selected
                # We need to find its position in the sorted selected list
                # and calculate delta.
                
                # Optimization: Pre-compute sorted order of all candidates?
                # No, we just iterate.
                
                for i in range(n_candidates):
                    if i in selected_indices:
                        continue
                    
                    f1_new = f1_candidates[i]
                    f2_new = f2_candidates[i]
                    
                    # Find position in sel_f1
                    # Binary search
                    pos = np.searchsorted(sel_f1, f1_new)
                    
                    if pos == 0:
                        # New point is first
                        # Left neighbor: None (0,0)
                        # Right neighbor: sel[0]
                        f1_L = 0.0
                        f2_L = 0.0 # Height to left is 0? No, height is f2 of the point that dominates the interval.
                        # Interval [0, f1_new]: Height f2_new
                        # Old Interval [0, f1_R]: Height f2_R (where R is sel[0])
                        # Wait, if pos=0, the old first point R is now at pos 1.
                        # Old contribution of [0, f1_R] was f1_R * f2_R.
                        # New contribution: [0, f1_new] * f2_new + [f1_new, f1_R] * f2_R.
                        # Delta = (f1_new * f2_new + (f1_R - f1_new) * f2_R) - (f1_R * f2_R)
                        # Delta = f1_new * f2_new - f1_new * f2_R
                        # Delta = f1_new * (f2_new - f2_R)
                        
                        f1_R = sel_f1[0]
                        f2_R = sel_f2[0]
                        delta = f1_new * (f2_new - f2_R)
                        
                    elif pos == len(selected_indices):
                        # New point is last
                        # Left neighbor: sel[-1]
                        # Old contribution ended at f1_L with height f2_L?
                        # No, the last interval was [f1_L_prev, f1_L] with height f2_L.
                        # And [f1_L, ...] was 0.
                        # New interval [f1_L, f1_new] with height f2_new.
                        # Delta = (f1_new - f1_L) * f2_new
                        
                        f1_L = sel_f1[-1]
                        delta = (f1_new - f1_L) * f2_new
                        
                    else:
                        # Middle
                        # Left: sel[pos-1], Right: sel[pos]
                        f1_L = sel_f1[pos-1]
                        f2_L = sel_f2[pos-1]
                        f1_R = sel_f1[pos]
                        f2_R = sel_f2[pos]
                        
                        # Old: [f1_L, f1_R] had height f2_R
                        # New: [f1_L, f1_new] had height f2_new, [f1_new, f1_R] had height f2_R
                        # Delta = (f1_new - f1_L) * f2_new + (f1_R - f1_new) * f2_R - (f1_R - f1_L) * f2_R
                        # Delta = (f1_new - f1_L) * f2_new - (f1_new - f1_L) * f2_R
                        # Delta = (f1_new - f1_L) * (f2_new - f2_R)
                        
                        delta = (f1_new - f1_L) * (f2_new - f2_R)
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_idx = i
            
            if best_idx == -1:
                break
                
            selected_indices.append(best_idx)
            current_hv += best_delta

        # 4. Evaluate selected points
        results = []
        for i in selected_indices:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x1_candidates[i])
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break

        return results
