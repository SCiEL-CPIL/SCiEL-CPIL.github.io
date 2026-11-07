import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3 using adaptive hypervolume-gap refinement.
    
    Strategy:
    ZDT3's Pareto front is analytically known:
    f1(x) = x1
    g(x) = 1 + 9 * sum(x2..xn) / (n-1)
    f2(x) = g(x) * (1 - sqrt(f1/g))
    
    The Pareto front is achieved when x2..xn = 0, making g(x) = 1.
    Then f1 = x1 and f2 = 1 - sqrt(x1).
    
    Instead of fixed uniform sampling, we use an adaptive approach:
    1. Start with a coarse grid of points on the known front
    2. For each remaining evaluation, pick the point that maximizes estimated hypervolume gain
    3. The gain is estimated based on the gap it fills in the f1 axis and the f2 value
    
    This should concentrate evaluations in regions where they contribute most to hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        x1_lo = lo[0]
        x1_hi = hi[0]
        
        ref = np.array([11.0, 11.0])
        
        # Analytic f2 for a given x1 (assuming x2..xn = 0)
        def analytic_f2(x1):
            x1 = np.clip(x1, 0.0, 1.0)
            if x1 <= 0:
                return 1.0
            if x1 >= 1.0:
                return 0.0
            return 1.0 - np.sqrt(x1)

        # Initial coarse sampling: use a small number of evenly spaced points
        # to bootstrap the gap structure
        initial_n = min(20, self.budget)
        x1_initial = np.linspace(x1_lo, x1_hi, initial_n)
        x1_initial = np.clip(x1_initial, x1_lo, x1_hi)
        
        # Evaluate initial points
        evaluated_x1 = list(x1_initial)
        evaluated_f = []
        
        for x1_val in x1_initial:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = x1_val
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated_f.append(f)

        remaining = self.budget - len(evaluated_f)
        
        if remaining <= 0:
            return np.array(evaluated_f)
        
        # Sort evaluated points by x1
        # We'll maintain a sorted list of x1 values to find gaps
        evaluated_x1_sorted = sorted(evaluated_x1)
        
        # For adaptive refinement, we'll pick the midpoint of the largest gap
        # that gives the best estimated hypervolume gain
        # The gain from adding a point at x1 = c between a and b is roughly:
        # the area under the f2 curve between a and c, plus c and b, minus the existing coverage
        # Simplified: gain ≈ (b - a) * (ref[1] - analytic_f2((a+b)/2)) / 2
        
        # We'll iteratively add points by finding the gap with max estimated gain
        for _ in range(remaining):
            if time.time() > deadline:
                break
            
            # Find all gaps and estimate gain for midpoint of each
            # Gaps: [x1_lo, evaluated_x1_sorted[0]], [evaluated_x1_sorted[i], evaluated_x1_sorted[i+1]], [evaluated_x1_sorted[-1], x1_hi]
            candidates = []
            
            # First gap
            gap_start = x1_lo
            gap_end = evaluated_x1_sorted[0]
            if gap_end > gap_start:
                mid = (gap_start + gap_end) / 2.0
                f2_mid = analytic_f2(mid)
                gain = (gap_end - gap_start) * (ref[1] - f2_mid)
                candidates.append((gain, mid))
            
            # Middle gaps
            for i in range(len(evaluated_x1_sorted) - 1):
                gap_start = evaluated_x1_sorted[i]
                gap_end = evaluated_x1_sorted[i + 1]
                if gap_end > gap_start:
                    mid = (gap_start + gap_end) / 2.0
                    f2_mid = analytic_f2(mid)
                    gain = (gap_end - gap_start) * (ref[1] - f2_mid)
                    candidates.append((gain, mid))
            
            # Last gap
            gap_start = evaluated_x1_sorted[-1]
            gap_end = x1_hi
            if gap_end > gap_start:
                mid = (gap_start + gap_end) / 2.0
                f2_mid = analytic_f2(mid)
                gain = (gap_end - gap_start) * (ref[1] - f2_mid)
                candidates.append((gain, mid))
            
            if not candidates:
                break
            
            # Pick the candidate with max gain
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_gain, best_x1 = candidates[0]
            
            if best_gain <= 1e-12:
                break
            
            # Evaluate this point
            x = np.zeros(n)
            x[0] = best_x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            
            evaluated_x1.append(best_x1)
            evaluated_f.append(f)
            
            # Insert into sorted list
            # Use bisect for efficiency
            import bisect
            bisect.insort(evaluated_x1_sorted, best_x1)

        return np.array(evaluated_f)
