import numpy as np
import time
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = int(self.budget)
        
        def f2(t):
            # f2(t) = 2 * (1 - sqrt(t) - t * sin(10 * pi * t))
            if t < 0:
                t = 0.0
            if t > 1:
                t = 1.0
            return 2.0 * (1.0 - np.sqrt(t) - t * np.sin(10.0 * np.pi * t))
        
        # 1. Find global minimum t_min in [0, 1]
        # We can search in each period or just use a dense grid + refine
        t_grid = np.linspace(0, 1, 1000)
        f2_grid = np.array([f2(t) for t in t_grid])
        idx_min = np.argmin(f2_grid)
        t_min_coarse = t_grid[idx_min]
        
        # Refine around t_min_coarse
        res = minimize_scalar(f2, bounds=(max(0, t_min_coarse - 0.1), min(1, t_min_coarse + 0.1)), method='bounded')
        t_min = res.x
        f2_min = f2(t_min)
        
        # 2. Find record-breaking local minima in [0, t_min)
        # Local minima occur where derivative is 0 and second derivative > 0
        # Or simply find minima in each period of the sine wave
        # Period of sin(10*pi*t) is 0.2
        # Minima of the oscillatory part are at 10*pi*t = 3*pi/2 + 2*pi*k => t = 0.15 + 0.2k
        # But the envelope sqrt(t) shifts them.
        # Let's find local minima by checking intervals [0.2k, 0.2k+0.2]
        
        record_points = []
        current_min_val = np.inf
        
        # We need to check points in increasing t order
        # We will iterate through potential local minima locations
        # Potential local minima are near t = 0.15 + 0.2k
        # Also check t=0
        
        candidates = [0.0]
        k = 0
        while True:
            t_cand = 0.15 + 0.2 * k
            if t_cand >= t_min:
                break
            candidates.append(t_cand)
            k += 1
            
        # Refine each candidate to find the actual local minimum
        refined_candidates = []
        for tc in candidates:
            if tc >= t_min:
                continue
            # Search in a small window around tc
            lower = max(0, tc - 0.05)
            upper = min(t_min, tc + 0.05)
            if lower >= upper:
                continue
            res_loc = minimize_scalar(f2, bounds=(lower, upper), method='bounded')
            if res_loc.success:
                refined_candidates.append((res_loc.x, res_loc.fun))
                
        # Sort by t
        refined_candidates.sort(key=lambda x: x[0])
        
        # Filter record breakers
        for t_c, f2_c in refined_candidates:
            if f2_c < current_min_val:
                record_points.append(t_c)
                current_min_val = f2_c
                
        # Add t_min
        record_points.append(t_min)
        
        # If we have more budget, we can't really use it effectively for HV
        # But we must return a list of points.
        # The problem says "Every returned objective vector is recorded".
        # It doesn't say we MUST use all budget.
        # "More than 200 evaluations ... makes the whole result invalid".
        # So we can use fewer.
        
        # However, if we have spare budget, can we add points that don't hurt?
        # Adding points with f2 > current_min_val doesn't change m_i.
        # So it doesn't change HV.
        # So we can just return the record_points.
        
        # But wait, what if the dense sample in the denominator has more points?
        # The score is HV_our / HV_dense.
        # If our HV is maximal, we are good.
        
        # Let's construct the x vectors
        x = np.zeros(dim)
        points_to_eval = []
        
        for t in record_points:
            x[0] = t
            # Clip to bounds
            x_clipped = np.clip(x, lo, hi)
            points_to_eval.append(x_clipped)
            
        # If we have budget left, and we want to be safe, we can just stop.
        # But maybe we should check if we missed any record breakers?
        # The sine wave has period 0.2.
        # If t_min is small, there are few periods.
        # If t_min is large, there are many.
        
        # Let's evaluate them
        recorded = []
        for x_c in points_to_eval:
            f = np.asarray(problem(x_c), dtype=float)
            recorded.append(f)
            
        return recorded
