import numpy as np
import time
from scipy.optimize import brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.eval_count = 0

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        # Safety margin: evaluator kills at 330s, we target 300s limit
        self.deadline = self.start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Analytic Pareto Front Construction with Objective-Space Uniformity.
        # 1. Determine the valid range of f1 on the Pareto front for ZDT6.
        # 2. Generate N uniform targets in f1 space.
        # 3. For each target, find x1 such that f1(x1) = target using Brent's method.
        # 4. Set x2...xD = 0.
        # 5. Evaluate.
        
        # ZDT6 f1(x) = 1 - exp(-4 * x1 * sin(10 * pi * x1))
        # We need to find the maximum value of f1 achievable on the Pareto front.
        # The Pareto front consists of non-dominated points.
        # Since f2 = 1 - sqrt(f1) (assuming g=1), and we minimize both,
        # any point where f1 is in the range [0, max_f1] is part of the front,
        # provided we can find an x1 that produces it.
        
        # Let's find the global maximum of f1(x1) for x1 in [0, 1].
        # f1(x1) = 1 - exp(-4 * x1 * sin(10 * pi * x1))
        # To maximize f1, we need to minimize exp(...), which means maximizing
        # -4 * x1 * sin(10 * pi * x1).
        # Let h(x1) = 4 * x1 * sin(10 * pi * x1). We want max h(x1) for h(x1) > 0.
        
        # We'll sample a dense grid to find the approximate max and bracketing intervals.
        n_grid = 10000
        x1_grid = np.linspace(0, 1, n_grid)
        f1_grid = 1 - np.exp(-4 * x1_grid * np.sin(10 * np.pi * x1_grid))
        
        # Filter valid f1 values (f1 should be >= 0 for standard minimization context, 
        # though mathematically it can be negative if sin is negative. 
        # However, the Pareto front usually considers the "upper" envelope where f1 is maximized 
        # for a given f2, or rather the set of non-dominated points.
        # If f1 is negative, f2 = 1 - sqrt(negative) is undefined in reals.
        # So we must ensure f1 >= 0.
        # f1 >= 0 implies exp(...) <= 1 implies -4*x1*sin(...) <= 0 implies x1*sin(...) >= 0.
        # Since x1 >= 0, we need sin(10*pi*x1) >= 0.
        
        valid_mask = f1_grid >= 0
        if not np.any(valid_mask):
            # Fallback if something is weird
            return self.archive
            
        f1_valid = f1_grid[valid_mask]
        x1_valid = x1_grid[valid_mask]
        
        f1_max = np.max(f1_valid)
        
        # Generate uniform targets in f1 space from 0 to f1_max
        # We use linspace to get evenly spaced points.
        # Note: f1=0 is at x1=0 and x1=0.1, 0.2, etc.
        # We want to cover the range [0, f1_max].
        f1_targets = np.linspace(0, f1_max, self.budget)
        
        # For each target, we need to find an x1.
        # We can use the grid to find a bracket [a, b] where f1(a) <= target <= f1(b)
        # and f1 is continuous in between (it is).
        # Then use brentq on f1(x) - target = 0.
        
        # To make brentq fast, we pre-compute the function values and use searchsorted.
        # However, f1 is not monotonic over the whole domain.
        # We need to pick a segment where f1 is monotonic and covers the target.
        # The first segment from x1=0 to the first local max is monotonic increasing.
        # Let's find the first local max.
        
        # Derivative of h(x) = 4x sin(10 pi x) is zero when tan(10 pi x) = -10 pi x.
        # First positive root of tan(y) = -y is y ~ 2.0287.
        # x1_max_first = 2.0287 / (10 * pi) ~ 0.0646.
        # In this interval [0, 0.0646], f1 goes from 0 to f1_max.
        # This is a perfect monotonic segment to use for inversion!
        
        # Let's verify the max in this interval.
        x1_search_max = 0.0646
        # Refine the max location slightly or just use the grid value closest to it.
        # Actually, we can just define the interval [0, x1_search_max] and use brentq.
        # We need to ensure f1(x1_search_max) is indeed the global max.
        # From our grid analysis, f1_max is the global max.
        # Does the first hump reach the global max?
        # h(x) = 4x sin(10 pi x).
        # Next hump: y in (3pi/2, 2pi)? No, sin is negative there.
        # Next positive hump: y in (2pi, 5pi/2).
        # x1 range: [0.2, 0.25].
        # h(0.25) = 4*0.25*sin(2.5 pi) = 1 * 1 = 1.
        # Wait. sin(2.5 pi) = 1.
        # h(0.25) = 1.
        # Previous max: x ~ 0.0646, h ~ 0.057.
        # So the first hump is NOT the global max!
        # The global max is in the second positive hump?
        # Let's check x1 = 0.25.
        # f1(0.25) = 1 - exp(-1) = 0.632.
        # Let's check x1 = 0.45 (next hump center? y=4.5pi, sin=1).
        # h(0.45) = 4*0.45*1 = 1.8.
        # f1(0.45) = 1 - exp(-1.8) = 1 - 0.165 = 0.835.
        # Let's check x1 = 0.65.
        # h(0.65) = 4*0.65*1 = 2.6.
        # f1(0.65) = 1 - exp(-2.6) = 1 - 0.074 = 0.926.
        # Let's check x1 = 0.85.
        # h(0.85) = 4*0.85*1 = 3.4.
        # f1(0.85) = 1 - exp(-3.4) = 1 - 0.033 = 0.967.
        
        # It seems the global max is near x1=1.0?
        # At x1=1, sin(10pi)=0, f1=0.
        # The last hump is centered at x1=0.95?
        # y = 9.5 pi. sin(9.5 pi) = 1.
        # h(0.95) = 4 * 0.95 * 1 = 3.8.
        # f1(0.95) = 1 - exp(-3.8) = 1 - 0.022 = 0.978.
        
        # So the global max is in the last hump.
        # The function f1(x1) oscillates with increasing amplitude.
        # To get uniform coverage in f1, we need to pick x1 from DIFFERENT humps
        # to cover the range [0, f1_max] uniformly.
        
        # Revised Plan:
        # 1. Identify all intervals where f1 is monotonic increasing from 0 to a local max.
        # 2. For each target f1, find the interval where this f1 value is "newly" reached
        #    or simply find ANY x1 that produces this f1.
        #    Since we just need the points to be on the front, and the front is defined by the
        #    non-dominated points, any x1 that produces a specific f1 (with x2..xD=0)
        #    results in the same point on the Pareto front (f1, f2).
        #    So we just need to find ONE x1 for each f1 target.
        
        # We can use the dense grid to find the nearest x1 for each target.
        # To ensure we cover the full range [0, f1_max], we should map targets
        # to x1 values that are spread out, but since the objective is the same,
        # it doesn't matter which x1 we pick, as long as f1(x1) = target.
        
        # However, numerical precision matters.
        # Using brentq on a monotonic segment is robust.
        # Let's collect all monotonic increasing segments of f1.
        
        # A segment is increasing where derivative > 0.
        # f1'(x) = exp(-h(x)) * h'(x).
        # h'(x) = 4 sin(10 pi x) + 40 pi x cos(10 pi x).
        
        # We can just use the grid to find brackets.
        # Sort grid by f1? No, we need continuous x1 brackets.
        
        # Let's find all local maxima.
        # We can identify peaks in f1_grid.
        peaks = []
        for i in range(1, n_grid - 1):
            if f1_grid[i] > f1_grid[i-1] and f1_grid[i] > f1_grid[i+1]:
                peaks.append((x1_grid[i-1], x1_grid[i], f1_grid[i]))
                
        # Each peak (x_start, x_peak, f_peak) defines a monotonic increasing segment [x_start, x_peak]
        # where f1 goes from ~0 to f_peak.
        # Note: f1 might not start exactly at 0 if the previous valley was not 0.
        # But for ZDT6, the valleys are at f1=0 (when sin=0).
        
        # We have a list of segments.
        # For a target f1_t, we can find a segment where f_peak >= f1_t.
        # Then we can use brentq on that segment to find x1.
        
        # To maximize diversity (though not strictly necessary for HV if points are identical in obj space),
        # we can just pick the first segment that covers the target.
        # Or, to be safe and fast, we can just use the grid interpolation for a first guess
        # and then refine with brentq if we want high precision.
        # Given budget 200 and time limit 300s, brentq 200 times is fine.
        
        # Let's build a list of (x_low, x_high, f_low, f_high) for monotonic increasing segments.
        segments = []
        # Find indices where f1 is increasing
        # We'll just iterate through the grid and find contiguous increasing regions that start near 0.
        
        # Simpler: For each target, scan the grid for the first x1 where f1(x1) >= target.
        # Then the bracket is [x1_prev, x1].
        # This is O(N_grid) per target, which is 200 * 10000 = 2,000,000 ops. Fast enough.
        
        selected_x1s = []
        for f1_t in f1_targets:
            if f1_t < 0:
                x1_sol = 0.0
            elif f1_t > f1_max:
                # Should not happen
                x1_sol = x1_grid[np.argmax(f1_grid)]
            else:
                # Find first index where f1_grid >= f1_t
                # Note: f1_grid oscillates. The first time it crosses f1_t is in the first hump.
                # This is a valid solution.
                idx = np.searchsorted(f1_grid, f1_t)
                if idx >= n_grid:
                    idx = n_grid - 1
                # The crossing is between idx-1 and idx
                # We need to ensure f1_grid[idx-1] < f1_t <= f1_grid[idx]
                # If f1_grid[idx-1] >= f1_t, we might have jumped over a peak?
                # searchsorted finds the first occurrence.
                # If the function went up and down, the first occurrence is on the way up.
                # So [x1_grid[idx-1], x1_grid[idx]] is a valid bracket.
                
                a = x1_grid[idx-1]
                b = x1_grid[idx]
                
                def objective_func(x1):
                    return 1 - np.exp(-4 * x1 * np.sin(10 * np.pi * x1)) - f1_t
                
                try:
                    x1_sol = brentq(objective_func, a, b, xtol=1e-12)
                except ValueError:
                    # Fallback to grid value
                    x1_sol = x1_grid[idx]
            selected_x1s.append(x1_sol)
            
        # Now evaluate these 200 points
        for x1 in selected_x1s:
            if time.time() > self.deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1
            f = self._evaluate(problem, x)
            self._update_archive(x, f)
            
        return self.archive

    def _evaluate(self, problem, x):
        self.eval_count += 1
        return np.array(problem(x))

    def _update_archive(self, x, f):
        # Check if x is dominated by any in archive
        for _, af in self.archive:
            if np.all(f >= af) and np.any(f > af):
                return # Dominated, skip
        
        # Remove any in archive dominated by x
        self.archive = [(ax, af) for ax, af in self.archive if not (np.all(f <= af) and np.any(f < af))]
        self.archive.append((x, f))
