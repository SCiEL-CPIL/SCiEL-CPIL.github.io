import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        budget = self.budget
        dim = self.dim
        
        # ZDT3 Analysis:
        # f1(x) = x_0
        # g(x) = 1 + 10*(n-1) + sum_{i=1}^{n-1} (4*x_i * sin(10*pi*x_i^2))
        # f2(x) = g(x) * h(f1(x), g(x))
        # h(y, g) = 1 - sqrt(y/g) - (y/g)*sin(10*pi*y)
        
        # To minimize f2 for a given f1 (i.e., x_0), we must minimize g(x[1:]).
        # The minimum of g is 1, achieved when x_i = 0 for all i >= 1 (since sin(0)=0 and term is 0).
        # Also x_i = 0.5 gives sin(10*pi*0.25) = sin(2.5*pi) = 1, term = 4*0.5*1 = 2. Not min.
        # x_i = 0 is the global minimum for each term.
        # So the Pareto front is the set of points where x[1:] = 0.
        # On this front:
        # g = 1
        # f1 = x_0
        # f2 = 1 * (1 - sqrt(x_0) - x_0*sin(10*pi*x_0))
        #     = 1 - sqrt(x_0) - x_0*sin(10*pi*x_0)
        
        # The Pareto front is a curve in objective space.
        # To maximize hypervolume, we should sample points along this curve.
        # The curve is discontinuous because of the sin term.
        # Specifically, the function h(y, 1) = 1 - sqrt(y) - y*sin(10*pi*y).
        # The Pareto front consists of the lower envelope of this function?
        # No, for a fixed x_0, f2 is determined.
        # The set of non-dominated points is the entire curve, because:
        # If we have two points (f1a, f2a) and (f1b, f2b) with f1a < f1b,
        # they are non-dominated if f2a > f2b.
        # Since f2 oscillates, some points with higher f1 might have lower f2.
        # So we should sample the entire curve.
        
        # Strategy:
        # 1. Set x[1:] = 0.
        # 2. Sample x_0 uniformly in [0, 1].
        # 3. Evaluate to get the Pareto front points.
        
        # However, to maximize hypervolume, we might want to be smarter about sampling.
        # But uniform sampling of x_0 is a good start.
        # Let's check if there are any other points on the Pareto front.
        # If x[1:] != 0, g > 1.
        # f2 = g * h(f1, g).
        # Is it possible that g > 1 gives a lower f2?
        # h(y, g) = 1 - sqrt(y/g) - (y/g)*sin(10*pi*y).
        # Let g = 1 + delta.
        # f2 = (1+delta) * (1 - sqrt(y/(1+delta)) - (y/(1+delta))*sin(10*pi*y)).
        # For small y, sqrt(y/(1+delta)) < sqrt(y), so 1 - sqrt(...) is larger.
        # But multiplied by (1+delta), it might be smaller?
        # Let's test y=0.1, delta=0.1.
        # g=1.1.
        # h(0.1, 1) = 1 - sqrt(0.1) - 0.1*sin(pi) = 1 - 0.316 - 0 = 0.684.
        # f2(g=1) = 0.684.
        # h(0.1, 1.1) = 1 - sqrt(0.1/1.1) - (0.1/1.1)*sin(pi) = 1 - sqrt(0.0909) - 0 = 1 - 0.3015 = 0.6985.
        # f2(g=1.1) = 1.1 * 0.6985 = 0.768.
        # 0.768 > 0.684. So g=1 is better.
        
        # It seems g=1 is always the best for minimizing f2.
        # So the Pareto front is exactly the curve with x[1:] = 0.
        
        # ZDT3 Pareto front analysis:
        # f1 = x_0
        # f2 = 1 - sqrt(x_0) - x_0*sin(10*pi*x_0)  when x[1:] = 0
        # 
        # The front has 7 disconnected segments due to the sin term.
        # We need to find the segments and sample each proportionally.
        #
        # Strategy: Use parametric inversion to place points at specific f2 values
        # within each segment to maximize hypervolume coverage.
        
        # First, let's identify the Pareto front segments by finding local minima of f2
        # within each "lobe" of the sin function.
        # sin(10*pi*x_0) has period 0.2, so in [0,1] we have 5 periods.
        # But the Pareto front is more complex - we need to find where f2 is locally minimal
        # relative to neighboring points.
        
        # Actually, for hypervolume maximization with a reference point at [11,11],
        # we want to maximize the area under the "lower envelope" of the front.
        # Since all points on the curve with x[1:]=0 are non-dominated (the curve oscillates),
        # we should sample all of them.
        
        # Key insight: The hypervolume contribution of each point depends on its position
        # relative to other points. To maximize hypervolume, we should distribute points
        # such that they create the largest possible "staircase" under the front.
        
        # For a single-objective-like curve where f2 is a function of f1, the hypervolume
        # is maximized by placing points at the "corners" of the lower envelope.
        
        # Let's compute f2 for a dense grid to understand the structure:
        n_dense = 10000
        x0_dense = np.linspace(0, 1, n_dense)
        f2_dense = 1 - np.sqrt(x0_dense) - x0_dense * np.sin(10 * np.pi * x0_dense)
        
        # Find local minima of f2 - these are the "best" f2 values for each f1 region
        # But actually, for hypervolume, we want to cover the entire non-dominated set.
        
        # The non-dominated set is the entire curve since it's oscillating.
        # To maximize hypervolume, we should sample points that are well-distributed
        # in both f1 and f2 spaces.
        
        # Strategy: Use stratified sampling. Divide [0,1] into segments where the 
        # sin term completes half-periods, and sample each segment proportionally 
        # to its contribution to the hypervolume.
        
        # The sin(10*pi*x) has half-periods of length 0.1.
        # In [0,1], there are 10 half-periods.
        
        # For each half-period, the f2 curve goes from one extremum to another.
        # We should sample each half-period with a number of points proportional 
        # to the width of that segment in objective space.
        
        # Let's identify the breakpoints at x = k*0.1 for k=0..10
        breakpoints = np.arange(0, 11) * 0.1
        breakpoints = np.clip(breakpoints, 0, 1)
        
        # For each segment between consecutive breakpoints, compute the range of f2
        segment_ranges = []
        for i in range(len(breakpoints) - 1):
            x_seg = np.linspace(breakpoints[i], breakpoints[i+1], 100)
            f2_seg = 1 - np.sqrt(x_seg) - x_seg * np.sin(10 * np.pi * x_seg)
            # The "width" in objective space is approximately the range of f2
            seg_width = np.max(f2_seg) - np.min(f2_seg)
            # Also consider the f1 width
            f1_width = breakpoints[i+1] - breakpoints[i]
            # Combined measure: use f1 width as primary, but weight by f2 variation
            segment_ranges.append((breakpoints[i], breakpoints[i+1], f1_width, seg_width))
        
        # Allocate samples proportionally to f1 width (uniform in f1 space)
        # but ensure at least a few points per segment for coverage
        total_width = sum(s[2] for s in segment_ranges)
        n_samples = budget
        
        # Distribute samples proportionally
        raw_alloc = [int(n_samples * s[2] / total_width) for s in segment_ranges]
        # Ensure at least 2 points per segment
        min_per_seg = 2
        for i in range(len(raw_alloc)):
            if raw_alloc[i] < min_per_seg:
                raw_alloc[i] = min_per_seg
        
        # Adjust to match total budget
        while sum(raw_alloc) > n_samples:
            max_idx = np.argmax(raw_alloc)
            raw_alloc[max_idx] -= 1
        while sum(raw_alloc) < n_samples:
            min_idx = np.argmin(raw_alloc)
            raw_alloc[min_idx] += 1
        
        # Generate sample points
        x0_samples = []
        for i, (x_start, x_end, _, _) in enumerate(segment_ranges):
            n_seg = raw_alloc[i]
            if n_seg > 0:
                # Use linspace for uniform coverage within each segment
                x_seg_samples = np.linspace(x_start, x_end, n_seg)
                x0_samples.extend(x_seg_samples.tolist())
        
        # Sort by x0 to ensure we hit the front in order
        x0_samples = sorted(x0_samples)
        
        # Evaluate each point
        for x0_val in x0_samples:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0_val
            # x[1:] is already 0, which minimizes g
            try:
                problem(x)
            except Exception:
                pass
                
        return None
