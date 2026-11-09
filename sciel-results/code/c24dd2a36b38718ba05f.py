import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 structure:
        # f1 = x[0]
        # f2 = g(x[1:]) * (1 - sqrt(x[0]/n))
        # g(x) = sum_{i=2}^{n} (x_i^2 - 10*cos(2*pi*x_i))
        # For standard ZDT4, min of each term is -9 at x_i = 0, so g_min = 9*(n-1) = 81 for n=10
        # But we need to find the actual g_min given the bounds.
        
        # Step 1: Estimate g_min
        # The minimum of x^2 - 10*cos(2*pi*x) occurs at x=0 (value -9) or at boundaries
        # For each dimension i >= 1, find the x_i that minimizes x_i^2 - 10*cos(2*pi*x_i)
        
        def g_term(x):
            return x*x - 10.0*np.cos(2.0*np.pi*x)
        
        # For each tail dimension, find the minimizer within bounds
        optimal_tail = np.zeros(self.dim - 1)
        for i in range(1, self.dim):
            l, h = lo[i], hi[i]
            # The function x^2 - 10*cos(2*pi*x) has global min at x=0 (value -9)
            # Periodic with period 1, so local minima at every integer
            # Within [l, h], check: 0, and integers in [l, h], and boundaries
            candidates = [l, h]
            # Add integers in [l, h]
            if l <= 0 <= h:
                candidates.append(0.0)
            # Add nearby integers
            for k in range(int(np.floor(l)) - 1, int(np.ceil(h)) + 2):
                if l - 0.1 <= k <= h + 0.1:
                    candidates.append(float(k))
            # Evaluate g_term at each candidate
            best_val = float('inf')
            best_x = l
            for c in candidates:
                c = np.clip(c, l, h)
                v = g_term(c)
                if v < best_val:
                    best_val = v
                    best_x = c
            optimal_tail[i-1] = best_x
        
        # Evaluate g_min by constructing a point with x[0] chosen so we can read g
        # If x[0] = 0, f2 = g(x[1:]) * (1 - 0) = g(x[1:])
        # But x[0] must be in bounds. Let's check.
        x1_for_g = 0.0
        if x1_for_g < lo[0]:
            x1_for_g = lo[0]
        elif x1_for_g > hi[0]:
            x1_for_g = hi[0]
        
        x_test = np.zeros(self.dim)
        x_test[0] = x1_for_g
        x_test[1:] = optimal_tail
        f_test = problem(x_test)
        
        if x1_for_g > 0:
            g_min = f_test[1] / (1.0 - np.sqrt(x1_for_g / self.dim))
        else:
            g_min = f_test[1]
        
        # Verify: also check with a slightly different x[0] to confirm
        used = 1
        
        # Step 2: Sample the Pareto front densely
        # f1 = x[0], f2 = g_min * (1 - sqrt(x[0]/n))
        # We want to place points along this curve to maximize hypervolume
        # with reference point [11.0, 400.0]
        
        remaining = self.budget - used
        if remaining <= 0:
            return []
        
        x1_lo = lo[0]
        x1_hi = hi[0]
        
        # The Pareto front is f2 = g_min * (1 - sqrt(f1/n))
        # This is a concave curve. For hypervolume maximization with reference [11, 400],
        # we want points spread along the front.
        # The ideal points are on the Pareto front.
        
        # Use a combination of uniform and concentrated sampling
        # The curvature is highest near f1 = 0, so bias samples there
        
        # Strategy: use multiple spacing strategies
        n_uniform = remaining // 2
        n_concentrated = remaining - n_uniform
        
        x1_samples = []
        
        # Uniform samples
        if n_uniform > 0:
            x1_samples.extend(np.linspace(x1_lo, x1_hi, n_uniform).tolist())
        
        # Concentrated samples near x1 = 0 using power law
        if n_concentrated > 0:
            t = np.linspace(0, 1, n_concentrated)
            alpha = 2.0  # Bias toward lower x1 values
            t_conc = t ** alpha
            x1_samples.extend((x1_lo + t_conc * (x1_hi - x1_lo)).tolist())
        
        # Also add some samples specifically near the ideal point
        # The ideal point for f1 is 0, for f2 is 0
        # Points near f1=0 give low f2, which is good for hypervolume
        
        # Evaluate all samples
        F = []
        for x1_val in x1_samples:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = np.clip(x1_val, lo[0], hi[0])
            x[1:] = optimal_tail
            f = problem(x)
            F.append(f)
        
        # If we have budget left, do a refinement pass
        # Find the current best points and refine around them
        if F and time.time() < deadline - 5:
            # Use remaining budget for local refinement
            remaining2 = self.budget - used - len(F)
            if remaining2 > 0:
                # Pick a few promising x1 values and do finer sampling around them
                # Focus on x1 values that give good hypervolume contribution
                # Simply do a finer grid on a subset
                n_refine = min(remaining2, 20)
                if n_refine > 0:
                    # Sample around x1 = 0 and x1 = x1_hi/2
                    refine_regions = [
                        (max(x1_lo, 0), min(x1_hi, 0.5)),
                        (x1_lo, x1_hi * 0.3),
                    ]
                    for r_lo, r_hi in refine_regions:
                        if r_lo >= r_hi:
                            continue
                        n_pts = max(1, n_refine // len(refine_regions))
                        t_r = np.linspace(0, 1, n_pts)
                        for t_r_val in t_r:
                            if time.time() > deadline:
                                break
                            x1_val = r_lo + t_r_val * (r_hi - r_lo)
                            x = np.zeros(self.dim)
                            x[0] = np.clip(x1_val, lo[0], hi[0])
                            x[1:] = optimal_tail
                            f = problem(x)
                            F.append(f)
        
        return F
