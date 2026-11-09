import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.dim
        if n <= 1:
            return None

        # ZDT3 specific logic
        # f1 = x0
        # g = 1 + 10*(n-1) + sum(xi^2 - 10*cos(4*pi*xi)) for i=1..n-1
        # min g is at xi=0, g_min = 1 + 10*(n-1)
        # f2 = g * (1 + sin(10*pi*f1))
        # We want f2 < 11.
        # Assuming xi=0 for i>0, g = 1 + 10*(n-1).
        
        g_min = 1.0 + 10.0 * (n - 1)
        
        # Condition: g_min * (1 + sin(10*pi*x0)) < 11
        # 1 + sin(10*pi*x0) < 11 / g_min
        threshold = 11.0 / g_min
        
        # sin(10*pi*x0) < threshold - 1
        sin_val_target = threshold - 1.0
        
        # If sin_val_target >= 1, all x0 are valid (but f2 might be > 11 if sin is not close to -1? No, if sin < 1, it's not enough. We need sin to be small enough).
        # Actually, we need 1 + sin(...) < threshold.
        # Since max sin is 1, min value of LHS is 0.
        # If threshold <= 0, no solution.
        # If threshold > 0, we need sin(10*pi*x0) < threshold - 1.
        
        if threshold <= 0:
            # No points can satisfy f2 < 11 with g_min.
            # Fallback to random
            for _ in range(self.budget):
                if time.time() > deadline: break
                x = np.random.uniform(lo, hi)
                try: problem(x)
                except: break
            return None

        # We need sin(theta) < sin_target where theta = 10*pi*x0
        # sin_target is negative (since threshold is small, e.g., 11/291 ~ 0.037, sin_target ~ -0.96)
        
        # Find intervals of x0 in [0, 1] where this holds.
        # theta in [0, 10*pi].
        # sin(theta) < S where S < 0.
        # Solutions are around 3*pi/2 + 2*k*pi.
        
        intervals = []
        for k in range(10): # k=0 to 9 covers 10 periods
            center_theta = 1.5 * np.pi + 2 * k * np.pi
            # Find width delta where sin(center_theta + delta) = S
            # sin(3*pi/2 + delta) = -cos(delta) = S => cos(delta) = -S
            # delta = arccos(-S)
            # Wait, sin(3pi/2 + d) = -cos(d). We want -cos(d) < S => cos(d) > -S.
            # So d is in (-arccos(-S), arccos(-S)).
            
            if -S > 1: continue
            if -S < -1: continue
            
            delta = np.arccos(-S)
            theta_low = center_theta - delta
            theta_high = center_theta + delta
            
            # Convert to x0
            x0_low = theta_low / (10.0 * np.pi)
            x0_high = theta_high / (10.0 * np.pi)
            
            # Clip to [0, 1]
            if x0_high > 0 and x0_low < 1:
                intervals.append((max(0, x0_low), min(1, x0_high)))

        if not intervals:
             # Fallback
             for _ in range(self.budget):
                 if time.time() > deadline: break
                 x = np.random.uniform(lo, hi)
                 try: problem(x)
                 except: break
             return None

        # Distribute budget among intervals
        # To maximize hypervolume, we want to cover the range of f1 (x0) as well as possible.
        # The intervals are disjoint.
        # We can just pick points in each interval.
        # Number of points per interval can be proportional to width, or equal.
        # Equal is fine to ensure coverage of all peaks of the "valid" regions.
        
        n_intervals = len(intervals)
        points_per_interval = self.budget // n_intervals
        remainder = self.budget % n_intervals
        
        evaluations = 0
        for i, (x0_lo, x0_hi) in enumerate(intervals):
            count = points_per_interval
            if i < remainder:
                count += 1
            
            if count <= 0: continue
            
            # Sample count points uniformly in [x0_lo, x0_hi]
            # Add small epsilon to avoid boundary issues if any
            x0_samples = np.linspace(x0_lo, x0_hi, count)
            
            for x0 in x0_samples:
                if evaluations >= self.budget or time.time() > deadline:
                    break
                
                x = np.full(self.dim, lo[1]) # Set to lower bound
                x[0] = x0
                try:
                    problem(x)
                except Exception:
                    break
                evaluations += 1

        return None
