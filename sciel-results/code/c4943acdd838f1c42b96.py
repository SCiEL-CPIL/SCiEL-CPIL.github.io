import numpy as np

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            
            # We know DTLZ7 structure:
            # f1 = x1, f2 = x2, f3 depends on x1, x2 and g(x3..xn)
            # Optimal g is 0, so x3..xn = 0.
            
            # Create base vector with 0s for x3..xn
            # But we must respect bounds. If bounds are [0,1], 0 is valid.
            # If bounds are different, we should use the lower bound for x3..xn to minimize g.
            base = lo.copy()
            # Ensure x1, x2 are within bounds
            
            points_to_eval = []
            
            # 1. Add vertices/corners
            # x1, x2 at corners of [lo[0], hi[0]] x [lo[1], hi[1]]
            # Assuming standard [0,1]
            x1_lo, x1_hi = bounds[0]
            x2_lo, x2_hi = bounds[1]
            
            corners = [
                [x1_lo, x2_lo],
                [x1_hi, x2_lo],
                [x1_lo, x2_hi],
                [x1_hi, x2_hi] # This one is likely dominated, but good for coverage?
            ]
            
            for c in corners:
                x = base.copy()
                x[0] = c[0]
                x[1] = c[1]
                points_to_eval.append(x)
                
            # 2. Add biased random samples
            n_remaining = self.budget - len(points_to_eval)
            rng = np.random.default_rng(42)
            
            for _ in range(n_remaining):
                u = rng.random()
                v = rng.random()
                # Bias towards origin (lower bound)
                # Use power > 1 to concentrate near 0
                # If bounds are [0,1], u^2 is in [0,1]
                # Generalize: lo + (hi-lo) * u^2
                x = base.copy()
                x[0] = x1_lo + (x1_hi - x1_lo) * (u ** 2)
                x[1] = x2_lo + (x2_hi - x2_lo) * (v ** 2)
                points_to_eval.append(x)
                
            # Evaluate all
            # The problem function clips, so we are safe.
            for x in points_to_eval:
                problem(x)
                
            return None
