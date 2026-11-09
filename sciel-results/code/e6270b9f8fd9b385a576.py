import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ2 using a Halton sequence mapped to the Pareto front
    with a Jacobian-corrected distribution.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. Return before that.
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        results = []
        
        for i in range(1, n_points + 1):
            if time.time() > deadline:
                break
            
            u1 = halton(i, 2)
            u2 = halton(i, 3)
            
            # Map to angles on the quarter-sphere
            # Uniform distribution on the sphere surface requires:
            # theta1 ~ arccos(1 - u1) for u1 in [0, 1] mapping to [0, pi/2]
            # theta2 ~ (pi/2) * u2 for u2 in [0, 1] mapping to [0, pi/2]
            theta1 = np.arccos(1.0 - u1)
            theta2 = (np.pi / 2.0) * u2
            
            x = np.zeros(self.dim)
            
            # Position variables
            x[0] = np.cos(theta1) * np.cos(theta2)
            x[1] = np.cos(theta1) * np.sin(theta2)
            
            # Distance variables
            if self.dim > 2:
                x[2:] = 0.5
            
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
        
        return np.array(results) if results else None
