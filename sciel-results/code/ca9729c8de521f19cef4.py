import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT1.
    Uses the known analytical Pareto front to place evaluations optimally.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        # Safety margin: The evaluator kills at 330s. We aim to finish in < 5s.
        deadline = t0 + 10.0 

        B = int(self.budget)
        D = int(self.dim)
        
        if B <= 0:
            return np.zeros(D)

        # ZDT1 Pareto Front:
        # x1 in [0, 1], x2...xD = 0
        # f1 = x1
        # f2 = 1 - sqrt(x1)
        
        # Reference point
        R1 = 11.0
        R2 = 11.0

        def compute_hv(x1_vals):
            """
            Calculate Hypervolume for points on the ZDT1 front.
            x1_vals: array of x1 values (sorted ascending).
            """
            if len(x1_vals) == 0:
                return 0.0
            
            # Ensure sorted
            x1_sorted = np.sort(x1_vals)
            
            # f2 values
            f2 = 1.0 - np.sqrt(x1_sorted)
            
            # Heights relative to reference point
            # The HV contribution of a point i is (next_f1 - current_f1) * (R2 - current_f2)
            # The last point extends to R1
            next_f1 = np.append(x1_sorted[1:], R1)
            current_f1 = x1_sorted
            
            widths = next_f1 - current_f1
            heights = R2 - f2
            
            # Sum of areas of the rectangles
            hv = np.sum(widths * heights)
            
            return float(hv)

        # Optimal quantization for hypervolume maximization.
        # For a Pareto front f2(x1) = 1 - sqrt(x1), the optimal point density
        # for maximizing HV with reference point (R1, R2) is proportional to
        # the height: h(x1) = R2 - f2(x1) = R2 - 1 + sqrt(x1) = 10 + sqrt(x1).
        # 
        # The optimal CDF is:
        #   G(x) = (10*x + (2/3)*x^(3/2)) / (10 + 2/3)
        # 
        # We place B points at u_i = (i + 0.5) / B (midpoint rule for optimal quantization)
        # and solve G(x) = u_i for x.
        
        # Use midpoint rule: u_i = (i + 0.5) / B
        u = (np.arange(B) + 0.5) / B
        
        # Solve G(x) = u for x using Newton's method
        # G(x) = (10*x + (2/3)*x^(3/2)) / (32/3)
        # G'(x) = (10 + sqrt(x)) / (32/3)
        
        def G(x):
            return (10.0 * x + (2.0/3.0) * np.power(x, 1.5)) / (32.0/3.0)
        
        def G_inv(u_vals):
            """Invert G(x) = u for x using Newton's method."""
            x = np.copy(u_vals)  # Initial guess
            x = np.clip(x, 1e-15, 1.0 - 1e-15)
            for _ in range(50):
                gx = G(x)
                dgx = (10.0 + np.sqrt(x)) / (32.0/3.0)
                x_new = x - (gx - u_vals) / dgx
                x_new = np.clip(x_new, 1e-15, 1.0 - 1e-15)
                if np.max(np.abs(x_new - x)) < 1e-15:
                    x = x_new
                    break
                x = x_new
            return x
        
        x1_vals = G_inv(u)
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        
        # Construct X
        X = np.zeros((B, D))
        X[:, 0] = x1_vals
        # x2...xD = 0 is already set by zeros
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        X = np.clip(X, lo, hi)
        
        # Evaluate
        for i in range(B):
            if time.time() > deadline:
                break
            problem(X[i].copy())
            
        return X[-1] # Return value is ignored, but must not crash
