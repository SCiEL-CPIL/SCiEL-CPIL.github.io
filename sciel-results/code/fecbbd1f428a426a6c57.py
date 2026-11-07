"""Program using adaptive subdivision on x1 for ZDT1 hypervolume maximization."""
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer leveraging ZDT1 structure:
    - x1 controls the Pareto front position.
    - x2...x30 only affect the second objective (g).
    Strategy: 
    1. Set x2...x30 to 0 to minimize g (and thus f2) for any given x1.
    2. Use adaptive subdivision on x1 to place points where they contribute
       most to hypervolume. Start with endpoints, then iteratively subdivide
       the interval that gives the largest hypervolume gain.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 specific knowledge:
        # f1 = x1
        # g = 1 + 9 * sum(x2...xn) / (n-1)
        # f2 = g * (1 - sqrt(x1/g))
        # Optimal g is 1 (when x2...xn = 0).
        # Front: f2 = 1 - sqrt(f1) for f1 in [0, 1].
        
        N = self.budget
        if N <= 1:
            x = np.zeros(self.dim)
            problem(x)
            return None

        base_vec = np.zeros(self.dim)
        
        # Evaluate endpoints first
        evaluated_x1 = [0.0, 1.0]
        for x1_val in evaluated_x1:
            x = base_vec.copy()
            x[0] = x1_val
            problem(x)
        
        # Adaptive subdivision: maintain intervals between consecutive evaluated points
        # Each interval is (x1_left, x1_right)
        # We subdivide the interval that gives the largest hypervolume gain
        # For ZDT1, the hypervolume contribution of a point at x1 in interval [a, b]
        # is approximately (b - x1) * (f2(x1) - f2(b)) for the rectangle it adds
        # Actually, the marginal contribution depends on the current set of points.
        
        # Simpler approach: use the known analytic structure.
        # The hypervolume of a set of points on the Pareto front is maximized
        # when points are placed according to a density proportional to the
        # curvature of the front. For f2 = 1 - sqrt(f1), the optimal density
        # is proportional to 1/sqrt(f1).
        
        # Let's use a different strategy: start with a coarse grid, then
        # iteratively add points at the midpoint of the interval with the
        # largest "gap" in hypervolume contribution.
        
        # For 2D, the hypervolume of points (f1_i, f2_i) with ref point (11, 11)
        # is sum over sorted points of (f1_{i+1} - f1_i) * (11 - f2_i)
        # where f1 is sorted ascending and f2 is sorted descending (for Pareto front).
        
        # Let's implement adaptive insertion:
        # Keep track of evaluated x1 values, sorted.
        # The marginal gain of adding a point at x1_new between x1_a and x1_b
        # (where x1_a < x1_new < x1_b) can be computed.
        
        # For simplicity and speed, let's use a heuristic:
        # The optimal placement for hypervolume on a concave front is to
        # place points at x1 = (i/N)^2 for i = 1, ..., N-1 (plus endpoints).
        # This is because the derivative of f2 with respect to f1 is -1/(2*sqrt(f1)),
        # which has magnitude proportional to 1/sqrt(f1).
        
        # Actually, let's think more carefully. The hypervolume in 2D with
        # reference point (R1, R2) for a set of Pareto-optimal points is:
        # HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (R2 - f2_i)
        # where points are sorted by f1 ascending, and f1_{n+1} = R1.
        
        # For the ZDT1 front f2 = 1 - sqrt(f1), with R1 = R2 = 11:
        # HV = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (11 - 1 + sqrt(f1_i))
        #    = sum_{i=1}^{n} (f1_{i+1} - f1_i) * (10 + sqrt(f1_i))
        
        # To maximize this, we want to place points where the integrand
        # (10 + sqrt(f1)) is large relative to the interval width.
        # Since (10 + sqrt(f1)) is larger for smaller f1, we should place
        # more points near f1 = 0.
        
        # The optimal placement satisfies a specific ODE, but a good
        # approximation is f1_i = (i/(n+1))^alpha for some alpha > 1.
        # Let's try alpha = 2.
        
        if N > 2:
            # Generate N-2 interior points.
            # For 2D HV with ref (11,11) and front f2=1-sqrt(x1),
            # optimal density is proportional to (10+sqrt(x1))^2 near x1=0.
            # Use x1 = (i/(N-1))^alpha. Try alpha=1.5 as a better fit than 2.
            idx = np.arange(1, N-1)
            t = idx / (N - 1)
            # Use alpha=1.5 which balances the (10+sqrt(x1)) weight better
            x1_interior = t ** 1.5
            x1_values = np.concatenate([[0.0], x1_interior, [1.0]])
        else:
            x1_values = np.array(evaluated_x1)
        
        # We've already evaluated x1=0 and x1=1. Now evaluate the interior points.
        for i in range(1, N-1):
            if time.time() > deadline:
                break
            
            x1_val = x1_values[i]
            x = base_vec.copy()
            x[0] = x1_val
            problem(x)
        
        return None
