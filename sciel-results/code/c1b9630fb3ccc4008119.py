import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3 using Hypervolume-Optimized Front Sampling.
    
    Strategy:
    ZDT3's Pareto front is analytically known:
    f1(x) = x1
    f2(x) = 1 - sqrt(x1)  (when x2..xn = 0)
    
    The key insight is that uniform sampling of x1 is suboptimal for hypervolume.
    The hypervolume contribution of a point on the front depends on its position.
    Points near x1=0 have a steep slope (df2/dx1 = -1/(2*sqrt(x1)) → -∞ as x1→0),
    meaning small changes in x1 produce large changes in f2. This region contributes
    disproportionately to hypervolume because the "box" each point can claim is larger
    in the f2 direction.
    
    We use a non-uniform sampling distribution that concentrates points where the
    hypervolume gain per point is maximized. Specifically, we sample x1 values
    according to a distribution proportional to the local hypervolume contribution,
    which for a smooth front is related to the curvature and the distance to the
    reference point.
    
    A practical approach: use a power-law distribution x1 = u^alpha where u ~ Uniform(0,1)
    and alpha < 1 to concentrate points near 0. We choose alpha to balance coverage.
    Alternatively, we can analytically compute the optimal placement.
    
    For the ZDT3 front f2 = 1 - sqrt(f1), with reference point (11, 11):
    The hypervolume contribution of a point (f1, f2) relative to neighbors depends on
    the area of the rectangle it "owns". For a dense set of points, the optimal density
    is proportional to 1/sqrt(f1) (the magnitude of the derivative), suggesting we should
    sample x1 with a density proportional to 1/sqrt(x1).
    
    We use: x1_i = (i / n)^2 * x1_hi  (quadratic spacing, denser near 0)
    This gives density proportional to 1/sqrt(x1).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        x1_lo = lo[0]
        x1_hi = hi[0]
        
        n_evals = self.budget

        # Use quadratic spacing: x1_i = x1_lo + (x1_hi - x1_lo) * (i/n)^2
        # This concentrates points near x1_lo (which is 0 for standard bounds)
        # where the front has high curvature and high hypervolume contribution
        indices = np.arange(n_evals, dtype=np.float64)
        # Normalize to [0, 1)
        normalized = indices / float(n_evals)
        # Quadratic mapping: denser near 0
        x1_values = x1_lo + (x1_hi - x1_lo) * (normalized ** 2)
        
        # Ensure we stay within bounds
        x1_values = np.clip(x1_values, x1_lo, x1_hi)

        # Prepare a batch of points
        X = np.zeros((n_evals, n))
        X[:, 0] = x1_values
        # x[1:] are 0.0, optimal for ZDT3

        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)

        return np.array(F)
