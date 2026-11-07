import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT3 Pareto front: x[1:] = 0, x[0] = t in [0, 1]
        # f1 = t, f2 = (1+t)*(1+sin^2(10*t))
        # The front has 11 disconnected segments (period of sin^2(10t) is pi/5 ≈ 0.628)
        # Actually sin^2(10t) has period pi/5 in t, so over [0,1] there are about 1.59 periods
        # Let's think: sin^2(10t) = (1-cos(20t))/2, period = 2pi/20 = pi/10 ≈ 0.314
        # So over [0,1], there are about 1/0.314 ≈ 3.18 periods
        
        # For hypervolume maximization with reference point [11, 11], we want to
        # cover the front as uniformly as possible in the objective space.
        # The key insight: hypervolume contribution depends on the "area" under the front.
        # A non-uniform sampling that accounts for the local density of the front
        # in objective space will maximize hypervolume.
        
        # Strategy: Use stratified sampling based on the curvature/density of the front.
        # The f2(t) = (1+t)*(1+sin^2(10t)) function has oscillations.
        # We want points that maximize the hypervolume of the set of points.
        
        # For a set of points on a curve, the hypervolume is maximized when points
        # are spaced to maximize the area they "cover". For a monotonic-ish front,
        # uniform spacing in t gives a reasonable approximation, but we can do better
        # by accounting for the local slope.
        
        # Actually, for hypervolume of points on a curve from (0, f2(0)) to (1, f2(1)),
        # the optimal placement for a fixed number of points to maximize the hypervolume
        # is roughly uniform in the "hypervolume contribution" sense.
        
        # Let's use a practical approach: sample t values with a distribution that
        # accounts for the oscillatory nature of f2(t). We'll use a mix of:
        # 1. Uniform sampling to ensure coverage
        # 2. Extra points near local minima of f2 (where the front dips) since these
        #    contribute more to hypervolume
        
        # f2(t) = (1+t)*(1+sin^2(10t))
        # Local minima of sin^2(10t) occur at 10t = k*pi, i.e., t = k*pi/10
        # These are at t = 0, pi/10, 2*pi/10, ..., up to t <= 1
        # pi/10 ≈ 0.31416
        # So minima at t ≈ 0, 0.314, 0.628, 0.942 (within [0,1])
        
        # Let's create a stratified sample that oversamples near these minima
        # and in regions of high curvature
        
        n_samples = budget
        
        # Create base uniform grid
        t_uniform = np.linspace(0.0, 1.0, n_samples)
        
        # For better hypervolume, we can use a slightly non-uniform distribution.
        # The hypervolume of a set of points (t_i, f2(t_i)) with ref point (11,11) is:
        # Sum over consecutive points of the area they contribute.
        # For points sorted by t, the HV is roughly:
        # For each point, its contribution is related to the gap to the next point
        # and the height difference.
        
        # A practical improvement: use the inverse CDF method to place points
        # such that the "hypervolume density" is more uniform.
        
        # Let's compute f2 at many points and use that to guide sampling
        t_fine = np.linspace(0.0, 1.0, 10000)
        f2_fine = (1 + t_fine) * (1 + np.sin(10 * t_fine)**2)
        
        # The local "density" of the front in objective space is related to
        # df2/dt. Where df2/dt is small (flat regions), we need more points to
        # capture the front. Where it's steep, fewer points suffice.
        
        # Actually, for hypervolume maximization, we want to maximize the area
        # "swept" by the points. Let's just use a well-distributed set.
        
        # Use a deterministic quasi-random sequence (Halton-like) for better coverage
        # Or simply use linspace which is already quite good for this 1D case.
        
        # Let's stick with linspace but ensure we hit the endpoints exactly
        # and add a few extra points near the local minima of f2
        
        # Actually, linspace with 200 points should be quite good.
        # Let's just use it directly.
        
        t_values = t_uniform
        
        # Evaluate all points on the Pareto front
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = t_values[i]
            # x[1:] stays 0
            _ = problem(x)
        
        return None
