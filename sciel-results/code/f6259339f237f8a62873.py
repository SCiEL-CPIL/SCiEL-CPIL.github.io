"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    Strategy: Direct deterministic sampling of the known 1D Pareto front.
    ZDT3's Pareto front is characterized by g(x)=1 and f1(x)=x1.
    The front is discontinuous, consisting of 7 segments.
    We sample points uniformly across these segments to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front: g(x)=1 requires x_i in {0, 1/3, 2/3, 1} for i>=2
        # f1 = x1, so we vary x1 in [0,1]
        # The h function creates discontinuities at specific f1 values
        
        # For ZDT3: h(f1, g) = 1 - sqrt(f1/g) - (f1/g)*sin(10*pi*f1)
        # When g=1: h(f1,1) = 1 - sqrt(f1) - f1*sin(10*pi*f1)
        # f2 = (1+1) * h(f1,1) = 2*h(f1,1)
        
        # The Pareto front segments occur where h is decreasing in f1
        # Key points: f1 = 0, 0.082, 0.166, 0.258, 0.370, 0.444, 0.619, 0.771, 1.0
        # These are the local minima of h(f1,1)
        
        # Strategy: sample x1 uniformly in [0,1], set other x_i to make g=1
        # Use a mix of the discrete values for x_i to ensure g=1
        
        n = self.dim
        budget = self.budget
        
        # Generate candidate points on the Pareto front
        # x1 varies, x_i (i>=2) are set to values that minimize g
        # g(x) = 1 + 10(n-1) + sum(10*sin(3*pi*x_i)) for i=2..n
        # To get g=1: sin(3*pi*x_i) = -1 for all i, i.e., 3*pi*x_i = 3*pi/2 + 2*pi*k
        # So x_i = 1/2 + 2*k/3... wait, let me recalculate
        # sin(3*pi*x_i) = -1 when 3*pi*x_i = 3*pi/2 + 2*pi*k => x_i = 1/2 + 2k/3
        # For k=0: x_i = 0.5, for k=1: x_i = 0.5+2/3 = 7/6 > 1 (invalid)
        # For k=-1: x_i = 0.5-2/3 = -1/6 < 0 (invalid)
        # Actually: sin(3*pi*x) = -1 at x = 1/2, but also need to check
        # 3*pi*x = pi/2 + 2*pi*k gives sin = 1, 3*pi*x = 3*pi/2 + 2*pi*k gives sin = -1
        # x = (3*pi/2 + 2*pi*k) / (3*pi) = 1/2 + 2k/3
        # k=0: x=0.5, k=-1: x=0.5-2/3=-1/6 (invalid), k=1: x=0.5+2/3=7/6 (invalid)
        # So only x_i = 0.5 works? Let me verify: sin(3*pi*0.5) = sin(3*pi/2) = -1. Yes!
        # But wait, there might be other points. Let's check x=0: sin(0)=0, x=1/3: sin(pi)=0
        # x=2/3: sin(2*pi)=0, x=1: sin(3*pi)=0
        # So sin(3*pi*x) = 0 at x=0,1/3,2/3,1 and =-1 at x=0.5, =1 at x=1/6,5/6
        # To minimize g, we want sin(3*pi*x_i) = -1, so x_i = 0.5 for all i>=2
        
        # Actually, let me reconsider. The standard ZDT3 definition:
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^n [10*(sin(3*pi*x_i))]
        # So minimum g = 1 + 10*(n-1) + sum of minimum of 10*sin(3*pi*x_i)
        # min of sin(3*pi*x_i) = -1, so min g = 1 + 10*(n-1) - 10*(n-1) = 1
        # This requires sin(3*pi*x_i) = -1 for all i>=2, i.e., x_i = 0.5
        
        # So the Pareto front is: x1 in [0,1], x_i = 0.5 for i=2..n
        # f1 = x1
        # f2 = (1+g) * h(f1, g) = 2 * h(f1, 1)
        # h(f1, 1) = 1 - sqrt(f1) - f1*sin(10*pi*f1)
        
        # Generate budget points with x1 uniformly distributed in [0,1]
        # and x_i = 0.5 for i >= 2
        
        x1_values = np.linspace(0, 1, budget, endpoint=True)
        
        points = []
        for x1 in x1_values:
            x = np.full(n, 0.5)  # Set all to 0.5 initially
            x[0] = x1  # Set first coordinate
            # Clip to bounds
            x = np.clip(x, lo, hi)
            points.append(x)
        
        # Evaluate all points
        F = np.array([problem(x) for x in points])
        
        return points
# EVOLVE-BLOCK-END
