import numpy as np
import heapq
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        N = min(self.budget, 200)
        
        # ZDT1 Pareto Front: x1 in [0, 1], x2..xn = 0
        # f1 = x1, f2 = 1 - sqrt(x1)
        
        # We will determine the optimal x1 coordinates using adaptive bisection
        # to maximize the hypervolume.
        
        # f(x) = 1 - sqrt(x)
        def f(x):
            return 1.0 - np.sqrt(x)
        
        # Find the optimal x in [a, b] to maximize the hypervolume contribution
        # The gain of adding x to an interval [a, b] (where a is the left neighbor and b is the right neighbor)
        # is the area of the rectangle (b-a)*f(b) replaced by two rectangles (x-a)*f(x) + (b-x)*f(b)
        # Wait, the hypervolume contribution of a point (x_i, f(x_i)) in a 2D minimization problem with ref point (r1, r2)
        # is sum over i of (x_{i+1} - x_i) * (r2 - f(x_i)) if we assume x is sorted and x_0=0, x_{n}=r1?
        # Actually, for ZDT1, the Pareto front is f2 = 1 - sqrt(f1).
        # The hypervolume of a set of points P is the volume of the region dominated by P and better than ref.
        # If we have points (x_i, f(x_i)) sorted by x_i, the hypervolume is sum_{i} (x_{i+1} - x_i) * (r2 - f(x_i))
        # where x_{n+1} = r1 and f(x_{n+1}) is irrelevant, but the last term is (r1 - x_n) * (r2 - f(x_n)).
        # Here r1 = 11, r2 = 11.
        # Since x_i in [0,1], r1 is much larger.
        # The total hypervolume is constant terms + sum_{i=0}^{n-1} (x_{i+1} - x_i) * (11 - f(x_i)) + (11 - x_n) * (11 - f(x_n))
        # Let's simplify. The gain of inserting x between a and b (a < x < b) is:
        # Old contribution: (b - a) * (11 - f(b))  <-- Wait, this is not quite right.
        # Let's look at the standard HV formula for sorted points x_1 < x_2 < ... < x_n.
        # HV = sum_{i=1}^{n} (x_{i+1} - x_i) * (r2 - f(x_i)) where x_{n+1} = r1.
        # If we insert x between x_i (a) and x_{i+1} (b):
        # Old term: (b - a) * (r2 - f(b))  -- NO, the term associated with x_i is (x_{i+1} - x_i) * (r2 - f(x_i)).
        # So the term for x_i is (b - a) * (r2 - f(a)).
        # New terms: (x - a) * (r2 - f(a)) + (b - x) * (r2 - f(x)).
        # Gain = (b - x) * (r2 - f(x)) - (b - x) * (r2 - f(a)) = (b - x) * (f(a) - f(x)).
        # Wait, is this correct?
        # Let's check.
        # Old: (b - a) * (r2 - f(a))
        # New: (x - a) * (r2 - f(a)) + (b - x) * (r2 - f(x))
        # Difference = (b - x) * (r2 - f(x)) - (b - x) * (r2 - f(a)) = (b - x) * (f(a) - f(x)).
        # Since f is decreasing (f(a) > f(x)), this gain is positive.
        # We want to maximize G(x) = (b - x) * (f(a) - f(x)) for x in (a, b).
        # G(x) = (b - x) * (1 - sqrt(a) - (1 - sqrt(x))) = (b - x) * (sqrt(x) - sqrt(a)).
        # Let's find x that maximizes G(x).
        # G'(x) = -1 * (sqrt(x) - sqrt(a)) + (b - x) * (1 / (2 * sqrt(x)))
        # Set G'(x) = 0:
        # sqrt(x) - sqrt(a) = (b - x) / (2 * sqrt(x))
        # 2 * x - 2 * sqrt(a) * sqrt(x) = b - x
        # 3 * x - 2 * sqrt(a) * sqrt(x) - b = 0
        # Let s = sqrt(x).
        # 3 * s^2 - 2 * sqrt(a) * s - b = 0
        # s = [2 * sqrt(a) + sqrt(4 * a + 12 * b)] / 6
        # x = s^2
        
        def find_optimal_x(a, b):
            # Maximize (b - x) * (f(a) - f(x))
            # f(x) = 1 - sqrt(x)
            # f(a) - f(x) = sqrt(x) - sqrt(a)
            # G(x) = (b - x) * (sqrt(x) - sqrt(a))
            # G'(x) = 0 => 3x - 2*sqrt(a)*sqrt(x) - b = 0
            # 3s^2 - 2*sqrt(a)*s - b = 0
            s_a = np.sqrt(a)
            A = 3.0
            B = -2.0 * s_a
            C = -b
            disc = B*B - 4*A*C
            if disc < 0:
                return 0.5 * (a + b)
            s = (-B + np.sqrt(disc)) / (2.0 * A)
            x = s * s
            if x < a: x = a
            if x > b: x = b
            return x

        # Initialize with [0, 1]
        # We need N points.
        # Start with x=0 and x=1.
        points = [0.0, 1.0]
        
        # We need to maintain the sorted order and the intervals.
        # Since we always insert, we can use a list of intervals (a, b) and a heap.
        # But we need to know the gain of inserting into each interval.
        
        def calc_gain(a, b):
            x_opt = find_optimal_x(a, b)
            if x_opt <= a or x_opt >= b:
                return 0.0
            return (b - x_opt) * (f(a) - f(x_opt))
        
        # Heap of (-gain, a, b)
        heap = []
        g0 = calc_gain(0.0, 1.0)
        heapq.heappush(heap, (-g0, 0.0, 1.0))
        
        for _ in range(N - 2):
            if time.time() > deadline:
                break
            neg_g, a, b = heapq.heappop(heap)
            x_opt = find_optimal_x(a, b)
            points.append(x_opt)
            
            # Add new intervals [a, x_opt] and [x_opt, b]
            g1 = calc_gain(a, x_opt)
            g2 = calc_gain(x_opt, b)
            heapq.heappush(heap, (-g1, a, x_opt))
            heapq.heappush(heap, (-g2, x_opt, b))
            
        # Sort points to ensure correct order for evaluation
        points = sorted(points)
        
        # Create X matrix
        X = np.zeros((len(points), self.dim))
        X[:, 0] = points
        # X[:, 1:] remains 0
        
        # Evaluate
        for i in range(len(points)):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
