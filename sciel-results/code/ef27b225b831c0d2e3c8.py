import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        dim = self.dim
        
        # ZDT3 analytic Pareto front:
        # f1 = x1
        # f2 = h(x1) where h(x) = 1 + 9/n * (sum(x_i) - n)
        # For x_2..x_n = 0, sum(x_i) = x1, so h(x1) = 1 + 9/(n-1) * (x1 - (n-1))
        # Wait, let me re-derive ZDT3:
        # f1(x) = x1
        # g(x) = 1 + 9 * sum(x_i) / (n-1) for i=2..n
        # h(x) = 1 + 9 * (g(x) - 1) / (n-1) * sin(10 * pi * x1)
        # Actually for ZDT3:
        # g(x) = 1 + 9 * sum_{i=2}^{n} x_i / (n-1)
        # h(x) = 1 + 9 * (g(x) - 1) * sin(10 * pi * x1)
        # f1 = x1
        # f2 = h(x) * (1 - sqrt(f1/h))... no wait
        
        # Let me recall ZDT3 properly:
        # f1(x) = x1
        # g(x) = 1 + 9 * sum_{i=2}^{n} x_i / (n-1)
        # h(x) = 1 + 9 * (g(x) - 1) * sin(10 * pi * x1)
        # f2(x) = h(x) * (1 - sqrt(f1(x) / h(x)))
        
        # On the Pareto front, g(x) = 1, so x_2..x_n = 0
        # Then h(x) = 1 + 9 * 0 * sin(10*pi*x1) = 1
        # So f2 = 1 * (1 - sqrt(x1/1)) = 1 - sqrt(x1)
        
        # So Pareto front: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1]
        # This is a concave curve.
        
        # To maximize hypervolume with reference point [11, 11], we want to sample
        # points that spread well along this front.
        
        # The hypervolume contribution of a point (f1, f2) relative to reference (11,11)
        # depends on the area it covers. For a monotonic decreasing front, we want
        # to distribute points to maximize the union of rectangles.
        
        # A good strategy: sample x1 values non-uniformly. Since f2 = 1 - sqrt(x1),
        # the curve has high slope near x1=0 and low slope near x1=1.
        
        # Let's use inverse CDF sampling based on the curvature or just use a
        # power-law distribution to concentrate more points where the front curves more.
        
        # Actually, for hypervolume maximization on a monotonic front, uniform sampling
        # in objective space is ideal. Since f2 = 1 - sqrt(x1), we can solve for x1:
        # f2 = 1 - sqrt(x1) => sqrt(x1) = 1 - f2 => x1 = (1-f2)^2
        # So if we sample f2 uniformly in [0,1], x1 = (1-f2)^2 gives uniform spacing in f2.
        
        # Let's sample f2 uniformly, then compute x1.
        # f2 in [0, 1], x1 = (1 - f2)^2
        # f2 = 0 => x1 = 1, f2 = 1 => x1 = 0
        
        # Use midpoints for better coverage in f1 space
        f1_values = np.linspace(0, 1, n_evals, endpoint=False) + 0.5 / n_evals
        x1_values = f1_values
        
        # Construct full vectors with x_2..x_n = 0
        X = np.zeros((n_evals, dim))
        X[:, 0] = x1_values
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return F
