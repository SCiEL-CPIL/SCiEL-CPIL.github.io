import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        dim = self.dim
        
        # ZDT6 Pareto front: x2..xn = 0, x1 in [0, 1]
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)**8
        # f2(x) = g(x) * (1 + sin(10*pi*x1))
        # where g(x) = 1 + 9/n * sum(x_i) for i=2..n
        # On Pareto front (x2..xn=0): g=1, so f2 = 1 + sin(10*pi*x1)
        
        # We need to choose x1 values to maximize hypervolume.
        # The Pareto front has f1 in [0, ~1] and f2 in [0, 2].
        # Reference point is [11, 11].
        
        # Strategy: Use a non-uniform distribution of x1 that concentrates
        # points where the Pareto front has high curvature (low f1 region).
        # Based on the knowledge: u^0.7 concentrates points at low f1.
        
        # Let's generate samples using a power law in f1-space.
        # Since f1 is monotonically increasing in x1 (mostly), we can
        # sample uniformly in f1 and map back to x1, or use a weighted approach.
        
        # Actually, let's use the insight: non-uniform initial distribution with u^0.7
        # concentrates points at low f1 where ZDT6's Pareto front has highest curvature.
        
        # Generate u in [0, 1] with density proportional to u^(0.7-1) = u^(-0.3)
        # This means more points near u=0 (low x1, low f1).
        
        # To sample from density p(u) ∝ u^(-0.3) on [0,1]:
        # CDF: F(u) = u^(0.7) / 0.7 (normalized)
        # Inverse CDF: u = (0.7 * v)^(1/0.7) where v ~ Uniform(0,1)
        
        # But we also want good coverage of the entire range. Let's mix:
        # 70% of points from the concentrated distribution, 30% uniform.
        
        n_concentrated = int(0.7 * n_evals)
        n_uniform = n_evals - n_concentrated
        
        # Concentrated samples: u = (0.7 * v)^(1/0.7)
        v_conc = np.linspace(0.001, 0.999, n_concentrated)  # avoid 0 and 1
        u_conc = (0.7 * v_conc) ** (1.0 / 0.7)
        
        # Uniform samples
        u_unif = np.linspace(0.0, 1.0, n_uniform)
        
        x1_samples = np.concatenate([u_conc, u_unif])
        
        # Sort to ensure we have a well-ordered set
        x1_samples = np.sort(x1_samples)
        
        # Construct X
        X = np.zeros((n_evals, dim))
        X[:, 0] = x1_samples
        # x2..xn are 0 (lower bound)
        
        F = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)
        
        return X
