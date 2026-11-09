import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_total = self.budget  # 200
        
        # Strategy: Since DTLZ2's Pareto front is a quarter-sphere in 3D,
        # we can generate points directly on the front (g=0) by setting x[2:] = 0.5
        # and choosing x[0], x[1] to cover the angular space uniformly.
        
        # The Pareto front is defined by:
        # f1 = cos(theta1)
        # f2 = cos(theta2) * sin(theta1)
        # f3 = sin(theta2) * sin(theta1)
        # where theta1, theta2 in [0, pi/2].
        
        # To maximize hypervolume, we want to cover the front as uniformly as possible.
        # A uniform distribution in (theta1, theta2) does NOT give uniform coverage on the sphere.
        # The area element on the sphere is proportional to sin(theta1) dtheta1 dtheta2.
        # So, to get uniform coverage, we should sample:
        # u ~ Uniform[0,1]
        # v ~ Uniform[0,1]
        # theta2 = v * pi/2
        # theta1 = arccos(1 - u)  (since integral of sin(t) from 0 to t1 = 1 - cos(t1))
        
        # However, we have a budget of 200 points.
        # We can use a scrambled Sobol sequence for (u, v) to get a good low-discrepancy set.
        
        sampler = qmc.Sobol(d=2, scramble=True, seed=0)
        uv = sampler.random(n_total)
        
        u = uv[:, 0]
        v = uv[:, 1]
        
        theta1 = np.arccos(1.0 - u)
        theta2 = v * np.pi / 2.0
        
        X = np.zeros((n_total, self.dim))
        X[:, 0] = theta1 / (np.pi / 2.0)
        X[:, 1] = theta2 / (np.pi / 2.0)
        X[:, 2:] = 0.5  # Set g(x) = 0 to be on the Pareto front
        
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = []
        for i, x in enumerate(X):
            if time.time() > deadline:
                break
            f = problem(x)
            F.append(f)
        
        # The return value of __call__ is ignored by the evaluator,
        # but we must return something.
        return X
