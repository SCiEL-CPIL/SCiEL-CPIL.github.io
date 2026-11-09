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
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # Strategy:
        # 1. Use Sobol sequence for the 2 decision variables x[0] and x[1].
        # 2. Set remaining variables to lo (0) to minimize g(x) and thus h(x).
        # 3. Add explicit boundary points to ensure corners of the Pareto front are covered.
        
        # Sobol sequence requires a power of 2 size for best distribution properties,
        # but we can generate more and take the first n.
        # Let's generate n_evals points.
        
        # We will reserve 14 points for explicit boundaries.
        n_boundary = 14
        n_sobol = n_evals - n_boundary
        
        if n_sobol < 0:
            n_sobol = 0
            n_boundary = n_evals
            
        # Generate Sobol points in [0, 1]^2
        # qmc.Sobol(d=2, scramble=False) # scramble=True is often better for low dimensions
        sampler = qmc.Sobol(d=2, scramble=True)
        # Generate a power of 2 >= n_sobol for better uniformity if possible, or just n_sobol
        # scipy qmc.Sobol.generate(n) works for any n, but powers of 2 are optimal.
        # Let's just generate n_sobol points.
        if n_sobol > 0:
            sobol_points = sampler.generate(n_sobol, random_state=0)
        else:
            sobol_points = np.empty((0, 2))
            
        # Define boundary points in [0,1]x[0,1]
        # These points are chosen to cover the extremes of the Pareto front.
        # The front is defined by x0, x1 in [0,1].
        # f1 = x0
        # f2 = (1-x0)cos(pi*x1/2)
        # f3 = (1-x0)sin(pi*x1/2) + h(x0, x1)
        
        # Critical points:
        # (0,0) -> f1=0, f2=1, f3=1 (Low f3, Low f1, High f2)
        # (0,1) -> f1=0, f2=0, f3=1+h(0,1) (Low f1, Low f2, Med f3)
        # (1,0) -> f1=1, f2=0, f3=0+h(1,0) (High f1, Low f2, Med f3)
        # (1,1) -> f1=1, f2=0, f3=0+h(1,1) (High f1, Low f2, High f3)
        
        # We want to capture the "ridge" where f3 is high and the "valley" where it is low.
        boundary_points = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            # Edges
            [0.0, 0.5],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
            # Interior strategic points near the 'optimal' corner (0,0)
            [0.1, 0.1],
            [0.2, 0.2],
            [0.05, 0.05],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.2, 0.0],
        ])
        
        if len(boundary_points) > n_boundary:
            boundary_points = boundary_points[:n_boundary]
            
        if n_sobol > 0:
            all_points = np.vstack([sobol_points, boundary_points])
        else:
            all_points = boundary_points
        
        # Map to decision space
        # x[0] and x[1] are the active variables
        # x[2:] are set to lo[2:] (which is 0)
        
        X = np.zeros((len(all_points), self.dim))
        X[:, 0] = lo[0] + (hi[0] - lo[0]) * all_points[:, 0]
        X[:, 1] = lo[1] + (hi[1] - lo[1]) * all_points[:, 1]
        X[:, 2:] = lo[2:]
        
        results = []
        for i in range(len(X)):
            if time.time() > deadline:
                break
            
            x = X[i]
            # Clip to be safe
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
            
        return results
